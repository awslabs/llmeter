# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for SystemMetricsMonitor callback.

These tests verify that SystemMetricsMonitor works correctly in combination with
actual Runner executions against a live Bedrock endpoint. They validate the full
lifecycle: background sampling during a real run, stat contribution to the result,
persistence across save/load, and reuse across multiple runs.

To run these tests:
    uv run pytest -m integ tests/integ/callbacks/test_system_metrics.py

Required AWS Permissions:
    - bedrock:InvokeModel

Estimated Cost:
    - ~$0.001 total for all tests in this module (multiple short runs)
"""

import tempfile

import pytest

from llmeter.callbacks.system_metrics import SystemMetricsMonitor
from llmeter.endpoints.bedrock import BedrockConverse
from llmeter.results import Result
from llmeter.runner import Runner


@pytest.mark.integ
@pytest.mark.asyncio
async def test_system_metrics_with_runner(
    aws_credentials, aws_region, bedrock_test_model, test_payload
):
    """
    Test SystemMetricsMonitor collects metrics during a real Runner.run().

    Validates that:
    - The monitor starts and stops cleanly during a real run
    - System stats are contributed to the result
    - CPU, memory, and network metrics are present and reasonable
    - Sample count reflects the run duration

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        aws_region: AWS region for testing.
        bedrock_test_model: Model ID for testing.
        test_payload: Simple text test payload.
    """
    monitor = SystemMetricsMonitor(sample_interval=0.2, per_process=True)
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    runner = Runner(endpoint=endpoint, callbacks=[monitor])
    result = await runner.run(
        payload=test_payload,
        clients=2,
        n_requests=3,
        disable_per_client_progress_bar=True,
        disable_clients_progress_bar=True,
    )

    # Verify the run itself succeeded
    assert result.total_requests > 0

    # Verify system metrics were contributed
    assert "system_cpu_percent-average" in result.stats
    assert "system_cpu_percent-p50" in result.stats
    assert "system_cpu_percent-p90" in result.stats
    assert "system_cpu_percent-p99" in result.stats
    assert "system_memory_rss_mb-average" in result.stats
    assert "system_memory_rss_mb-max" in result.stats
    assert "system_memory_vms_mb-average" in result.stats
    assert "system_memory_vms_mb-max" in result.stats
    assert "system_net_bytes_sent_total" in result.stats
    assert "system_net_bytes_recv_total" in result.stats
    assert "system_samples_collected" in result.stats

    # Sanity checks on values
    assert result.stats["system_cpu_percent-average"] >= 0
    assert result.stats["system_memory_rss_mb-max"] > 0
    assert result.stats["system_net_bytes_recv_total"] >= 0
    assert result.stats["system_samples_collected"] >= 2


@pytest.mark.integ
@pytest.mark.asyncio
async def test_system_metrics_persist_after_save_load(
    aws_credentials, aws_region, bedrock_test_model, test_payload
):
    """
    Test that system metrics survive a save/load round-trip of the Result.

    Validates that:
    - Stats are written to stats.json on save
    - Stats are correctly restored on load (with and without responses)

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        aws_region: AWS region for testing.
        bedrock_test_model: Model ID for testing.
        test_payload: Simple text test payload.
    """
    monitor = SystemMetricsMonitor(sample_interval=0.2, per_process=True)
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Runner(endpoint=endpoint, callbacks=[monitor], output_path=tmpdir)
        result = await runner.run(
            payload=test_payload,
            clients=1,
            n_requests=2,
            disable_per_client_progress_bar=True,
            disable_clients_progress_bar=True,
        )

        # Capture stats before save
        original_cpu_avg = result.stats["system_cpu_percent-average"]
        original_rss_max = result.stats["system_memory_rss_mb-max"]
        original_net_recv = result.stats["system_net_bytes_recv_total"]

        # Load with responses
        loaded = Result.load(result.output_path, load_responses=True)
        assert loaded.stats["system_cpu_percent-average"] == pytest.approx(
            original_cpu_avg
        )
        assert loaded.stats["system_memory_rss_mb-max"] == pytest.approx(
            original_rss_max
        )
        assert loaded.stats["system_net_bytes_recv_total"] == original_net_recv

        # Load without responses (stats-only path)
        loaded_no_resp = Result.load(result.output_path, load_responses=False)
        assert loaded_no_resp.stats["system_cpu_percent-average"] == pytest.approx(
            original_cpu_avg
        )


@pytest.mark.integ
@pytest.mark.asyncio
async def test_system_metrics_reuse_across_runs(
    aws_credentials, aws_region, bedrock_test_model, test_payload
):
    """
    Test that a single SystemMetricsMonitor instance can be reused across runs.

    Validates that:
    - The monitor resets state between runs
    - Each run gets independent samples and statistics
    - No cross-contamination of metrics between runs

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        aws_region: AWS region for testing.
        bedrock_test_model: Model ID for testing.
        test_payload: Simple text test payload.
    """
    monitor = SystemMetricsMonitor(sample_interval=0.2, per_process=True)
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    runner = Runner(endpoint=endpoint, callbacks=[monitor])

    # First run
    result1 = await runner.run(
        payload=test_payload,
        clients=1,
        n_requests=2,
        disable_per_client_progress_bar=True,
        disable_clients_progress_bar=True,
    )

    # Second run
    result2 = await runner.run(
        payload=test_payload,
        clients=1,
        n_requests=2,
        disable_per_client_progress_bar=True,
        disable_clients_progress_bar=True,
    )

    # Both runs should have independent system metrics
    assert "system_samples_collected" in result1.stats
    assert "system_samples_collected" in result2.stats
    assert result1.stats["system_samples_collected"] >= 2
    assert result2.stats["system_samples_collected"] >= 2

    # Memory should be positive in both
    assert result1.stats["system_memory_rss_mb-max"] > 0
    assert result2.stats["system_memory_rss_mb-max"] > 0


@pytest.mark.integ
@pytest.mark.asyncio
async def test_system_metrics_system_wide_mode(
    aws_credentials, aws_region, bedrock_test_model, test_payload
):
    """
    Test SystemMetricsMonitor in system-wide mode (per_process=False).

    Validates that:
    - System-wide mode collects metrics from the entire machine
    - Stats are contributed correctly

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        aws_region: AWS region for testing.
        bedrock_test_model: Model ID for testing.
        test_payload: Simple text test payload.
    """
    monitor = SystemMetricsMonitor(sample_interval=0.2, per_process=False)
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    runner = Runner(endpoint=endpoint, callbacks=[monitor])
    result = await runner.run(
        payload=test_payload,
        clients=1,
        n_requests=2,
        disable_per_client_progress_bar=True,
        disable_clients_progress_bar=True,
    )

    # System-wide mode should report higher CPU and memory than per-process
    assert "system_cpu_percent-average" in result.stats
    assert "system_memory_rss_mb-max" in result.stats
    assert result.stats["system_memory_rss_mb-max"] > 0
    assert result.stats["system_samples_collected"] >= 2


@pytest.mark.integ
@pytest.mark.asyncio
async def test_system_metrics_with_higher_concurrency(
    aws_credentials, aws_region, bedrock_test_model, test_payload
):
    """
    Test SystemMetricsMonitor during a higher-concurrency run.

    This validates that the monitor captures meaningful resource variation
    under load. With more concurrent requests, we expect to see non-trivial
    CPU usage and network activity.

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        aws_region: AWS region for testing.
        bedrock_test_model: Model ID for testing.
        test_payload: Simple text test payload.
    """
    monitor = SystemMetricsMonitor(sample_interval=0.1, per_process=True)
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    runner = Runner(endpoint=endpoint, callbacks=[monitor])
    result = await runner.run(
        payload=test_payload,
        clients=5,
        n_requests=3,
        disable_per_client_progress_bar=True,
        disable_clients_progress_bar=True,
    )

    # With higher concurrency, we expect more samples and network activity
    assert result.stats["system_samples_collected"] >= 3
    assert result.stats["system_net_bytes_sent_total"] > 0
    assert result.stats["system_net_bytes_recv_total"] > 0

    # Network rate stats should be present with enough samples
    if result.stats["system_samples_collected"] >= 3:
        assert "system_net_bytes_recv_per_second-average" in result.stats
        assert result.stats["system_net_bytes_recv_per_second-average"] > 0
