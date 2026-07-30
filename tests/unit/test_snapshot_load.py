# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests that load *static, synthetic* result snapshots from disk.

Unlike the other load tests (which build data in-memory at test time), these load committed
fixture files under ``tests/unit/fixtures/result_snapshots/``. They pin the on-disk de/serialization
contract across the whole loading pipeline - ``Result.load``, ``_RunConfig.load``,
``LoadTestResult.load`` and the endpoint / tokenizer / callback restoration underneath - so format
drift can't silently break loading of previously-saved runs.

The fixtures are deliberately synthetic (fake model IDs like ``synthetic.test-model-v1``,
lorem-ipsum responses, timestamps in the past) so nobody mistakes them for real benchmark data.
They cover a base "happy path" modern-format snapshot plus variations: legacy formats, an
interrupted run, error/annotation edge cases, and a multi-concurrency load test.

To regenerate the fixtures (e.g. after an intentional format change), run:
    uv run python tests/unit/fixtures/_generate_fixtures.py
"""

from datetime import datetime, timezone

import httpx
import pytest

from llmeter.callbacks.cost.dimensions import InputTokens, OutputTokens
from llmeter.callbacks.cost.model import CostModel
from llmeter.callbacks.mlflow import MlflowCallback
from llmeter.endpoints.bedrock import BedrockConverse, BedrockConverseStream
from llmeter.endpoints.openai import OpenAICompletionStreamEndpoint
from llmeter.experiments import LoadTestResult
from llmeter.results import Result
from llmeter.runner import _RunConfig
from llmeter.tokenizers import DummyTokenizer


class TestBaseLoad:
    """The modern-format baseline: OpenAI endpoint + CostModel + MlflowCallback."""

    @pytest.fixture
    def base_dir(self, snapshots_dir):
        return snapshots_dir / "base"

    def test_result_loads(self, base_dir):
        result = Result.load(base_dir)
        assert len(result.responses) == 5
        assert result.total_requests == 5
        assert result.clients == 2
        assert result.n_requests == 3
        assert result.model_id == "synthetic.test-model-v1"
        assert result.run_name == "synthetic-base-test"

    def test_null_output_path_filled_from_load_path(self, base_dir):
        """A null output_path in summary.json must be filled with the real load path.

        This is what makes a relocated/copied result directory reloadable - the on-disk
        path won't match wherever the data now lives.
        """
        result = Result.load(base_dir)
        assert result.output_path is not None
        assert str(base_dir) in str(result.output_path)

    def test_timestamps_parsed_as_datetime(self, base_dir):
        result = Result.load(base_dir)
        assert result.start_time == datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert result.end_time == datetime(2025, 1, 15, 10, 0, 4, tzinfo=timezone.utc)
        assert isinstance(result.responses[0].request_time, datetime)

    def test_stats_match_frozen_values(self, base_dir):
        """Stats loaded from stats.json (no responses) should match the frozen values."""
        result = Result.load(base_dir, load_responses=False)
        stats = result.stats
        assert stats["total_requests"] == 5
        assert stats["failed_requests"] == 0
        assert stats["total_input_tokens"] == 185
        assert stats["total_output_tokens"] == 386
        assert stats["time_to_first_token-p50"] == pytest.approx(0.415)
        assert stats["requests_per_minute"] == pytest.approx(66.35700066357)

    def test_response_optional_token_fields(self, base_dir):
        result = Result.load(base_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-base-002"].num_tokens_input_cached == 12
        assert by_id["synth-base-003"].num_tokens_output_reasoning == 8

    # --- run_config.json / endpoint / tokenizer / callback restoration ---

    @pytest.fixture
    def base_config(self, base_dir):
        return _RunConfig.load(base_dir)

    def test_endpoint_restored(self, base_config):
        ep = base_config._endpoint
        assert isinstance(ep, OpenAICompletionStreamEndpoint)
        assert ep.model_id == "synthetic.test-model-v1"
        assert ep.endpoint_name == "synthetic-openai-stream"
        assert ep.provider == "openai"

    def test_endpoint_client_config_restored(self, base_config):
        """The OpenAI client's config surface (base_url, retries, timeout) must round-trip."""
        client = base_config._endpoint._client
        assert str(client.base_url) == "https://api.synthetic-provider.example.com/v1/"
        assert client.max_retries == 3
        assert isinstance(client.timeout, httpx.Timeout)
        assert client.timeout.connect == 5.0
        assert client.timeout.read == 60.0
        assert client.timeout.write == 5.0
        assert client.timeout.pool == 5.0

    def test_tokenizer_restored(self, base_config):
        assert isinstance(base_config._tokenizer, DummyTokenizer)

    def test_cost_model_callback_restored(self, base_config):
        cost_model = base_config.callbacks[0]
        assert isinstance(cost_model, CostModel)

        input_dim = cost_model.request_dims["InputTokens"]
        output_dim = cost_model.request_dims["OutputTokens"]
        assert isinstance(input_dim, InputTokens)
        assert isinstance(output_dim, OutputTokens)
        assert input_dim.price_per_million == 3.0
        assert output_dim.price_per_million == 15.0
        assert cost_model.run_dims == {}

    def test_mlflow_callback_restored(self, base_config):
        mlflow_cb = base_config.callbacks[1]
        assert isinstance(mlflow_cb, MlflowCallback)
        assert mlflow_cb.step is None
        assert mlflow_cb.nested is True


class TestLegacyV01EndpointTypeLoad:
    """Legacy v0.1 format: ``endpoint_type`` dispatch + flat cost annotations on responses."""

    @pytest.fixture
    def legacy_dir(self, snapshots_dir):
        return snapshots_dir / "legacy" / "v0_1_endpoint_type"

    def test_result_loads(self, legacy_dir):
        result = Result.load(legacy_dir)
        assert len(result.responses) == 5
        assert result.model_id == "synthetic.legacy-model-v1"
        assert result.run_name == "synthetic-legacy-test"

    def test_legacy_cost_keys_collected_into_annotations(self, legacy_dir):
        """Flat ``cost_*`` keys that current InvocationResponse doesn't declare go to annotations."""
        result = Result.load(legacy_dir)
        first = result.responses[0]
        assert first.annotations["cost_InputTokens"] == pytest.approx(0.000084)
        assert first.annotations["cost_OutputTokens"] == pytest.approx(0.00108)
        assert first.annotations["cost_total"] == pytest.approx(0.001164)

    def test_endpoint_restored_via_endpoint_type(self, legacy_dir):
        cfg = _RunConfig.load(legacy_dir)
        ep = cfg._endpoint
        assert isinstance(ep, BedrockConverse)
        assert ep.model_id == "synthetic.legacy-model-v1"
        assert ep.region == "us-west-2"
        # provider is derived by the constructor, not read from the (legacy) config dict
        assert ep.provider == "bedrock"

    def test_tokenizer_restored_via_tokenizer_module(self, legacy_dir):
        cfg = _RunConfig.load(legacy_dir)
        assert isinstance(cfg._tokenizer, DummyTokenizer)


class TestLegacyV01StrCallbacksLoad:
    """Legacy v0.1 data where callbacks were serialized as Python repr strings."""

    @pytest.fixture
    def strcb_dir(self, snapshots_dir):
        return snapshots_dir / "legacy" / "v0_1_str_callbacks"

    def test_result_loads(self, strcb_dir):
        result = Result.load(strcb_dir)
        assert len(result.responses) == 5
        assert result.model_id == "synthetic.strcb-model-v1"

    def test_config_loads_without_raising_on_str_callbacks(self, strcb_dir):
        """Unrestorable string callbacks must not break run_config loading."""
        cfg = _RunConfig.load(strcb_dir)
        assert isinstance(cfg._endpoint, BedrockConverseStream)
        # String callbacks are left untouched (not restored to objects, but not fatal either)
        assert cfg.callbacks is not None
        assert len(cfg.callbacks) == 2
        assert all(isinstance(cb, str) for cb in cfg.callbacks)


class TestLegacyV01InterruptedRunLoad:
    """A legacy-format interrupted run: no summary.json + top-level endpoint fields.

    Exercises the legacy branch of ``Result._recover_metadata``: older data kept endpoint
    fields (including the derived ``provider``) at the top level rather than nested in a
    ``__llmeter_state__`` envelope. Recovery must handle it - consistent with the legacy
    handling in ``_RunConfig.load`` / ``Endpoint.load`` - and warn that the format is
    deprecated.
    """

    @pytest.fixture
    def legacy_interrupted_dir(self, snapshots_dir):
        return snapshots_dir / "legacy" / "v0_1_interrupted"

    def test_no_summary_file_present(self, legacy_interrupted_dir):
        assert not (legacy_interrupted_dir / "summary.json").exists()

    def test_recovers_top_level_endpoint_fields(self, legacy_interrupted_dir):
        result = Result.load(legacy_interrupted_dir)
        assert result.model_id == "synthetic.legacy-interrupted-v1"
        assert result.endpoint_name == "synthetic-legacy-intr-endpoint"
        # Legacy configs persist the derived provider at the top level, so it recovers
        assert result.provider == "bedrock"
        assert result.clients == 2
        assert result.n_requests == 4

    def test_recovery_warns_legacy_format(self, legacy_interrupted_dir, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="llmeter.results"):
            Result.load(legacy_interrupted_dir)
        assert any("legacy data format" in r.message for r in caplog.records)


class TestInterruptedRunLoad:
    """A snapshot with no summary.json - exercises metadata recovery."""

    @pytest.fixture
    def interrupted_dir(self, snapshots_dir):
        return snapshots_dir / "interrupted_run"

    def test_no_summary_file_present(self, interrupted_dir):
        assert not (interrupted_dir / "summary.json").exists()

    def test_recovers_responses(self, interrupted_dir):
        result = Result.load(interrupted_dir)
        assert len(result.responses) == 5
        assert result.total_requests == 5

    def test_recovers_metadata_from_run_config(self, interrupted_dir):
        result = Result.load(interrupted_dir)
        assert result.clients == 4
        assert result.n_requests == 20
        assert result.run_name == "synthetic-interrupted-test"

    def test_recovers_endpoint_fields_from_modern_config(self, interrupted_dir):
        """Endpoint fields must be read from the modern ``__llmeter_state__`` envelope.

        A *current* interrupted run writes run_config.json via ``dump_object`` (nested
        state), so recovery must look inside ``__llmeter_state__`` and not just the top
        level. ``provider`` is derived by the endpoint constructor and not persisted, so
        it stays ``None`` on this lightweight recovery path (no endpoint instantiation).
        """
        result = Result.load(interrupted_dir)
        assert result.model_id == "synthetic.interrupted-model-v1"
        assert result.endpoint_name == "synthetic-interrupted-endpoint"
        assert result.provider is None

    def test_derives_timestamps_from_responses(self, interrupted_dir):
        result = Result.load(interrupted_dir)
        # request_time runs 16:30:02 .. 16:30:10 (i*2 for i in 1..5)
        assert result.start_time == datetime(
            2025, 1, 20, 16, 30, 2, tzinfo=timezone.utc
        )
        assert result.end_time == datetime(2025, 1, 20, 16, 30, 10, tzinfo=timezone.utc)

    def test_modern_config_recovery_does_not_warn_legacy(self, interrupted_dir, caplog):
        """The modern-format recovery path must not emit the legacy-format warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="llmeter.results"):
            Result.load(interrupted_dir)
        assert not any("legacy data format" in r.message for r in caplog.records)

    def test_computes_stats_from_responses(self, interrupted_dir):
        result = Result.load(interrupted_dir)
        stats = result.stats
        assert stats["failed_requests"] == 0
        assert "time_to_first_token-p50" in stats


class TestErrorsAndAnnotationsLoad:
    """Responses carrying errors, retries, cached tokens, and custom annotations."""

    @pytest.fixture
    def errors_dir(self, snapshots_dir):
        return snapshots_dir / "errors_and_annotations"

    def test_result_loads(self, errors_dir):
        result = Result.load(errors_dir)
        assert len(result.responses) == 5

    def test_errors_preserved(self, errors_dir):
        result = Result.load(errors_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-err-002"].error == "ThrottlingException: Rate exceeded"
        assert by_id["synth-err-004"].error == "ReadTimeoutError: timed out after 60s"
        assert by_id["synth-err-001"].error is None

    def test_failed_requests_counted(self, errors_dir):
        result = Result.load(errors_dir)
        assert result.stats["failed_requests"] == 2

    def test_retries_preserved(self, errors_dir):
        result = Result.load(errors_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-err-002"].retries == 2
        assert by_id["synth-err-005"].retries == 1
        assert by_id["synth-err-001"].retries == 0

    def test_custom_annotations_preserved(self, errors_dir):
        result = Result.load(errors_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-err-001"].annotations == {
            "custom_metric": 42.5,
            "experiment_tag": "baseline",
        }
        assert by_id["synth-err-005"].annotations["cache_hit_ratio"] == pytest.approx(
            0.667
        )

    def test_cached_input_tokens_preserved(self, errors_dir):
        result = Result.load(errors_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-err-005"].num_tokens_input_cached == 30


class TestLoadTestResultLoad:
    """A multi-concurrency snapshot loaded via LoadTestResult.load."""

    @pytest.fixture
    def load_test_dir(self, snapshots_dir):
        return snapshots_dir / "load_test"

    def test_loads_all_concurrency_levels(self, load_test_dir):
        lt = LoadTestResult.load(load_test_dir)
        assert set(lt.results.keys()) == {1, 3}

    def test_sub_results_have_correct_data(self, load_test_dir):
        lt = LoadTestResult.load(load_test_dir)
        assert len(lt.results[1].responses) == 5
        assert len(lt.results[3].responses) == 5
        assert lt.results[1].clients == 1
        assert lt.results[3].clients == 3

    def test_sub_result_stats_populated(self, load_test_dir):
        lt = LoadTestResult.load(load_test_dir)
        for clients in (1, 3):
            stats = lt.results[clients].stats
            assert stats["total_requests"] == 5
            assert "time_to_first_token-p50" in stats

    def test_load_without_responses(self, load_test_dir):
        lt = LoadTestResult.load(load_test_dir, load_responses=False)
        assert lt.results[1].responses == []
        assert lt.results[1].stats["total_requests"] == 5
