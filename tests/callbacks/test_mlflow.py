# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from llmeter.callbacks.mlflow import MlflowCallback
from llmeter.results import Result


@pytest.fixture
def sample_result():
    result = Result(
        responses=[],
        total_requests=100,
        clients=5,
        n_requests=20,
        model_id="test-model",
        output_path="test/path",  # type: ignore
        endpoint_name="test-endpoint",
        provider="test-provider",
        run_name="test-run",
        run_description="test description",
    )
    return result


@pytest.fixture
def mlflow_callback():
    with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
        mock_mlflow.__version__ = "2.0.0"
        callback = MlflowCallback(step=1)
        yield callback


@pytest.fixture
def mlflow_callback_nested():
    with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
        mock_mlflow.__version__ = "2.0.0"
        callback = MlflowCallback(step=1, nested=True)
        yield callback


@pytest.mark.asyncio
async def test_log_llmeter_run(mlflow_callback, sample_result):
    with patch("llmeter.callbacks.mlflow.mlflow.log_params") as mock_log_params, patch(
        "llmeter.callbacks.mlflow.mlflow.log_metrics"
    ) as mock_log_metrics, patch(
        "llmeter.results.Result.stats", new_callable=PropertyMock
    ) as mock_stats:
        mock_stats.return_value = {
            "latency_mean": 0.5,
            "latency_p50": 0.4,
            "latency_p90": 0.6,
            "latency_p99": 0.8,
            "requests_per_second": 10.0,
        }

        await mlflow_callback._log_llmeter_run(sample_result)

        # Verify parameters logging
        expected_params = {
            "total_requests": 100,
            "clients": 5,
            "n_requests": 20,
            "model_id": "test-model",
            "output_path": "test/path",
            "endpoint_name": "test-endpoint",
            "provider": "test-provider",
            "run_name": "test-run",
            "run_description": "test description",
        }
        mock_log_params.assert_called_once_with(expected_params, synchronous=False)

        # Verify metrics logging
        expected_metrics = {
            "latency_mean": 0.5,
            "latency_p50": 0.4,
            "latency_p90": 0.6,
            "latency_p99": 0.8,
            "requests_per_second": 10.0,
        }
        mock_log_metrics.assert_called_once_with(
            expected_metrics, step=1, synchronous=False
        )


@pytest.mark.asyncio
async def test_log_nested_run(mlflow_callback_nested, sample_result):
    with patch("llmeter.callbacks.mlflow.mlflow.start_run") as mock_start_run, patch(
        "llmeter.callbacks.mlflow.mlflow.log_params"
    ) as mock_log_params, patch(
        "llmeter.callbacks.mlflow.mlflow.log_metrics"
    ) as mock_log_metrics:
        mock_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_context

        await mlflow_callback_nested._log_nested_run(sample_result)

        # Verify nested run was started
        mock_start_run.assert_called_once_with(run_name="test-run", nested=True)

        # Verify parameters and metrics were logged
        mock_log_params.assert_called_once()
        mock_log_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_after_run_nested(mlflow_callback_nested, sample_result):
    with patch.object(mlflow_callback_nested, "_log_nested_run") as mock_log_nested:
        await mlflow_callback_nested.after_run(sample_result)
        mock_log_nested.assert_called_once_with(sample_result)


@pytest.mark.asyncio
async def test_after_run_non_nested(mlflow_callback, sample_result):
    with patch.object(mlflow_callback, "_log_llmeter_run") as mock_log_run:
        await mlflow_callback.after_run(sample_result)
        mock_log_run.assert_called_once_with(sample_result)


def test_initialization():
    """Test that MlflowCallback raises ImportError when mlflow is not available."""
    # Test that the ImportError is raised correctly when mlflow is not available
    # We need to patch the mlflow import to simulate it not being available
    from llmeter.utils import DeferredError

    with patch(
        "llmeter.callbacks.mlflow.mlflow",
        DeferredError(
            "Please install mlflow (or mlflow-skinny) to use the MlflowCallback"
        ),
    ):
        with pytest.raises(ImportError, match="Please install mlflow"):
            MlflowCallback(step=5, nested=True)


def test_initialization_with_mlflow():
    """Test that MlflowCallback initializes correctly when mlflow is available."""
    with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
        mock_mlflow.__version__ = "2.0.0"
        callback = MlflowCallback(step=5, nested=True)
        assert callback.step == 5
        assert callback.nested is True


# @pytest.mark.asyncio
# async def test_load_from_file():
#     # Test the placeholder method
#     result = await MlflowCallback._load_from_file("test/path")
#     assert result is None


# def test_save_to_file():
#     # Test the placeholder method
#     callback = MlflowCallback()
#     result = callback.save_to_file()
#     assert result is None
