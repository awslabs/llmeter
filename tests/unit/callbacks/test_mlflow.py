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
    with (
        patch("llmeter.callbacks.mlflow.mlflow.log_params") as mock_log_params,
        patch("llmeter.callbacks.mlflow.mlflow.log_metrics") as mock_log_metrics,
        patch("llmeter.results.Result.stats", new_callable=PropertyMock) as mock_stats,
    ):
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
    with (
        patch("llmeter.callbacks.mlflow.mlflow.start_run") as mock_start_run,
        patch("llmeter.callbacks.mlflow.mlflow.log_params") as mock_log_params,
        patch("llmeter.callbacks.mlflow.mlflow.log_metrics") as mock_log_metrics,
    ):
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
        DeferredError(ImportError()),
    ):
        with pytest.raises(ImportError):
            MlflowCallback(step=5, nested=True)


def test_initialization_with_mlflow():
    """Test that MlflowCallback initializes correctly when mlflow is available."""
    with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
        mock_mlflow.__version__ = "2.0.0"
        callback = MlflowCallback(step=5, nested=True)
        assert callback.step == 5
        assert callback.nested is True


class TestMlflowCallbackSerialization:
    """Tests for MlflowCallback serialization and deserialization."""

    def _make_callback(self, **kwargs):
        """Create an MlflowCallback with mlflow mocked out."""
        with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.__version__ = "2.0.0"
            return MlflowCallback(**kwargs)

    def test_to_dict_includes_class_marker(self):
        """to_dict() includes __llmeter_class__ with the correct module:ClassName."""
        cb = self._make_callback(step=3, nested=True)
        d = cb.to_dict()
        assert d["__llmeter_class__"] == "llmeter.callbacks.mlflow:MlflowCallback"
        assert d["step"] == 3
        assert d["nested"] is True

    def test_to_dict_default_values(self):
        """to_dict() correctly serializes default parameter values."""
        cb = self._make_callback()
        d = cb.to_dict()
        assert d["__llmeter_class__"] == "llmeter.callbacks.mlflow:MlflowCallback"
        assert d["step"] is None
        assert d["nested"] is False

    def test_to_dict_only_has_expected_keys(self):
        """to_dict() produces exactly the keys we expect — no extras leak in."""
        cb = self._make_callback(step=1, nested=False)
        d = cb.to_dict()
        assert set(d.keys()) == {"__llmeter_class__", "step", "nested"}

    def test_from_dict_round_trip(self):
        """from_dict() reconstructs an MlflowCallback from to_dict() output."""
        cb = self._make_callback(step=5, nested=True)
        d = cb.to_dict()
        with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.__version__ = "2.0.0"
            restored = MlflowCallback.from_dict(d)
        assert isinstance(restored, MlflowCallback)
        assert restored.step == 5
        assert restored.nested is True

    def test_from_dict_without_class_marker(self):
        """from_dict() on the concrete class works without __llmeter_class__."""
        with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.__version__ = "2.0.0"
            restored = MlflowCallback.from_dict({"step": 2, "nested": False})
        assert isinstance(restored, MlflowCallback)
        assert restored.step == 2
        assert restored.nested is False

    def test_from_dict_via_base_class(self):
        """Callback.from_dict() dispatches to MlflowCallback via __llmeter_class__."""
        from llmeter.callbacks.base import Callback

        d = {
            "__llmeter_class__": "llmeter.callbacks.mlflow:MlflowCallback",
            "step": 7,
            "nested": True,
        }
        with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.__version__ = "2.0.0"
            restored = Callback.from_dict(d)
        assert isinstance(restored, MlflowCallback)
        assert restored.step == 7
        assert restored.nested is True

    def test_json_round_trip(self):
        """Round-trip through JSON string serialization."""
        import json

        from llmeter.json_utils import (
            llmeter_default_deserializer,
            llmeter_default_serializer,
        )

        cb = self._make_callback(step=10, nested=True)
        encoded = json.dumps(cb.to_dict(), default=llmeter_default_serializer)
        with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.__version__ = "2.0.0"
            decoded = json.loads(encoded, object_hook=llmeter_default_deserializer)
        assert isinstance(decoded, MlflowCallback)
        assert decoded.step == 10
        assert decoded.nested is True

    def test_save_and_load_from_file(self, tmp_path):
        """Round-trip through save_to_file / load_from_file."""
        from llmeter.callbacks.base import Callback

        cb = self._make_callback(step=4, nested=False)
        path = tmp_path / "mlflow_cb.json"
        cb.save_to_file(path)

        # Verify the JSON on disk
        import json

        with open(path) as f:
            raw = json.load(f)
        assert raw["__llmeter_class__"] == "llmeter.callbacks.mlflow:MlflowCallback"
        assert raw["step"] == 4
        assert raw["nested"] is False

        # Load it back
        with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.__version__ = "2.0.0"
            restored = Callback.load_from_file(path)
        assert isinstance(restored, MlflowCallback)
        assert restored.step == 4
        assert restored.nested is False

    def test_save_and_load_with_none_step(self, tmp_path):
        """Round-trip preserves step=None correctly."""
        from llmeter.callbacks.base import Callback

        cb = self._make_callback(step=None, nested=True)
        path = tmp_path / "mlflow_cb_none_step.json"
        cb.save_to_file(path)

        with patch("llmeter.callbacks.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.__version__ = "2.0.0"
            restored = Callback.load_from_file(path)
        assert isinstance(restored, MlflowCallback)
        assert restored.step is None
        assert restored.nested is True
