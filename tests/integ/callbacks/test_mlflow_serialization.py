# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for MlflowCallback serialization.

These tests use the real mlflow library (no mocks) to verify that
MlflowCallback round-trips correctly through all serialization paths.
"""

import json

import pytest

mlflow = pytest.importorskip("mlflow", reason="mlflow not installed")

from llmeter.callbacks.base import Callback  # noqa: E402
from llmeter.callbacks.mlflow import MlflowCallback  # noqa: E402
from llmeter.json_utils import llmeter_default_deserializer, llmeter_default_serializer  # noqa: E402


class TestMlflowCallbackSerializationInteg:
    """Integration tests using the real mlflow library — no mocks."""

    def test_to_dict_round_trip(self):
        """to_dict() / from_dict() round-trip with real mlflow."""
        cb = MlflowCallback(step=3, nested=True)
        d = cb.to_dict()

        assert d == {
            "__llmeter_class__": "llmeter.callbacks.mlflow:MlflowCallback",
            "step": 3,
            "nested": True,
        }

        restored = MlflowCallback.from_dict(d)
        assert isinstance(restored, MlflowCallback)
        assert restored.step == 3
        assert restored.nested is True

    def test_to_dict_defaults(self):
        """Default parameter values serialize correctly."""
        cb = MlflowCallback()
        d = cb.to_dict()
        assert d["step"] is None
        assert d["nested"] is False

    def test_base_class_dispatch(self):
        """Callback.from_dict() dispatches to MlflowCallback via __llmeter_class__."""
        d = {
            "__llmeter_class__": "llmeter.callbacks.mlflow:MlflowCallback",
            "step": 5,
            "nested": False,
        }
        restored = Callback.from_dict(d)
        assert type(restored) is MlflowCallback
        assert restored.step == 5
        assert restored.nested is False

    def test_json_string_round_trip(self):
        """Round-trip through json.dumps / json.loads with LLMeter hooks."""
        cb = MlflowCallback(step=10, nested=True)
        encoded = json.dumps(cb.to_dict(), default=llmeter_default_serializer)
        decoded = json.loads(encoded, object_hook=llmeter_default_deserializer)

        assert type(decoded) is MlflowCallback
        assert decoded.step == 10
        assert decoded.nested is True

    def test_save_and_load_file(self, tmp_path):
        """Round-trip through save_to_file / load_from_file on disk."""
        cb = MlflowCallback(step=7, nested=True)
        path = tmp_path / "mlflow_callback.json"
        cb.save_to_file(path)

        # Verify raw JSON on disk
        with open(path) as f:
            raw = json.load(f)
        assert raw["__llmeter_class__"] == "llmeter.callbacks.mlflow:MlflowCallback"
        assert raw["step"] == 7
        assert raw["nested"] is True

        # Load back via the base class static method
        restored = Callback.load_from_file(path)
        assert type(restored) is MlflowCallback
        assert restored.step == 7
        assert restored.nested is True

    def test_save_and_load_none_step(self, tmp_path):
        """step=None survives the file round-trip."""
        cb = MlflowCallback(step=None, nested=False)
        path = tmp_path / "mlflow_none_step.json"
        cb.save_to_file(path)

        restored = Callback.load_from_file(path)
        assert type(restored) is MlflowCallback
        assert restored.step is None
        assert restored.nested is False
