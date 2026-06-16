# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from llmeter.callbacks.base import Callback
from llmeter.callbacks.mlflow import MlflowCallback


class TestBase:
    def test_save_to_file_creates_valid_json(self, tmp_path):
        """save_to_file creates a valid JSON file with _class and _state."""
        cb = MlflowCallback(step=5, nested=True)
        path = tmp_path / "callback.json"
        cb.save_to_file(path)

        with open(path) as f:
            data = json.load(f)

        assert "_class" in data
        assert "_state" in data
        assert data["_class"] == "llmeter.callbacks.mlflow.MlflowCallback"
        assert data["_state"] == {"step": 5, "nested": True}

    def test_load_from_file_restores_callback(self, tmp_path):
        """load_from_file correctly restores a callback instance."""
        cb = MlflowCallback(step=3, nested=False)
        path = tmp_path / "callback.json"
        cb.save_to_file(path)

        restored = Callback.load_from_file(path)

        assert isinstance(restored, MlflowCallback)
        assert restored.step == 3
        assert restored.nested is False

    def test_getstate_setstate_roundtrip(self):
        """__getstate__ / __setstate__ round-trips correctly."""
        cb = MlflowCallback(step=7, nested=True)
        state = cb.__getstate__()

        restored = MlflowCallback.__new__(MlflowCallback)
        restored.__setstate__(state)

        assert restored.step == 7
        assert restored.nested is True
