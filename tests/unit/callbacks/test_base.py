# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pytest

from llmeter.callbacks.base import Callback


class _DummyCallback(Callback):
    """A minimal concrete callback for testing serialization."""

    def __init__(self, alpha: int = 1, beta: str = "hello"):
        self.alpha = alpha
        self.beta = beta


class TestCallbackSerialization:
    def test_to_dict_includes_callback_type(self):
        cb = _DummyCallback(alpha=42, beta="world")
        d = cb.to_dict()
        assert (
            d["_callback_type"]
            == f"{_DummyCallback.__module__}:{_DummyCallback.__qualname__}"
        )
        assert d["alpha"] == 42
        assert d["beta"] == "world"

    def test_to_dict_excludes_private_attrs(self):
        cb = _DummyCallback()
        cb._internal = "secret"
        d = cb.to_dict()
        assert "_internal" not in d

    def test_from_dict_round_trip(self):
        cb = _DummyCallback(alpha=7, beta="test")
        d = cb.to_dict()
        restored = Callback.from_dict(d)
        assert isinstance(restored, _DummyCallback)
        assert restored.alpha == 7
        assert restored.beta == "test"

    def test_from_dict_missing_callback_type_raises(self):
        with pytest.raises(ValueError, match="_callback_type"):
            Callback.from_dict({"alpha": 1})

    def test_to_json_round_trip(self):
        cb = _DummyCallback(alpha=99)
        json_str = cb.to_json()
        restored = Callback.from_json(json_str)
        assert isinstance(restored, _DummyCallback)
        assert restored.alpha == 99

    def test_save_and_load_from_file(self, tmp_path):
        cb = _DummyCallback(alpha=5, beta="file_test")
        file_path = tmp_path / "callback.json"
        cb.save_to_file(file_path)

        loaded = Callback.load_from_file(file_path)
        assert isinstance(loaded, _DummyCallback)
        assert loaded.alpha == 5
        assert loaded.beta == "file_test"

        # Verify the file is valid JSON with the type marker
        with open(file_path) as f:
            data = json.load(f)
        assert "_callback_type" in data
