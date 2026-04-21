# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest

from llmeter.callbacks.base import Callback
from llmeter.callbacks.cost.model import CostModel


class DummyCallback(Callback):
    """A minimal concrete Callback for testing serialization."""

    def __init__(self, alpha: int = 1, beta: str = "hello"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["alpha"] = self.alpha
        result["beta"] = self.beta
        return result

    @classmethod
    def from_dict(cls, raw: dict, **kwargs) -> "DummyCallback":
        data = {**raw}
        data.pop("__llmeter_class__", None)
        return cls(**data, **kwargs)


class TestCallbackSerialization:
    def test_to_dict_includes_class_marker(self):
        """to_dict() includes __llmeter_class__ with module:ClassName format."""
        cb = DummyCallback(alpha=42, beta="world")
        d = cb.to_dict()
        assert "__llmeter_class__" in d
        assert (
            d["__llmeter_class__"]
            == f"{DummyCallback.__module__}:{DummyCallback.__name__}"
        )
        assert d["alpha"] == 42
        assert d["beta"] == "world"

    def test_from_dict_round_trip(self):
        """from_dict() reconstructs the correct subclass from a to_dict() output."""
        cb = DummyCallback(alpha=7, beta="test")
        d = cb.to_dict()
        restored = Callback.from_dict(d)
        assert isinstance(restored, DummyCallback)
        assert restored.alpha == 7
        assert restored.beta == "test"

    def test_from_dict_on_concrete_subclass(self):
        """Calling from_dict on a concrete subclass works without __llmeter_class__."""
        d = {"alpha": 99, "beta": "direct"}
        restored = DummyCallback.from_dict(d)
        assert isinstance(restored, DummyCallback)
        assert restored.alpha == 99

    def test_from_dict_base_class_without_marker_raises(self):
        """Calling Callback.from_dict without __llmeter_class__ raises ValueError."""
        with pytest.raises(ValueError, match="__llmeter_class__"):
            Callback.from_dict({"some_key": "some_value"})

    def test_save_and_load_from_file(self):
        """Round-trip through save_to_file / load_from_file."""
        cb = DummyCallback(alpha=5, beta="file_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "callback.json"
            cb.save_to_file(path)

            # Verify the JSON on disk has the class marker
            with open(path) as f:
                raw = json.load(f)
            assert (
                raw["__llmeter_class__"]
                == f"{DummyCallback.__module__}:{DummyCallback.__name__}"
            )

            # Load it back
            restored = Callback.load_from_file(path)
            assert isinstance(restored, DummyCallback)
            assert restored.alpha == 5
            assert restored.beta == "file_test"

    def test_json_round_trip(self):
        """Round-trip through JSON string serialization."""
        from llmeter.json_utils import (
            llmeter_default_deserializer,
            llmeter_default_serializer,
        )

        cb = DummyCallback(alpha=3, beta="json")
        encoded = json.dumps(cb.to_dict(), default=llmeter_default_serializer)
        decoded = json.loads(encoded, object_hook=llmeter_default_deserializer)
        assert isinstance(decoded, DummyCallback)
        assert decoded.alpha == 3
        assert decoded.beta == "json"


class TestCostModelCallbackSerialization:
    def test_cost_model_to_dict_has_class_marker(self):
        """CostModel.to_dict() includes __llmeter_class__ alongside _type."""
        model = CostModel(request_dims=[], run_dims=[])
        d = model.to_dict()
        assert "__llmeter_class__" in d
        assert d["__llmeter_class__"] == "llmeter.callbacks.cost.model:CostModel"
        assert d["_type"] == "CostModel"

    def test_cost_model_round_trip_via_callback_from_dict(self):
        """CostModel can be reconstructed via Callback.from_dict()."""
        spec = {
            "__llmeter_class__": "llmeter.callbacks.cost.model:CostModel",
            "_type": "CostModel",
            "request_dims": {
                "TokensIn": {"_type": "InputTokens", "price_per_million": 30},
            },
            "run_dims": {
                "ComputeSeconds": {"_type": "EndpointTime", "price_per_hour": 50},
            },
        }
        model = Callback.from_dict(spec)
        assert isinstance(model, CostModel)
        assert model.request_dims["TokensIn"].price_per_million == 30
        assert model.run_dims["ComputeSeconds"].price_per_hour == 50

    def test_cost_model_save_load_file(self):
        """CostModel round-trips through save_to_file / load_from_file."""
        from llmeter.callbacks.cost.dimensions import EndpointTime, InputTokens

        model = CostModel(
            request_dims=[InputTokens(price_per_million=15)],
            run_dims=[EndpointTime(price_per_hour=25)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cost_model.json"
            model.save_to_file(path)

            restored = Callback.load_from_file(path)
            assert isinstance(restored, CostModel)
            assert restored.request_dims["InputTokens"].price_per_million == 15
            assert restored.run_dims["EndpointTime"].price_per_hour == 25
