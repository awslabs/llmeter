# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for loading *legacy-format* LLMeter data written by older releases.

This covers artifacts produced before the current ``dump_object`` serialization, across the three
data types that can be reloaded:

* **Run configs** (``run_config.json``) and **endpoint configs** - identified the endpoint class
  with an ``endpoint_type`` key (rather than the current ``__llmeter_class__`` envelope) and the
  tokenizer with a ``tokenizer_module`` key, and stored derived / read-only attributes such as
  ``provider`` alongside real constructor arguments. Modern endpoint constructors compute
  ``provider`` internally and reject it as a keyword argument.
* **Responses** (``responses.jsonl``, via ``Result.load``) - carried extra per-response fields that
  callbacks wrote directly onto the response (e.g. ``cost_InputTokens`` / ``cost_total``), plus a
  since-removed ``end_time`` field. Current ``InvocationResponse`` doesn't declare these, so a
  strict load would raise; they are now collected into ``annotations`` instead.

All of these previously broke loading of real, local data. These tests pin the backward-compatible
behavior so it can't silently regress on a minor-version release. If you intend to drop
legacy-format support, that is a breaking change and these tests should be updated deliberately.
"""

import json

import pytest

from llmeter.endpoints.base import Endpoint, _filter_legacy_ctor_kwargs
from llmeter.endpoints.bedrock import BedrockConverse, BedrockConverseStream
from llmeter.results import Result
from llmeter.runner import _RunConfig
from llmeter.tokenizers import DummyTokenizer


def _legacy_endpoint_config() -> dict:
    """A legacy endpoint config dict, matching the on-disk ``examples/outputs`` format.

    Note ``provider`` is present here but is *not* a parameter of
    ``BedrockConverseStream.__init__`` (it is derived internally).
    """
    return {
        "endpoint_name": "amazon bedrock",
        "model_id": "apac.amazon.nova-pro-v1:0",
        "provider": "bedrock",
        "region": "us-east-1",
        "endpoint_type": "BedrockConverseStream",
    }


def _legacy_run_config() -> dict:
    """A full legacy ``run_config.json`` payload, mirroring real ``examples/outputs`` data."""
    return {
        "endpoint": _legacy_endpoint_config(),
        "output_path": "outputs/apac.amazon.nova-pro-v1:0/20251210-0115",
        "tokenizer": {"tokenizer_module": "llmeter"},
        "clients": 20,
        "n_requests": 10,
        "payload": "outputs/apac.amazon.nova-pro-v1:0/20251210-0115/payload.jsonl",
        "run_name": "20251210-0115",
        "run_description": None,
        "timeout": 60,
        "callbacks": None,
    }


class TestLegacyEndpointConfigLoad:
    """Endpoint.load / load_from_file must accept legacy ``endpoint_type`` configs."""

    def test_load_drops_derived_provider_field(self):
        """A legacy dict carrying a derived ``provider`` must load, not raise TypeError."""
        endpoint = Endpoint.load(_legacy_endpoint_config())

        assert isinstance(endpoint, BedrockConverseStream)
        assert endpoint.model_id == "apac.amazon.nova-pro-v1:0"
        assert endpoint.region == "us-east-1"
        assert endpoint.endpoint_name == "amazon bedrock"
        # provider is derived by the constructor, and should round-trip to the same value
        assert endpoint.provider == "bedrock"

    def test_load_from_file_drops_derived_provider_field(self, tmp_path):
        """The file-based legacy loader must behave the same as the dict loader."""
        path = tmp_path / "endpoint.json"
        path.write_text(json.dumps(_legacy_endpoint_config()))

        endpoint = Endpoint.load_from_file(path)

        assert isinstance(endpoint, BedrockConverseStream)
        assert endpoint.model_id == "apac.amazon.nova-pro-v1:0"
        assert endpoint.region == "us-east-1"
        assert endpoint.provider == "bedrock"

    def test_load_missing_endpoint_type_still_raises(self):
        """Legacy loading still requires ``endpoint_type`` to identify the class."""
        config = _legacy_endpoint_config()
        del config["endpoint_type"]
        with pytest.raises(KeyError):
            Endpoint.load(config)


class TestLegacyRunConfigLoad:
    """_RunConfig.load must reconstruct endpoint + tokenizer from a legacy run_config.json."""

    def test_load_reconstructs_legacy_run_config(self, tmp_path):
        (tmp_path / "run_config.json").write_text(json.dumps(_legacy_run_config()))

        cfg = _RunConfig.load(tmp_path)

        # Endpoint: legacy endpoint_type -> concrete class, provider derived
        assert isinstance(cfg._endpoint, BedrockConverseStream)
        assert cfg._endpoint.model_id == "apac.amazon.nova-pro-v1:0"
        assert cfg._endpoint.region == "us-east-1"
        assert cfg._endpoint.provider == "bedrock"

        # Tokenizer: legacy tokenizer_module -> DummyTokenizer
        assert isinstance(cfg._tokenizer, DummyTokenizer)

        # Scalar fields survive unchanged
        assert cfg.clients == 20
        assert cfg.n_requests == 10
        assert cfg.timeout == 60
        assert cfg.run_name == "20251210-0115"

    def test_load_legacy_run_config_custom_filename(self, tmp_path):
        """The legacy loader honors a custom config file name."""
        (tmp_path / "my_config.json").write_text(json.dumps(_legacy_run_config()))

        cfg = _RunConfig.load(tmp_path, file_name="my_config.json")

        assert isinstance(cfg._endpoint, BedrockConverseStream)
        assert isinstance(cfg._tokenizer, DummyTokenizer)


class TestLegacyResponsesLoad:
    """Result.load must tolerate older responses.jsonl carrying extra per-response keys.

    Older LLMeter versions let callbacks (e.g. the cost callback) write extra fields such as
    ``cost_InputTokens`` / ``cost_total`` directly onto each response, and serialized them into
    ``responses.jsonl``. A couple of runs also carry a since-removed ``end_time`` response field.
    Current ``InvocationResponse`` doesn't declare these, so a strict ``cls(**data)`` load would
    raise ``TypeError``. These are collected into ``annotations`` instead, so the data is preserved
    and the file still loads.
    """

    def test_load_result_with_legacy_response_extra_keys(self, tmp_path):
        summary = {
            "total_requests": 2,
            "clients": 1,
            "n_requests": 2,
            "model_id": "some-model",
        }
        (tmp_path / "summary.json").write_text(json.dumps(summary))

        lines = [
            json.dumps(
                {
                    "response_text": "a",
                    "id": "r1",
                    "num_tokens_input": 10,
                    "num_tokens_output": 5,
                    "cost_InputTokens": 0.01,
                    "cost_OutputTokens": 0.02,
                    "cost_total": 0.03,
                }
            ),
            json.dumps(
                {
                    "response_text": "b",
                    "id": "r2",
                    "num_tokens_input": 20,
                    "num_tokens_output": 8,
                    "cost_total": 0.05,
                    "end_time": "2024-01-01T00:00:00Z",  # since-removed field
                }
            ),
        ]
        (tmp_path / "responses.jsonl").write_text("\n".join(lines) + "\n")

        # Previously raised TypeError: unexpected keyword argument 'cost_InputTokens'
        result = Result.load(tmp_path)

        assert len(result.responses) == 2
        r1, r2 = result.responses

        # Core fields are still parsed normally
        assert r1.num_tokens_input == 10
        assert r2.id == "r2"

        # Legacy extra keys are preserved under annotations rather than dropped
        assert r1.annotations == {
            "cost_InputTokens": 0.01,
            "cost_OutputTokens": 0.02,
            "cost_total": 0.03,
        }
        assert r2.annotations["cost_total"] == 0.05
        assert r2.annotations["end_time"] == "2024-01-01T00:00:00Z"

    def test_load_result_without_summary_recovers_legacy_responses(self, tmp_path):
        """Metadata recovery (no summary.json) also streams responses and must tolerate extras."""
        lines = [
            json.dumps(
                {
                    "response_text": "a",
                    "id": "r1",
                    "request_time": "2024-01-01T00:00:00Z",
                    "cost_total": 0.03,
                }
            ),
        ]
        (tmp_path / "responses.jsonl").write_text("\n".join(lines) + "\n")

        result = Result.load(tmp_path)  # no summary.json -> _recover_metadata path

        assert len(result.responses) == 1
        assert result.responses[0].annotations["cost_total"] == 0.03


class TestFilterLegacyCtorKwargs:
    """Unit tests for the constructor-kwarg filtering helper."""

    def test_drops_keys_not_accepted_by_constructor(self):
        config = {
            "model_id": "m",
            "region": "us-east-1",
            "provider": "bedrock",  # not a BedrockConverseStream.__init__ param
        }
        filtered = _filter_legacy_ctor_kwargs(BedrockConverseStream, config)

        assert "provider" not in filtered
        assert filtered == {"model_id": "m", "region": "us-east-1"}
        # Original dict is not mutated
        assert "provider" in config

    def test_keeps_keys_the_constructor_accepts(self):
        config = {"model_id": "m", "endpoint_name": "n", "region": "us-east-1"}
        filtered = _filter_legacy_ctor_kwargs(BedrockConverse, config)
        assert filtered == config

    def test_passthrough_when_constructor_accepts_var_keyword(self):
        class AcceptsKwargs:
            def __init__(self, model_id, **kwargs):
                self.model_id = model_id

        config = {"model_id": "m", "anything": 1, "provider": "x"}
        # **kwargs means every key is acceptable, so nothing is dropped
        assert _filter_legacy_ctor_kwargs(AcceptsKwargs, config) == config
