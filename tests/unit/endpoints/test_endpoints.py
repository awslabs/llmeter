# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

import llmeter
import llmeter.endpoints
from llmeter.endpoints.base import (
    Endpoint,
    InvocationResponse,
    backfill_reasoning_type_from_token_counts,
    infer_reasoning_visibility_from_model_id,
)

# Tests for InvocationResponse


def test_invocation_response_initialization():
    response = InvocationResponse(
        id="test_id",
        response_text="Hello, world!",
        input_prompt="Say hello",
        time_to_first_token=0.1,
        time_to_last_token=0.5,
        num_tokens_input=3,
        num_tokens_output=2,
        time_per_output_token=0.2,
    )
    assert response.id == "test_id"
    assert response.response_text == "Hello, world!"
    assert response.input_prompt == "Say hello"
    assert response.time_to_first_token == 0.1
    assert response.time_to_last_token == 0.5
    assert response.num_tokens_input == 3
    assert response.num_tokens_output == 2
    assert response.time_per_output_token == 0.2
    assert response.error is None


def test_invocation_response_to_json():
    response = InvocationResponse(
        id="test_id", response_text="Hello, world!", input_prompt="Say hello"
    )
    json_str = response.to_json()
    assert "test_id" in json_str
    assert "Hello, world!" in json_str
    assert "Say hello" in json_str


def test_invocation_response_error_output():
    error_response = InvocationResponse.error_output(
        input_payload={"input": "Test prompt"}, error="Test error"
    )
    assert error_response.response_text is None
    assert error_response.input_payload == {"input": "Test prompt"}
    assert error_response.error == "Test error"
    assert error_response.id is not None


def test_invocation_response_repr_and_str():
    response = InvocationResponse(
        id="test_id", response_text="Hello, world!", input_prompt="Say hello"
    )
    repr_str = repr(response)
    str_str = str(response)
    assert "test_id" in repr_str
    assert "Hello, world!" in repr_str
    assert "Say hello" in str_str
    assert repr_str != str_str  # str should be indented


# Tests for BaseEndpoint


class ConcreteEndpoint(Endpoint[dict]):
    def __init__(self, endpoint_name: str, model_id: str, provider: str):
        super().__init__(
            endpoint_name=endpoint_name, model_id=model_id, provider=provider
        )

    @Endpoint.llmeter_invoke
    def invoke(self, payload: dict) -> dict:
        return payload

    def process_raw_response(
        self, raw_response: dict, start_t: float, response: InvocationResponse
    ) -> None:
        response.id = "test_id"
        response.response_text = f"Invoked with payload: {raw_response}"
        response.input_prompt = raw_response.get("prompt", "")

    @classmethod
    def create_payload(cls, prompt: str):
        return {"prompt": prompt}


llmeter.endpoints.ConcreteEndpoint = ConcreteEndpoint  # type: ignore


@pytest.fixture
def concrete_endpoint():
    return ConcreteEndpoint("test_endpoint", "test_model", "test_provider")


def test_base_endpoint_initialization(concrete_endpoint):
    assert concrete_endpoint.endpoint_name == "test_endpoint"
    assert concrete_endpoint.model_id == "test_model"


def test_base_endpoint_invoke(concrete_endpoint):
    payload = {"prompt": "Hello"}
    response = concrete_endpoint.invoke(payload)
    assert isinstance(response, InvocationResponse)
    assert response.id == "test_id"
    assert response.response_text == "Invoked with payload: {'prompt': 'Hello'}"
    assert response.input_prompt == "Hello"


def test_invoke_sets_request_time(concrete_endpoint):
    """The invoke wrapper should always set request_time on the response."""
    from datetime import datetime, timezone

    before = datetime.now(timezone.utc)
    response = concrete_endpoint.invoke({"prompt": "Hello"})
    after = datetime.now(timezone.utc)

    assert response.request_time is not None
    assert before <= response.request_time <= after


def test_invoke_error_sets_request_time(concrete_endpoint):
    """request_time should be set even when invoke raises an exception."""
    from datetime import datetime, timezone
    from unittest.mock import patch

    # Make parse_response raise to trigger the error path
    with patch.object(
        type(concrete_endpoint),
        "process_raw_response",
        side_effect=RuntimeError("boom"),
    ):
        before = datetime.now(timezone.utc)
        response = concrete_endpoint.invoke({"prompt": "Hello"})
        after = datetime.now(timezone.utc)

    assert response.error is not None
    assert response.request_time is not None
    assert before <= response.request_time <= after


def test_base_endpoint_create_payload():
    payload = ConcreteEndpoint.create_payload("Test prompt")
    assert payload == {"prompt": "Test prompt"}


def test_endpoint_abstract_methods():
    with pytest.raises(TypeError):
        Endpoint("test", "test", "test")  # type: ignore


def test_endpoint_to_dict(concrete_endpoint):
    endpoint_dict = concrete_endpoint.to_dict()
    assert endpoint_dict == {
        "endpoint_name": "test_endpoint",
        "model_id": "test_model",
        "provider": "test_provider",
        "endpoint_type": "ConcreteEndpoint",
    }


def test_endpoint_save_and_load(concrete_endpoint, tmp_path):
    save_path = tmp_path / "test_endpoint.json"
    concrete_endpoint.save(save_path)

    loaded_endpoint = ConcreteEndpoint.load_from_file(save_path)
    assert loaded_endpoint.endpoint_name == concrete_endpoint.endpoint_name
    assert loaded_endpoint.model_id == concrete_endpoint.model_id
    assert loaded_endpoint.provider == concrete_endpoint.provider


def test_endpoint_load_from_dict():
    config = {
        "endpoint_name": "test_endpoint",
        "model_id": "test_model",
        "provider": "test_provider",
        "endpoint_type": "ConcreteEndpoint",
    }
    loaded_endpoint = ConcreteEndpoint.load(config)
    assert loaded_endpoint.endpoint_name == "test_endpoint"
    assert loaded_endpoint.model_id == "test_model"
    assert loaded_endpoint.provider == "test_provider"


def test_invocation_response_annotations_roundtrip():
    """The `annotations` dict round-trips through to_json/from_json."""
    resp = InvocationResponse(
        response_text="hi", annotations={"cost_total": 0.01, "label": "x"}
    )
    restored = InvocationResponse.from_json(resp.to_json())
    assert restored.annotations == {"cost_total": 0.01, "label": "x"}


def test_invocation_response_annotations_default_is_empty_dict():
    """annotations defaults to an independent empty dict per instance."""
    a = InvocationResponse(response_text="a")
    b = InvocationResponse(response_text="b")
    assert a.annotations == {}
    a.annotations["x"] = 1
    assert b.annotations == {}  # no shared mutable default


def test_from_json_collects_unknown_fields_into_annotations():
    """Unknown top-level keys (e.g. legacy callback-written fields) are captured into annotations."""
    raw = json.dumps(
        {
            "response_text": "hi",
            "id": "r1",
            "num_tokens_input": 10,
            "cost_total": 0.02,  # legacy callback-written field
            "cost_InputTokens": 0.01,
            "end_time": "2024-01-01T00:00:00Z",  # since-removed field
        }
    )
    resp = InvocationResponse.from_json(raw)

    # Known fields are parsed normally
    assert resp.response_text == "hi"
    assert resp.id == "r1"
    assert resp.num_tokens_input == 10
    # Unknown fields are preserved (not dropped, not raising) under annotations
    assert resp.annotations == {
        "cost_total": 0.02,
        "cost_InputTokens": 0.01,
        "end_time": "2024-01-01T00:00:00Z",
    }


def test_from_json_merges_unknown_fields_with_existing_annotations():
    """Explicit annotations and stray unknown keys both survive load."""
    raw = json.dumps({"response_text": "hi", "annotations": {"kept": 1}, "stray": 2})
    resp = InvocationResponse.from_json(raw)
    assert resp.annotations == {"kept": 1, "stray": 2}


def test_invocation_response_to_dict():
    response = InvocationResponse(
        id="test_id",
        response_text="Hello, world!",
        input_prompt="Say hello",
        time_to_first_token=0.1,
        time_to_last_token=0.5,
        num_tokens_input=3,
        num_tokens_output=2,
        time_per_output_token=0.2,
    )
    response_dict = response.to_dict()
    assert response_dict["id"] == "test_id"
    assert response_dict["response_text"] == "Hello, world!"
    assert response_dict["input_prompt"] == "Say hello"
    assert response_dict["time_to_first_token"] == 0.1
    assert response_dict["time_to_last_token"] == 0.5
    assert response_dict["num_tokens_input"] == 3
    assert response_dict["num_tokens_output"] == 2
    assert response_dict["time_per_output_token"] == 0.2


def test_endpoint_subclasshook():
    class ValidEndpoint(Endpoint):
        def invoke(self, payload):
            pass

        @staticmethod
        def create_payload():
            pass

    class InvalidEndpoint:
        pass

    assert issubclass(ValidEndpoint, Endpoint)
    assert not issubclass(InvalidEndpoint, Endpoint)


def test_endpoint_load_from_file_error(tmp_path):
    invalid_path = tmp_path / "nonexistent_file.json"
    with pytest.raises(FileNotFoundError):
        ConcreteEndpoint.load_from_file(invalid_path)


def test_endpoint_load_error():
    invalid_config = {
        "endpoint_name": "test_endpoint",
        "model_id": "test_model",
        # Missing "provider" key
    }
    with pytest.raises(KeyError):
        ConcreteEndpoint.load(invalid_config)


# ---------------------------------------------------------------------------
# Tests: base-module reasoning helpers
# ---------------------------------------------------------------------------


class TestDefaultReasoningVisibilityInference:
    """The model-ID heuristic in `llmeter.endpoints.base`, independent of any endpoint.

    Lives here rather than with a provider's tests because it is provider-agnostic: it handles
    Bedrock-style `.` namespacing *and* LiteLLM-style `/` prefixes, and is used by the Bedrock,
    OpenAI-compatible and LiteLLM endpoints alike.
    """

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("anthropic.claude-opus-4-7", "summary"),
            ("us.anthropic.claude-sonnet-4-6", "summary"),
            ("global.anthropic.claude-opus-4-7", "summary"),
            ("openai.gpt-oss-120b-1:0", "verbatim"),
            ("qwen.qwen3-32b-v1:0", "verbatim"),
            ("deepseek.r1-v1:0", "verbatim"),
            # Substring matches must not count -- only whole dot-separated segments
            ("acme.anthropic-lookalike-v1", "verbatim"),
            # LiteLLM-style provider prefixes use "/" rather than "."
            ("bedrock/anthropic.claude-opus-4-6", "summary"),
            ("anthropic/claude-sonnet-4-6", "summary"),
            ("openai/gpt-4", "verbatim"),
        ],
    )
    def test_inference_from_model_id(self, model_id, expected):
        assert infer_reasoning_visibility_from_model_id(model_id) == expected


class TestBackfillReasoningTypeFromTokenCounts:
    """Inferring that reasoning happened from token accounting alone.

    Guards the case where a provider bills for reasoning tokens without streaming anything that
    identifies them (OpenAI Chat Completions, OpenAI Responses without a requested summary, or any
    gateway that strips reasoning fields). Left unresolved, `reasoning_type=None` would claim the
    model did not reason, and TPOT would divide a post-reasoning window by a reasoning-inclusive
    token count.
    """

    @pytest.mark.parametrize("reasoning_tokens", [1, 42, 100_000])
    def test_positive_count_resolves_to_unknown(self, reasoning_tokens):
        response = InvocationResponse(
            response_text=None, num_tokens_output_reasoning=reasoning_tokens
        )

        backfill_reasoning_type_from_token_counts(response)

        assert response.reasoning_type == "unknown"

    @pytest.mark.parametrize(
        "reasoning_tokens,why",
        [
            (None, "provider reported no breakdown at all"),
            (0, "provider confirmed the model did not reason"),
        ],
    )
    def test_no_evidence_leaves_reasoning_type_unset(self, reasoning_tokens, why):
        response = InvocationResponse(
            response_text=None, num_tokens_output_reasoning=reasoning_tokens
        )

        backfill_reasoning_type_from_token_counts(response)

        assert response.reasoning_type is None, why

    @pytest.mark.parametrize(
        "established", ["verbatim", "summary", "redacted", "unknown"]
    )
    def test_never_overrides_a_value_from_the_response_content(self, established):
        """Observation beats inference: `"redacted"` in particular must not be downgraded."""
        response = InvocationResponse(
            response_text=None,
            reasoning_type=established,
            num_tokens_output_reasoning=99,
        )

        backfill_reasoning_type_from_token_counts(response)

        assert response.reasoning_type == established

    @pytest.mark.parametrize(
        "reasoning_tokens,why",
        [
            (True, "bool is an int subclass but is not a token count"),
            ("7", "a string count is not trustworthy"),
            (7.5, "a float count is not trustworthy"),
            (MagicMock(), "a placeholder/mock attribute must not look like a count"),
        ],
    )
    def test_non_integer_counts_are_ignored(self, reasoning_tokens, why):
        response = InvocationResponse(
            response_text=None, num_tokens_output_reasoning=reasoning_tokens
        )

        backfill_reasoning_type_from_token_counts(response)

        assert response.reasoning_type is None, why


class TestSilentReasoningTypeByConnector:
    """Which label each connector uses for reasoning it can see was billed but not disclosed.

    `"redacted"` and `"unknown"` route TPOT identically, so this is purely about how confidently the
    situation can be described - which differs by how completely the connector can parse its API.
    Pinned here, in one place, because the rationale is a cross-connector judgement rather than a
    property of any single endpoint.
    """

    def test_default_is_the_cautious_label(self):
        """Anything that has not opted in must not claim withholding it cannot demonstrate."""
        assert Endpoint.silent_reasoning_type == "unknown"

    @pytest.mark.parametrize(
        "module,class_names,expected,why",
        [
            (
                "llmeter.endpoints.openai_response",
                ["OpenAIResponseEndpoint", "OpenAIResponseStreamEndpoint"],
                "redacted",
                "the Responses schema states disclosure structurally, so silence is withholding",
            ),
            (
                "llmeter.endpoints.openai",
                ["OpenAICompletionEndpoint", "OpenAICompletionStreamEndpoint"],
                "unknown",
                "Chat Completions has no reasoning field, so an unrecognized vendor extension is "
                "indistinguishable from withholding",
            ),
            (
                "llmeter.endpoints.anthropic_messages",
                ["AnthropicMessages", "AnthropicMessagesStream"],
                "unknown",
                "every documented thinking mode leaves a trace, so reaching the fallback means "
                "something undocumented happened and is worth surfacing",
            ),
            (
                "llmeter.endpoints.litellm",
                ["LiteLLM", "LiteLLMStreaming"],
                "unknown",
                "LiteLLM may not normalize a given provider's reasoning fields",
            ),
            (
                "llmeter.endpoints.bedrock",
                ["BedrockConverse", "BedrockConverseStream"],
                "unknown",
                "Converse reports no reasoning-token breakdown, so this is unreachable anyway",
            ),
        ],
    )
    def test_per_connector_label(self, module, class_names, expected, why):
        mod = pytest.importorskip(module)
        for name in class_names:
            assert getattr(mod, name).silent_reasoning_type == expected, (
                f"{name}: {why}"
            )

    def test_not_serialized_with_the_endpoint(self):
        """A class-level constant describing the API, not user configuration."""
        endpoint = ConcreteEndpoint("test_endpoint", "test_model", "test_provider")

        assert "silent_reasoning_type" not in endpoint.to_dict()


class TestLlmeterInvokeAppliesReasoningBackfill:
    """The backfill is wired into `llmeter_invoke`, so custom endpoints get it for free."""

    class _HiddenReasoningEndpoint(Endpoint[dict]):
        """An endpoint whose provider bills reasoning tokens but streams no reasoning content."""

        def __init__(
            self, reasoning_tokens: int | None, reasoning_type=None, silent=None
        ):
            super().__init__(
                endpoint_name="hidden", model_id="test-model", provider="test"
            )
            self._reasoning_tokens = reasoning_tokens
            self._declared_type = reasoning_type
            if silent is not None:
                self.silent_reasoning_type = silent

        @Endpoint.llmeter_invoke
        def invoke(self, payload: dict) -> dict:
            return payload

        def process_raw_response(self, raw_response, start_t, response) -> None:
            response.response_text = "Answer"
            response.num_tokens_output = 20
            response.num_tokens_output_reasoning = self._reasoning_tokens
            response.reasoning_type = self._declared_type

    def test_backfilled_when_endpoint_leaves_it_unset(self):
        endpoint = self._HiddenReasoningEndpoint(reasoning_tokens=12)

        response = endpoint.invoke({"prompt": "hi"})

        assert response.reasoning_type == "unknown"

    def test_not_backfilled_without_reasoning_tokens(self):
        endpoint = self._HiddenReasoningEndpoint(reasoning_tokens=None)

        response = endpoint.invoke({"prompt": "hi"})

        assert response.reasoning_type is None

    def test_endpoint_classification_is_preserved(self):
        endpoint = self._HiddenReasoningEndpoint(
            reasoning_tokens=12, reasoning_type="summary"
        )

        response = endpoint.invoke({"prompt": "hi"})

        assert response.reasoning_type == "summary"

    def test_uses_the_endpoints_declared_label(self):
        """`llmeter_invoke` must pass `silent_reasoning_type` through, not hard-code a default."""
        endpoint = self._HiddenReasoningEndpoint(reasoning_tokens=12, silent="redacted")

        response = endpoint.invoke({"prompt": "hi"})

        assert response.reasoning_type == "redacted"
