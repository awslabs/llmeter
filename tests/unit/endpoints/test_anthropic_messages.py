# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Python Built-Ins:
from pathlib import Path
import json
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

# External Dependencies:
import pytest

# Local Dependencies:
from llmeter.endpoints.anthropic_messages import (
    _ANTHROPIC_CLIENTS,
    AnthropicMessages,
    AnthropicMessagesEndpoint,
    AnthropicMessagesStream,
)
from llmeter.endpoints.anthropic_messages import _extract_thinking_tokens
from llmeter.endpoints.base import InvocationResponse

_PATCH_CLIENTS = "llmeter.endpoints.anthropic_messages._ANTHROPIC_CLIENTS"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_message(
    msg_id="msg_test123",
    text="Hello, world!",
    input_tokens=10,
    output_tokens=5,
    cache_read_input_tokens=None,
):
    """Build a mock non-streaming Message response."""
    text_block = Mock()
    text_block.type = "text"
    text_block.text = text

    usage = Mock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens

    message = Mock()
    message.id = msg_id
    message.content = [text_block]
    message.usage = usage
    return message


def _make_stream_events(
    msg_id="msg_stream123",
    text_chunks=None,
    input_tokens=10,
    output_tokens=5,
    cache_read_input_tokens=None,
):
    """Build a list of mock SSE streaming events."""
    if text_chunks is None:
        text_chunks = ["Hello", ", ", "world!"]

    events = []

    # message_start
    msg_start_usage = Mock()
    msg_start_usage.input_tokens = input_tokens
    msg_start_usage.cache_read_input_tokens = cache_read_input_tokens

    msg_start_message = Mock()
    msg_start_message.id = msg_id
    msg_start_message.usage = msg_start_usage

    msg_start = Mock()
    msg_start.type = "message_start"
    msg_start.message = msg_start_message
    events.append(msg_start)

    # content_block_start
    block_start = Mock()
    block_start.type = "content_block_start"
    events.append(block_start)

    # content_block_delta events (text)
    for chunk in text_chunks:
        delta = Mock()
        delta.type = "text_delta"
        delta.text = chunk

        event = Mock()
        event.type = "content_block_delta"
        event.delta = delta
        events.append(event)

    # content_block_stop
    block_stop = Mock()
    block_stop.type = "content_block_stop"
    events.append(block_stop)

    # message_delta with usage
    msg_delta_usage = Mock()
    msg_delta_usage.output_tokens = output_tokens

    msg_delta = Mock()
    msg_delta.type = "message_delta"
    msg_delta.usage = msg_delta_usage
    events.append(msg_delta)

    # message_stop
    msg_stop = Mock()
    msg_stop.type = "message_stop"
    events.append(msg_stop)

    return events


def _make_draft_response() -> InvocationResponse:
    """Create a draft InvocationResponse like llmeter_invoke does."""
    return InvocationResponse(response_text=None)


def _message_start_event(msg_id="msg_1", input_tokens=5, cache_read_input_tokens=None):
    """Build a single mock `message_start` event."""
    event = Mock()
    event.type = "message_start"
    event.message = Mock()
    event.message.id = msg_id
    event.message.usage = Mock()
    event.message.usage.input_tokens = input_tokens
    event.message.usage.cache_read_input_tokens = cache_read_input_tokens
    return event


def _delta_event(delta_type: str, **delta_attrs):
    """Build a single mock `content_block_delta` event of the given delta type.

    Example: ``_delta_event("text_delta", text="Hi")``
    """
    event = Mock()
    event.type = "content_block_delta"
    event.delta = Mock()
    event.delta.type = delta_type
    for name, value in delta_attrs.items():
        setattr(event.delta, name, value)
    return event


@pytest.fixture()
def mock_client():
    """Replace every provider in _ANTHROPIC_CLIENTS with a single Mock class."""
    cls = Mock()
    with patch.dict(_ANTHROPIC_CLIENTS, {k: cls for k in _ANTHROPIC_CLIENTS}):
        yield cls


# ---------------------------------------------------------------------------
# Tests: client construction
# ---------------------------------------------------------------------------


class TestBuildClient:
    def test_anthropic_provider(self, mock_client):
        AnthropicMessages(model_id="test-model", provider="anthropic", api_key="k")
        mock_client.assert_called_once_with(api_key="k")

    def test_bedrock_mantle_provider(self, mock_client):
        AnthropicMessages(
            model_id="test-model", provider="bedrock-mantle", aws_region="us-east-1"
        )
        mock_client.assert_called_once_with(aws_region="us-east-1")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            AnthropicMessages(model_id="test-model", provider="unknown")


# ---------------------------------------------------------------------------
# Tests: create_payload
# ---------------------------------------------------------------------------


class TestCreatePayload:
    def test_basic_payload(self):
        payload = AnthropicMessagesEndpoint.create_payload("Hello!")
        assert payload == {
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 256,
        }

    def test_custom_max_tokens(self):
        payload = AnthropicMessagesEndpoint.create_payload("Hi", max_tokens=1024)
        assert payload["max_tokens"] == 1024

    def test_extra_kwargs(self):
        payload = AnthropicMessagesEndpoint.create_payload(
            "Hi", system="Be helpful", temperature=0.7
        )
        assert payload["system"] == "Be helpful"
        assert payload["temperature"] == 0.7

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="positive integer"):
            AnthropicMessagesEndpoint.create_payload("Hi", max_tokens=-1)

    def test_zero_max_tokens(self):
        with pytest.raises(ValueError, match="positive integer"):
            AnthropicMessagesEndpoint.create_payload("Hi", max_tokens=0)

    def test_non_string_message(self):
        with pytest.raises(TypeError, match="must be a str"):
            AnthropicMessagesEndpoint.create_payload(123)

    def test_thinking_adaptive(self):
        payload = AnthropicMessagesEndpoint.create_payload(
            "Think hard", max_tokens=16000, thinking={"type": "adaptive"}
        )
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["max_tokens"] == 16000

    def test_thinking_enabled_with_budget(self):
        payload = AnthropicMessagesEndpoint.create_payload(
            "Prove it",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
        )
        assert payload["thinking"]["type"] == "enabled"
        assert payload["thinking"]["budget_tokens"] == 10000

    def test_thinking_disabled(self):
        payload = AnthropicMessagesEndpoint.create_payload(
            "Hello", thinking={"type": "disabled"}
        )
        assert payload["thinking"] == {"type": "disabled"}

    def test_thinking_none_omitted(self):
        payload = AnthropicMessagesEndpoint.create_payload("Hello")
        assert "thinking" not in payload


# ---------------------------------------------------------------------------
# Tests: _parse_payload
# ---------------------------------------------------------------------------


class TestParsePayload:
    def test_parse_string_content(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        assert endpoint._parse_payload(payload) == "Hello"

    def test_parse_block_content(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ]
        }
        assert endpoint._parse_payload(payload) == "Hello\nWorld"

    def test_parse_empty_messages(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        assert endpoint._parse_payload({"messages": []}) == ""

    def test_parse_no_messages_key(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        assert endpoint._parse_payload({}) == ""

    def test_parse_mixed_content_types(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this:"},
                        {"type": "image", "source": {"type": "base64", "data": "..."}},
                    ],
                }
            ]
        }
        assert endpoint._parse_payload(payload) == "Describe this:"

    def test_parse_multi_turn(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        assert endpoint._parse_payload(payload) == "Hello\nHi there!\nHow are you?"


# ---------------------------------------------------------------------------
# Tests: AnthropicMessages (non-streaming)
# ---------------------------------------------------------------------------


class TestAnthropicMessages:
    def test_process_raw_response(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        mock_response = _make_mock_message()
        response = _make_draft_response()

        endpoint.process_raw_response(mock_response, time.perf_counter(), response)

        assert response.id == "msg_test123"
        assert response.response_text == "Hello, world!"
        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 5
        assert response.time_to_last_token is not None
        # Non-streaming: neither first-token metric is measurable
        assert response.time_to_first_token is None
        assert response.time_to_first_content_token is None

    def test_process_raw_response_with_cache(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        mock_response = _make_mock_message(cache_read_input_tokens=3)
        response = _make_draft_response()

        endpoint.process_raw_response(mock_response, time.perf_counter(), response)

        assert response.num_tokens_input_cached == 3

    def test_process_raw_response_no_usage(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        mock_response = _make_mock_message()
        mock_response.usage = None
        response = _make_draft_response()

        endpoint.process_raw_response(mock_response, time.perf_counter(), response)

        assert response.num_tokens_input is None
        assert response.num_tokens_output is None
        assert response.num_tokens_input_cached is None

    def test_process_raw_response_multiple_text_blocks(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")

        block1 = Mock()
        block1.type = "text"
        block1.text = "Part 1. "

        block2 = Mock()
        block2.type = "text"
        block2.text = "Part 2."

        mock_response = _make_mock_message()
        mock_response.content = [block1, block2]
        response = _make_draft_response()

        endpoint.process_raw_response(mock_response, time.perf_counter(), response)
        assert response.response_text == "Part 1. Part 2."

    def test_invoke_success(self, mock_client):
        mock_client.return_value.messages.create.return_value = _make_mock_message()

        endpoint = AnthropicMessages(model_id="test-model")
        result = endpoint.invoke(
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}
        )

        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hello, world!"
        assert result.error is None

    def test_invoke_api_error(self, mock_client):
        mock_client.return_value.messages.create.side_effect = Exception("API error")

        endpoint = AnthropicMessages(model_id="test-model")
        result = endpoint.invoke(
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}
        )

        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "API error" in result.error

    def test_prepare_payload_sets_model(self, mock_client):
        endpoint = AnthropicMessages(model_id="claude-opus-4-7")
        payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 256}

        prepared = endpoint.prepare_payload(payload)

        assert prepared["model"] == "claude-opus-4-7"

    def test_prepare_payload_merges_kwargs(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")
        payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 256}

        prepared = endpoint.prepare_payload(payload, temperature=0.5)

        assert prepared["temperature"] == 0.5
        assert prepared["model"] == "test-model"


# ---------------------------------------------------------------------------
# Tests: AnthropicMessagesStream (streaming)
# ---------------------------------------------------------------------------


class TestAnthropicMessagesStream:
    def test_process_raw_response_basic(self, mock_client):
        endpoint = AnthropicMessagesStream(model_id="test-model")
        events = _make_stream_events()
        response = _make_draft_response()

        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.id == "msg_stream123"
        assert response.response_text == "Hello, world!"
        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 5
        assert response.time_to_first_token is not None
        assert response.time_to_last_token is not None
        assert response.time_to_first_token <= response.time_to_last_token

    def test_process_raw_response_with_cache(self, mock_client):
        endpoint = AnthropicMessagesStream(model_id="test-model")
        events = _make_stream_events(cache_read_input_tokens=7)
        response = _make_draft_response()

        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.num_tokens_input_cached == 7

    def test_process_raw_response_empty_stream(self, mock_client):
        endpoint = AnthropicMessagesStream(model_id="test-model")
        response = _make_draft_response()

        endpoint.process_raw_response(iter([]), time.perf_counter(), response)

        assert response.response_text is None
        assert response.id is None

    def test_process_raw_response_no_text_deltas(self, mock_client):
        """Stream with message_start and message_delta but no text content."""
        endpoint = AnthropicMessagesStream(model_id="test-model")
        events = _make_stream_events(text_chunks=[])
        response = _make_draft_response()

        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.response_text is None
        assert response.time_to_first_token is None
        assert response.time_to_first_content_token is None

    def test_process_raw_response_timing(self, mock_client):
        """Verify TTFT is captured on the first text delta."""
        endpoint = AnthropicMessagesStream(model_id="test-model")
        events = _make_stream_events(text_chunks=["First", " Second"])
        response = _make_draft_response()

        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        assert response.time_to_first_token is not None
        assert response.time_to_last_token is not None
        assert response.time_to_first_token > 0
        assert response.time_to_last_token >= response.time_to_first_token

    def test_prepare_payload_sets_stream(self, mock_client):
        endpoint = AnthropicMessagesStream(model_id="test-model")
        payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 256}

        prepared = endpoint.prepare_payload(payload)

        assert prepared["stream"] is True
        assert prepared["model"] == "test-model"

    def test_invoke_success(self, mock_client):
        mock_client.return_value.messages.create.return_value = iter(
            _make_stream_events(text_chunks=["Hi!"])
        )

        endpoint = AnthropicMessagesStream(model_id="test-model")
        result = endpoint.invoke(
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}
        )

        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hi!"
        assert result.error is None

    def test_invoke_api_error(self, mock_client):
        mock_client.return_value.messages.create.side_effect = Exception("Stream error")

        endpoint = AnthropicMessagesStream(model_id="test-model")
        result = endpoint.invoke(
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}
        )

        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "Stream error" in result.error

    def test_process_raw_response_thinking_deltas_excluded_from_text(self, mock_client):
        """Thinking deltas don't contribute to response_text (default ttft_visible_tokens_only=True)."""
        endpoint = AnthropicMessagesStream(model_id="test-model")

        events = []

        # message_start
        msg_start = Mock()
        msg_start.type = "message_start"
        msg_start.message = Mock()
        msg_start.message.id = "msg_1"
        msg_start.message.usage = Mock()
        msg_start.message.usage.input_tokens = 5
        msg_start.message.usage.cache_read_input_tokens = None
        events.append(msg_start)

        # A thinking delta (should not affect response_text or TTFT)
        thinking_delta = Mock()
        thinking_delta.type = "content_block_delta"
        thinking_delta.delta = Mock()
        thinking_delta.delta.type = "thinking_delta"
        thinking_delta.delta.thinking = "Let me think..."
        events.append(thinking_delta)

        # A text delta
        text_delta = Mock()
        text_delta.type = "content_block_delta"
        text_delta.delta = Mock()
        text_delta.delta.type = "text_delta"
        text_delta.delta.text = "Answer"
        events.append(text_delta)

        # message_delta
        msg_delta = Mock()
        msg_delta.type = "message_delta"
        msg_delta.usage = Mock()
        msg_delta.usage.output_tokens = 3
        events.append(msg_delta)

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.response_text == "Answer"
        assert response.num_tokens_output == 3

    def test_ttft_includes_thinking_delta(self, mock_client):
        """A thinking_delta is a real output token, so it sets TTFT but not the content TTFT."""
        endpoint = AnthropicMessagesStream(model_id="test-model")

        events = [
            _message_start_event(),
            _delta_event("thinking_delta", thinking="Reasoning..."),
            _delta_event("text_delta", text="Result"),
        ]

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            # message_start, thinking_delta, text_delta
            clock.side_effect = [100.1, 100.4, 100.9]
            endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.4)
        assert response.time_to_first_content_token == pytest.approx(0.9)
        assert response.response_text == "Result"

    def test_ttft_equals_content_ttft_without_thinking(self, mock_client):
        """With no thinking, both first-token metrics describe the same token."""
        endpoint = AnthropicMessagesStream(model_id="test-model")

        events = [
            _message_start_event(),
            _delta_event("text_delta", text="Result"),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token == response.time_to_first_content_token

    def test_thinking_deltas_do_not_set_content_ttft(self, mock_client):
        """A thinking-only response has no visible content token to time."""
        endpoint = AnthropicMessagesStream(model_id="test-model")

        events = [
            _message_start_event(),
            _delta_event("thinking_delta", thinking="Reasoning..."),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_content_token is None
        assert response.response_text is None


# ---------------------------------------------------------------------------
# Tests: Endpoint initialization
# ---------------------------------------------------------------------------


class TestEndpointInit:
    def test_default_provider(self, mock_client):
        endpoint = AnthropicMessages(model_id="claude-opus-4-7")
        assert endpoint.provider == "anthropic"
        assert endpoint.model_id == "claude-opus-4-7"
        assert endpoint.endpoint_name == "anthropic-messages"

    def test_bedrock_mantle_provider(self, mock_client):
        endpoint = AnthropicMessages(
            model_id="anthropic.claude-opus-4-7",
            provider="bedrock-mantle",
            aws_region="us-east-1",
        )
        assert endpoint.provider == "bedrock-mantle"
        mock_client.assert_called_once_with(aws_region="us-east-1")

    def test_custom_endpoint_name(self, mock_client):
        endpoint = AnthropicMessages(
            model_id="test-model", endpoint_name="my-custom-endpoint"
        )
        assert endpoint.endpoint_name == "my-custom-endpoint"


# ---------------------------------------------------------------------------
# Tests: Serialization
# ---------------------------------------------------------------------------


class TestAnthropicSerialization:
    """Test that client configuration round-trips through serialization."""

    def test_serializes_basic_fields(self):
        """Test serialized state includes provider and model."""
        from llmeter.serialization import dump_object

        endpoint = AnthropicMessages(model_id="claude-opus-4-7", api_key="test")
        state = dump_object(endpoint)["__llmeter_state__"]
        assert state["model_id"] == "claude-opus-4-7"
        assert state["provider"] == "anthropic"
        assert state["endpoint_name"] == "anthropic-messages"
        assert state["max_retries"] == 2
        assert "base_url" in state

    def test_secrets_excluded(self):
        """Test that secrets (keys, tokens) are not persisted."""
        from llmeter.serialization import dump_object

        endpoint = AnthropicMessages(model_id="claude-opus-4-7", api_key="sk-secret")
        state = dump_object(endpoint)["__llmeter_state__"]
        for key in state:
            assert "key" not in key, f"Secret-like key '{key}' found in state"
            assert "token" not in key, f"Secret-like key '{key}' found in state"
        assert "sk-secret" not in str(state.values())

    def test_bedrock_mantle_serializes_region(self):
        """Test Bedrock Mantle provider captures aws_region."""
        from llmeter.serialization import dump_object

        endpoint = AnthropicMessages(
            model_id="anthropic.claude-opus-4-7",
            provider="bedrock-mantle",
            aws_region="us-west-2",
        )
        state = dump_object(endpoint)["__llmeter_state__"]
        assert state["aws_region"] == "us-west-2"
        assert state["provider"] == "bedrock-mantle"
        assert "aws_secret_key" not in state
        assert "aws_access_key" not in state
        assert "aws_session_token" not in state

    def test_custom_headers_serialized(self):
        """Test that user-supplied headers round-trip but SDK headers don't."""
        from llmeter.serialization import dump_object

        endpoint = AnthropicMessages(
            model_id="claude-opus-4-7",
            api_key="test",
            default_headers={"X-Custom": "val"},
        )
        state = dump_object(endpoint)["__llmeter_state__"]
        assert state["default_headers"] == {"X-Custom": "val"}

    def test_no_headers_when_none_custom(self):
        """Test that default_headers is absent when no custom headers set."""
        from llmeter.serialization import dump_object

        endpoint = AnthropicMessages(model_id="claude-opus-4-7", api_key="test")
        state = dump_object(endpoint)["__llmeter_state__"]
        assert "default_headers" not in state

    def test_timeout_serialized_as_dict(self):
        """Test httpx.Timeout is serialized as a dict."""
        import httpx

        from llmeter.serialization import dump_object

        endpoint = AnthropicMessages(
            model_id="claude-opus-4-7",
            api_key="test",
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        state = dump_object(endpoint)["__llmeter_state__"]
        assert state["timeout"] == {
            "connect": 5.0,
            "read": 30.0,
            "write": 30.0,
            "pool": 30.0,
        }

    def test_round_trip_direct_provider(self):
        """Test save/load round-trip for direct Anthropic provider."""
        from llmeter.endpoints.base import Endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            original = AnthropicMessages(
                model_id="claude-opus-4-7",
                api_key="test",
                max_retries=5,
                base_url="https://custom.anthropic.com",
                default_headers={"X-Foo": "bar"},
            )
            path = Path(tmpdir) / "endpoint.json"
            original.save_to_file(path)

            loaded = Endpoint.load_from_file(path)

            assert isinstance(loaded, AnthropicMessages)
            assert loaded.model_id == "claude-opus-4-7"
            assert loaded.provider == "anthropic"
            assert loaded._client.max_retries == 5
            assert "custom.anthropic.com" in str(loaded._client.base_url)
            assert loaded._client._custom_headers == {"X-Foo": "bar"}

    def test_round_trip_bedrock_mantle(self):
        """Test save/load round-trip for Bedrock Mantle provider."""
        from llmeter.endpoints.base import Endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            original = AnthropicMessages(
                model_id="anthropic.claude-opus-4-7",
                provider="bedrock-mantle",
                aws_region="eu-west-1",
            )
            path = Path(tmpdir) / "endpoint.json"
            original.save_to_file(path)

            loaded = Endpoint.load_from_file(path)

            assert isinstance(loaded, AnthropicMessages)
            assert loaded.provider == "bedrock-mantle"
            assert loaded._client.aws_region == "eu-west-1"

    def test_round_trip_stream_endpoint(self):
        """Test save/load round-trip for streaming endpoint."""
        from llmeter.endpoints.base import Endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            original = AnthropicMessagesStream(
                model_id="claude-opus-4-7",
                api_key="test",
                max_retries=4,
            )
            path = Path(tmpdir) / "endpoint.json"
            original.save_to_file(path)

            loaded = Endpoint.load_from_file(path)

            assert isinstance(loaded, AnthropicMessagesStream)
            assert loaded.model_id == "claude-opus-4-7"
            assert loaded._client.max_retries == 4

    def test_saved_config_omits_deprecated_ttft_flag(self):
        """Newly-saved configs should not carry the retired flag forward."""
        with tempfile.TemporaryDirectory() as tmpdir:
            endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7", api_key="k")
            path = Path(tmpdir) / "endpoint.json"
            endpoint.save_to_file(path)

            state = json.loads(path.read_text())["__llmeter_state__"]

        assert state.get("ttft_visible_tokens_only") is None

    def test_legacy_config_with_ttft_flag_still_loads(self):
        """A config saved by an older LLMeter version must still load, with a warning."""
        from llmeter.endpoints.base import Endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "endpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "__llmeter_class__": "llmeter.endpoints.anthropic_messages.AnthropicMessagesStream",
                        "__llmeter_state__": {
                            "model_id": "claude-opus-4-7",
                            "endpoint_name": "anthropic-messages",
                            "provider": "anthropic",
                            "api_key": "test",
                            "ttft_visible_tokens_only": True,
                        },
                    }
                )
            )

            with pytest.warns(DeprecationWarning, match="ttft_visible_tokens_only"):
                loaded = Endpoint.load_from_file(path)

        assert isinstance(loaded, AnthropicMessagesStream)
        assert loaded.model_id == "claude-opus-4-7"

    def test_round_trip_with_granular_timeout(self):
        """Test httpx.Timeout round-trips correctly through dict serialization."""
        import httpx

        from llmeter.endpoints.base import Endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            original = AnthropicMessages(
                model_id="claude-opus-4-7",
                api_key="test",
                timeout=httpx.Timeout(10.0, connect=2.0),
            )
            path = Path(tmpdir) / "endpoint.json"
            original.save_to_file(path)

            loaded = Endpoint.load_from_file(path)

            assert loaded._client.timeout == httpx.Timeout(10.0, connect=2.0)


# ---------------------------------------------------------------------------
# Tests: thinking-token accounting
# ---------------------------------------------------------------------------


class TestExtractThinkingTokens:
    """`output_tokens_details` can arrive as a modelled object or a plain dict."""

    def test_modelled_object_shape(self):
        """Newer anthropic SDKs model the field, so it has attribute access."""
        usage = SimpleNamespace(
            output_tokens=100,
            output_tokens_details=SimpleNamespace(thinking_tokens=42),
        )
        assert _extract_thinking_tokens(usage) == 42

    def test_plain_dict_shape(self):
        """Older SDKs don't model it, so pydantic keeps it as a dict in `model_extra`."""
        usage = SimpleNamespace(
            output_tokens=100, output_tokens_details={"thinking_tokens": 42}
        )
        assert _extract_thinking_tokens(usage) == 42

    def test_real_sdk_roundtrip(self):
        """Guard against SDK-version drift in how the extra field is surfaced."""
        from anthropic.types import Usage

        usage = Usage.model_validate(
            {
                "input_tokens": 10,
                "output_tokens": 100,
                "output_tokens_details": {"thinking_tokens": 42},
            }
        )
        assert _extract_thinking_tokens(usage) == 42

    def test_zero_is_preserved(self):
        """0 means 'known to be no thinking', which is different from None."""
        usage = SimpleNamespace(output_tokens_details={"thinking_tokens": 0})
        assert _extract_thinking_tokens(usage) == 0

    def test_absent_details(self):
        assert _extract_thinking_tokens(SimpleNamespace(output_tokens=100)) is None

    def test_null_details(self):
        usage = SimpleNamespace(output_tokens=100, output_tokens_details=None)
        assert _extract_thinking_tokens(usage) is None

    def test_details_without_thinking_tokens(self):
        usage = SimpleNamespace(output_tokens_details={"something_else": 1})
        assert _extract_thinking_tokens(usage) is None

    def test_non_integer_value_rejected(self):
        """A Mock or other placeholder must not be mistaken for a count."""
        usage = SimpleNamespace(output_tokens_details=Mock())
        assert _extract_thinking_tokens(usage) is None

    def test_bool_rejected(self):
        usage = SimpleNamespace(output_tokens_details={"thinking_tokens": True})
        assert _extract_thinking_tokens(usage) is None


class TestThinkingTokenAccounting:
    def test_non_streaming_reports_thinking_tokens(self, mock_client):
        endpoint = AnthropicMessages(model_id="test-model")

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Answer"
        message = Mock()
        message.id = "msg_1"
        message.content = [text_block]
        message.usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=100,
            cache_read_input_tokens=None,
            output_tokens_details={"thinking_tokens": 60},
        )

        response = _make_draft_response()
        endpoint.process_raw_response(message, time.perf_counter(), response)

        assert response.num_tokens_output == 100
        assert response.num_tokens_output_reasoning == 60

    def test_streaming_reports_thinking_tokens(self, mock_client):
        endpoint = AnthropicMessagesStream(model_id="test-model")

        msg_delta = Mock()
        msg_delta.type = "message_delta"
        msg_delta.usage = SimpleNamespace(
            output_tokens=100, output_tokens_details={"thinking_tokens": 60}
        )

        events = [
            _message_start_event(),
            _delta_event("thinking_delta", thinking="hmm"),
            _delta_event("text_delta", text="Answer"),
            msg_delta,
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.num_tokens_output == 100
        assert response.num_tokens_output_reasoning == 60

    def test_streaming_without_breakdown_leaves_none(self, mock_client):
        """An API response with no `output_tokens_details` must not invent a value."""
        endpoint = AnthropicMessagesStream(model_id="test-model")

        msg_delta = Mock()
        msg_delta.type = "message_delta"
        msg_delta.usage = SimpleNamespace(output_tokens=100)

        events = [
            _message_start_event(),
            _delta_event("text_delta", text="Answer"),
            msg_delta,
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.num_tokens_output == 100
        assert response.num_tokens_output_reasoning is None


# ---------------------------------------------------------------------------
# Tests: omitted-mode thinking, end to end through endpoint + Runner
# ---------------------------------------------------------------------------


class TestRedactedThinkingTpotRecovery:
    """`display: "omitted"` still yields a correct TPOT, via `reasoning_type`.

    Unlike the arithmetic-only cases in
    ``tests/unit/test_runner.py::TestComputeTimePerOutputToken``, this drives the real endpoint
    parser so the whole chain is covered: the stream shape produces
    ``reasoning_type="redacted"``, ``usage.output_tokens_details.thinking_tokens`` populates
    ``num_tokens_output_reasoning``, and the Runner then declines the full-output pairing and uses
    the visible-token one. A regression in any one of the three breaks this.

    Note TTFT *is* populated here (the signature time), so the Runner cannot rely on its absence --
    it has to honour ``reasoning_type``.
    """

    @staticmethod
    def _omitted_mode_events(thinking_tokens: int | None):
        """Build the event sequence a `display: "omitted"` request produces.

        No `thinking_delta` is streamed -- only a `signature_delta`, after thinking completes.
        """
        usage = SimpleNamespace(output_tokens=100)
        if thinking_tokens is not None:
            usage = SimpleNamespace(
                output_tokens=100,
                output_tokens_details={"thinking_tokens": thinking_tokens},
            )
        msg_delta = Mock()
        msg_delta.type = "message_delta"
        msg_delta.usage = usage

        return [
            _message_start_event(),
            _delta_event("signature_delta", signature="EosnCkYICxIMMb3LzNrMu..."),
            _delta_event("text_delta", text="55"),
            _delta_event("text_delta", text="5"),
            msg_delta,
        ]

    @pytest.mark.asyncio
    async def test_tpot_uses_visible_pairing_despite_ttft_being_present(
        self, mock_client
    ):
        from llmeter.runner import _Run

        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
        response = _make_draft_response()

        with patch("time.perf_counter") as clock:
            # message_start, signature_delta, text_delta, text_delta, message_delta
            clock.side_effect = [100.1, 100.5, 102.0, 105.0, 106.0]
            endpoint.process_raw_response(
                iter(self._omitted_mode_events(thinking_tokens=60)), 100.0, response
            )

        # Endpoint: TTFT is the signature time, and the disclosure level is recorded
        assert response.time_to_first_token == pytest.approx(0.5)
        assert response.time_to_first_content_token == pytest.approx(2.0)
        assert response.time_to_last_token == pytest.approx(5.0)
        assert response.reasoning_type == "redacted"
        assert response.num_tokens_output == 100
        assert response.num_tokens_output_reasoning == 60

        await _Run._compute_time_per_output_token(response)

        # Visible pairing: (5.0 - 2.0) / ((100 - 60) - 1)
        assert response.time_per_output_token == pytest.approx(3.0 / 39)
        # NOT the full-output pairing, which TTFT alone would have selected
        assert response.time_per_output_token != pytest.approx((5.0 - 0.5) / (100 - 1))

    @pytest.mark.asyncio
    async def test_tpot_unavailable_without_thinking_token_count(self, mock_client):
        """If a provider or gateway strips the breakdown, TPOT must be None rather than wrong.

        Pins the boundary of the recovery above: it depends entirely on `thinking_tokens` being
        reported, since without it the visible-token count cannot be derived. TTFT is present, so
        this also confirms the Runner does not silently fall back to the full-output pairing.
        """
        from llmeter.runner import _Run

        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
        response = _make_draft_response()

        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.1, 100.5, 102.0, 105.0, 106.0]
            endpoint.process_raw_response(
                iter(self._omitted_mode_events(thinking_tokens=None)), 100.0, response
            )

        assert response.time_to_first_token == pytest.approx(0.5)
        assert response.reasoning_type == "redacted"
        assert response.num_tokens_output_reasoning is None

        await _Run._compute_time_per_output_token(response)
        assert response.time_per_output_token is None


# ---------------------------------------------------------------------------
# Tests: reasoning_type resolution, across both Anthropic endpoints
# ---------------------------------------------------------------------------


def _redacted_block_event(data="EmwKAhgBEgy3va3pzix"):
    """A whole `redacted_thinking` block, which arrives via `content_block_start`.

    The Messages API has no redacted-thinking *delta* type, so a connector inspecting only
    `content_block_delta` cannot see these at all.
    """
    event = Mock()
    event.type = "content_block_start"
    event.content_block = Mock()
    event.content_block.type = "redacted_thinking"
    event.content_block.data = data
    return event


def _plain_block_start_event(block_type="text"):
    event = Mock()
    event.type = "content_block_start"
    event.content_block = Mock()
    event.content_block.type = block_type
    return event


def _stream_for_shape(shape: str):
    events = {
        "thinking": [_delta_event("thinking_delta", thinking="hmm")],
        "redacted_block": [_redacted_block_event()],
        "partial": [
            _delta_event("thinking_delta", thinking="hmm"),
            _redacted_block_event(),
        ],
        "none": [],
    }[shape]
    return iter(
        [_message_start_event(), *events, _delta_event("text_delta", text="Answer")]
    )


def _sync_for_shape(shape: str):
    specs = {
        "thinking": ["thinking"],
        "redacted_block": ["redacted_thinking"],
        "partial": ["thinking", "redacted_thinking"],
        "none": [],
    }[shape]
    blocks = []
    for spec in [*specs, "text"]:
        block = Mock()
        block.type = spec
        if spec == "text":
            block.text = "Answer"
        blocks.append(block)
    msg = Mock()
    msg.id = "msg_1"
    msg.content = blocks
    msg.usage = SimpleNamespace(
        input_tokens=10, output_tokens=100, cache_read_input_tokens=None
    )
    return msg


#: Parametrising over the transport is the point: streaming and non-streaming must resolve
#: equivalent content to the same `reasoning_type`, and writing them as separate suites let that
#: parity drift unchecked.
_MODES = {
    "streaming": (AnthropicMessagesStream, _stream_for_shape),
    "non-streaming": (AnthropicMessages, _sync_for_shape),
}

_SHAPES = ("thinking", "redacted_block", "partial", "none")


def _resolve(mode: str, shape: str, declared=None, model_id="claude-opus-4-7"):
    endpoint_cls, build = _MODES[mode]
    endpoint = endpoint_cls(model_id=model_id, default_reasoning_visibility=declared)
    response = _make_draft_response()
    endpoint.process_raw_response(build(shape), time.perf_counter(), response)
    return response


class TestReasoningTypeResolution:
    """How `reasoning_type` is resolved, for both Anthropic endpoints.

    Whether streamed thinking is raw or summarized is not observable - Claude 3.7 Sonnet returns the
    full thinking output while Claude 4 returns a summary, with identical structure - so it falls
    back to the declared visibility. Redaction *is* observable and takes precedence, including
    alongside readable thinking, since partial redaction still means some reasoning was withheld.
    """

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "shape,expected",
        [
            ("thinking", "summary"),
            ("redacted_block", "redacted"),
            ("partial", "redacted"),
            # No reasoning at all must stay unset, *not* take the endpoint's default
            ("none", None),
        ],
    )
    def test_resolution_by_content_shape(self, mock_client, mode, shape, expected):
        assert _resolve(mode, shape).reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "declared,expected",
        [
            ("verbatim", "verbatim"),
            ("summary", "summary"),
            # `None` means "use this endpoint's default", consistently with the endpoints that
            # infer from the model ID -- it is *not* a way to decline guessing.
            (None, "summary"),
            # Declining is explicit, and must still record that reasoning happened.
            ("unknown", "unknown"),
        ],
    )
    def test_declared_visibility_for_readable_thinking(
        self, mock_client, mode, declared, expected
    ):
        assert _resolve(mode, "thinking", declared=declared).reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    def test_observable_redaction_beats_declared_visibility(self, mock_client, mode):
        response = _resolve(mode, "redacted_block", declared="verbatim")
        assert response.reasoning_type == "redacted"

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize("shape", _SHAPES)
    def test_reasoning_never_leaks_into_response_text(self, mock_client, mode, shape):
        assert _resolve(mode, shape).response_text == "Answer"

    def test_non_streaming_records_no_first_token_metrics(self, mock_client):
        response = _resolve("non-streaming", "thinking")
        assert response.time_to_first_token is None
        assert response.time_to_first_content_token is None


class TestStreamingOnlyReasoningSignals:
    """Signals that exist only in the streamed transport."""

    def test_signature_only_is_redacted_and_sets_ttft(self, mock_client):
        """`display: "omitted"`: the signature is the first output received, so it sets TTFT.

        It arrives *after* thinking completes, so it is a poor proxy for when generation started --
        which is exactly what `reasoning_type="redacted"` flags to the Runner.
        """
        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
        events = [
            _message_start_event(),
            _delta_event("signature_delta", signature="EosnCkYICxIMMb3LzNrMu..."),
            _delta_event("text_delta", text="Answer"),
        ]

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.1, 100.6, 100.8]
            endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.reasoning_type == "redacted"
        assert response.time_to_first_token == pytest.approx(0.6)
        assert response.time_to_first_content_token == pytest.approx(0.8)
        assert response.response_text == "Answer"

    def test_signature_alongside_thinking_does_not_imply_redaction(self, mock_client):
        """A signature accompanies *readable* thinking too, so it is only redaction evidence alone.

        Treating it as redaction unconditionally would mislabel every Claude 4 summarized response.
        """
        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
        events = [
            _message_start_event(),
            _delta_event("thinking_delta", thinking="hmm"),
            _delta_event("signature_delta", signature="Eosn..."),
            _delta_event("text_delta", text="Answer"),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.reasoning_type == "summary"

    def test_fully_redacted_block_is_not_mistaken_for_no_reasoning(self, mock_client):
        """Regression: no thinking_delta *and* no signature_delta.

        Such a response used to report `reasoning_type=None` with TTFT taken from the first *text*
        delta -- claiming the model did no reasoning when it demonstrably did.
        """
        endpoint = AnthropicMessagesStream(model_id="claude-3-7-sonnet")
        events = [
            _message_start_event(),
            _redacted_block_event(),
            _delta_event("text_delta", text="Answer"),
        ]

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.1, 100.4, 100.9]
            endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.reasoning_type == "redacted"
        # TTFT is the redacted block, not the later text delta
        assert response.time_to_first_token == pytest.approx(0.4)
        assert response.time_to_first_content_token == pytest.approx(0.9)

    def test_ordinary_block_starts_do_not_set_timings(self, mock_client):
        """Only `redacted_thinking` starts are timed; text/thinking block starts are structural."""
        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
        events = [
            _message_start_event(),
            _plain_block_start_event("thinking"),
            _plain_block_start_event("text"),
            _delta_event("text_delta", text="Answer"),
        ]

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.1, 100.2, 100.3, 100.9]
            endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.reasoning_type is None
        assert response.time_to_first_token == pytest.approx(0.9)
        assert response.time_to_first_token == response.time_to_first_content_token


# ---------------------------------------------------------------------------
# Tests: thinking that is billed but not identifiable in the response
# ---------------------------------------------------------------------------


class TestAnthropicHiddenThinking:
    """Defence in depth for a stream whose thinking signals never arrive.

    Both Anthropic thinking modes normally leave a structural trace -- `thinking_delta` when
    summarized, `signature_delta` or a `redacted_thinking` block when not -- so this endpoint
    resolves `reasoning_type` from the content in every documented case. But a proxy or gateway can
    forward `usage` while dropping signature deltas, and then `thinking_tokens` is the only evidence
    left. Falling back to it keeps such a response off the whole-output TPOT pairing.
    """

    @staticmethod
    def _stream_without_thinking_signals(thinking_tokens: int | None):
        usage_attrs = {"output_tokens": 100}
        if thinking_tokens is not None:
            usage_attrs["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
        msg_delta = Mock()
        msg_delta.type = "message_delta"
        msg_delta.usage = SimpleNamespace(**usage_attrs)
        return [
            _message_start_event(),
            _delta_event("text_delta", text="Answer"),
            msg_delta,
        ]

    @staticmethod
    def _invoke(endpoint, raw_response):
        """Drive the full `llmeter_invoke` path, which is where the backfill is applied."""
        endpoint._client.messages.create = Mock(return_value=raw_response)
        return endpoint.invoke({"messages": [], "max_tokens": 64})

    def test_streaming_resolves_unknown_from_thinking_tokens(self, mock_client):
        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
        events = self._stream_without_thinking_signals(thinking_tokens=60)

        response = self._invoke(endpoint, iter(events))

        assert response.num_tokens_output_reasoning == 60
        assert response.reasoning_type == "unknown", (
            "thinking tokens were billed, so `None` would wrongly assert no thinking"
        )

    def test_streaming_stays_unset_when_no_thinking_tokens(self, mock_client):
        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
        events = self._stream_without_thinking_signals(thinking_tokens=0)

        response = self._invoke(endpoint, iter(events))

        assert response.reasoning_type is None

    def test_observed_signals_still_win(self, mock_client):
        """`"redacted"` is observed; the token-count fallback must not blur it to `"unknown"`."""
        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
        msg_delta = Mock()
        msg_delta.type = "message_delta"
        msg_delta.usage = SimpleNamespace(
            output_tokens=100, output_tokens_details={"thinking_tokens": 60}
        )
        events = [
            _message_start_event(),
            _delta_event("signature_delta", signature="sig"),
            _delta_event("text_delta", text="Answer"),
            msg_delta,
        ]

        response = self._invoke(endpoint, iter(events))

        assert response.reasoning_type == "redacted"

    def test_non_streaming_resolves_unknown_from_thinking_tokens(self, mock_client):
        """A response body with no thinking block, but a thinking-token charge."""
        endpoint = AnthropicMessages(model_id="claude-opus-4-7")
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Answer"
        message = Mock()
        message.id = "msg_1"
        message.content = [text_block]
        message.usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=100,
            cache_read_input_tokens=None,
            output_tokens_details={"thinking_tokens": 60},
        )

        response = self._invoke(endpoint, message)

        assert response.reasoning_type == "unknown"
