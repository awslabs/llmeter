# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from unittest.mock import Mock, patch

import pytest

from llmeter.endpoints.anthropic_messages import (
    _ANTHROPIC_CLIENTS,
    AnthropicMessages,
    AnthropicMessagesEndpoint,
    AnthropicMessagesStream,
)
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
        mock_client.return_value.messages.create.side_effect = Exception(
            "Stream error"
        )

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

    def test_ttft_visible_tokens_only_true_ignores_thinking(self, mock_client):
        """With ttft_visible_tokens_only=True (default), TTFT is set on first text_delta."""
        endpoint = AnthropicMessagesStream(model_id="test-model")

        events = []

        msg_start = Mock()
        msg_start.type = "message_start"
        msg_start.message = Mock()
        msg_start.message.id = "msg_1"
        msg_start.message.usage = Mock()
        msg_start.message.usage.input_tokens = 5
        msg_start.message.usage.cache_read_input_tokens = None
        events.append(msg_start)

        # Thinking delta — should NOT set TTFT
        thinking = Mock()
        thinking.type = "content_block_delta"
        thinking.delta = Mock()
        thinking.delta.type = "thinking_delta"
        thinking.delta.thinking = "Reasoning..."
        events.append(thinking)

        # Text delta — should set TTFT
        text = Mock()
        text.type = "content_block_delta"
        text.delta = Mock()
        text.delta.type = "text_delta"
        text.delta.text = "Result"
        events.append(text)

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        assert response.time_to_first_token is not None
        assert response.response_text == "Result"

    def test_ttft_visible_tokens_only_false_includes_thinking(self, mock_client):
        """With ttft_visible_tokens_only=False, TTFT is set on first thinking_delta."""
        endpoint = AnthropicMessagesStream(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        events = []

        msg_start = Mock()
        msg_start.type = "message_start"
        msg_start.message = Mock()
        msg_start.message.id = "msg_1"
        msg_start.message.usage = Mock()
        msg_start.message.usage.input_tokens = 5
        msg_start.message.usage.cache_read_input_tokens = None
        events.append(msg_start)

        # Thinking delta — should set TTFT
        thinking = Mock()
        thinking.type = "content_block_delta"
        thinking.delta = Mock()
        thinking.delta.type = "thinking_delta"
        thinking.delta.thinking = "Reasoning..."
        events.append(thinking)

        # Text delta — TTFT already set, should not change it
        text = Mock()
        text.type = "content_block_delta"
        text.delta = Mock()
        text.delta.type = "text_delta"
        text.delta.text = "Result"
        events.append(text)

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        # TTFT was set on the thinking delta, before the text delta
        assert response.time_to_first_token is not None
        assert response.time_to_first_token <= response.time_to_last_token
        assert response.response_text == "Result"

    def test_ttft_signature_delta_with_display_omitted(self, mock_client):
        """With display=omitted, no thinking_delta is emitted — only signature_delta.

        When ttft_visible_tokens_only=False, the signature_delta should set TTFT.
        """
        endpoint = AnthropicMessagesStream(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        events = []

        msg_start = Mock()
        msg_start.type = "message_start"
        msg_start.message = Mock()
        msg_start.message.id = "msg_1"
        msg_start.message.usage = Mock()
        msg_start.message.usage.input_tokens = 5
        msg_start.message.usage.cache_read_input_tokens = None
        events.append(msg_start)

        # signature_delta — the only thinking-block signal in omitted mode
        sig = Mock()
        sig.type = "content_block_delta"
        sig.delta = Mock()
        sig.delta.type = "signature_delta"
        sig.delta.signature = "EosnCkYICxIMMb3LzNrMu..."
        events.append(sig)

        # Text delta
        text = Mock()
        text.type = "content_block_delta"
        text.delta = Mock()
        text.delta.type = "text_delta"
        text.delta.text = "Answer"
        events.append(text)

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        # TTFT was set on the signature_delta, not the text_delta
        assert response.time_to_first_token is not None
        assert response.time_to_first_token <= response.time_to_last_token
        assert response.response_text == "Answer"

    def test_ttft_signature_delta_ignored_when_visible_only(self, mock_client):
        """With ttft_visible_tokens_only=True (default), signature_delta is ignored for TTFT."""
        endpoint = AnthropicMessagesStream(model_id="test-model")

        events = []

        msg_start = Mock()
        msg_start.type = "message_start"
        msg_start.message = Mock()
        msg_start.message.id = "msg_1"
        msg_start.message.usage = Mock()
        msg_start.message.usage.input_tokens = 5
        msg_start.message.usage.cache_read_input_tokens = None
        events.append(msg_start)

        # signature_delta — should NOT set TTFT
        sig = Mock()
        sig.type = "content_block_delta"
        sig.delta = Mock()
        sig.delta.type = "signature_delta"
        sig.delta.signature = "EosnCkYICxIMMb3LzNrMu..."
        events.append(sig)

        # Text delta — should set TTFT
        text = Mock()
        text.type = "content_block_delta"
        text.delta = Mock()
        text.delta.type = "text_delta"
        text.delta.text = "Answer"
        events.append(text)

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        assert response.time_to_first_token is not None
        assert response.response_text == "Answer"


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
