# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from llmeter.endpoints.anthropic_messages import (
    AnthropicMessages,
    AnthropicMessagesEndpoint,
    AnthropicMessagesStream,
    _build_anthropic_client,
)
from llmeter.endpoints.base import InvocationResponse


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


# ---------------------------------------------------------------------------
# Tests: _build_anthropic_client
# ---------------------------------------------------------------------------


class TestBuildClient:
    @patch("llmeter.endpoints.anthropic_messages.anthropic")
    def test_build_anthropic_client(self, mock_anthropic):
        _build_anthropic_client(provider="anthropic", api_key="test-key")
        mock_anthropic.Anthropic.assert_called_once_with(api_key="test-key")

    @patch("llmeter.endpoints.anthropic_messages.anthropic")
    def test_build_bedrock_client(self, mock_anthropic):
        _build_anthropic_client(provider="bedrock", aws_region="us-west-2")
        mock_anthropic.AnthropicBedrock.assert_called_once_with(
            aws_region="us-west-2"
        )

    @patch("llmeter.endpoints.anthropic_messages.anthropic")
    def test_build_bedrock_mantle_client(self, mock_anthropic):
        _build_anthropic_client(provider="bedrock-mantle", aws_region="us-east-1")
        mock_anthropic.AnthropicBedrockMantle.assert_called_once_with(
            aws_region="us-east-1"
        )

    @patch("llmeter.endpoints.anthropic_messages.anthropic")
    def test_build_unknown_provider_raises(self, mock_anthropic):
        with pytest.raises(ValueError, match="Unknown provider"):
            _build_anthropic_client(provider="unknown")


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


# ---------------------------------------------------------------------------
# Tests: _parse_payload
# ---------------------------------------------------------------------------


class TestParsePayload:
    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_string_content(self, mock_build):
        endpoint = AnthropicMessages(model_id="test-model")
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        assert endpoint._parse_payload(payload) == "Hello"

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_block_content(self, mock_build):
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

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_empty_messages(self, mock_build):
        endpoint = AnthropicMessages(model_id="test-model")
        assert endpoint._parse_payload({"messages": []}) == ""

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_no_messages_key(self, mock_build):
        endpoint = AnthropicMessages(model_id="test-model")
        assert endpoint._parse_payload({}) == ""

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_mixed_content_types(self, mock_build):
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

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_multi_turn(self, mock_build):
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
    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response(self, mock_build):
        endpoint = AnthropicMessages(model_id="test-model")
        mock_response = _make_mock_message()

        result = endpoint.parse_response(mock_response, time.perf_counter())

        assert isinstance(result, InvocationResponse)
        assert result.id == "msg_test123"
        assert result.response_text == "Hello, world!"
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 5

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response_with_cache(self, mock_build):
        endpoint = AnthropicMessages(model_id="test-model")
        mock_response = _make_mock_message(cache_read_input_tokens=3)

        result = endpoint.parse_response(mock_response, time.perf_counter())

        assert result.num_tokens_input_cached == 3

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response_no_usage(self, mock_build):
        endpoint = AnthropicMessages(model_id="test-model")
        mock_response = _make_mock_message()
        mock_response.usage = None

        result = endpoint.parse_response(mock_response, time.perf_counter())

        assert result.num_tokens_input is None
        assert result.num_tokens_output is None
        assert result.num_tokens_input_cached is None

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response_multiple_text_blocks(self, mock_build):
        endpoint = AnthropicMessages(model_id="test-model")

        block1 = Mock()
        block1.type = "text"
        block1.text = "Part 1. "

        block2 = Mock()
        block2.type = "text"
        block2.text = "Part 2."

        mock_response = _make_mock_message()
        mock_response.content = [block1, block2]

        result = endpoint.parse_response(mock_response, time.perf_counter())
        assert result.response_text == "Part 1. Part 2."

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_invoke_success(self, mock_build):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message()

        endpoint = AnthropicMessages(model_id="test-model")
        result = endpoint.invoke(
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}
        )

        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hello, world!"
        assert result.error is None

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_invoke_api_error(self, mock_build):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        endpoint = AnthropicMessages(model_id="test-model")
        result = endpoint.invoke(
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}
        )

        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "API error" in result.error

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_prepare_payload_sets_model(self, mock_build):
        endpoint = AnthropicMessages(model_id="claude-opus-4-7")
        payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 256}

        prepared = endpoint.prepare_payload(payload)

        assert prepared["model"] == "claude-opus-4-7"

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_prepare_payload_merges_kwargs(self, mock_build):
        endpoint = AnthropicMessages(model_id="test-model")
        payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 256}

        prepared = endpoint.prepare_payload(payload, temperature=0.5)

        assert prepared["temperature"] == 0.5
        assert prepared["model"] == "test-model"


# ---------------------------------------------------------------------------
# Tests: AnthropicMessagesStream (streaming)
# ---------------------------------------------------------------------------


class TestAnthropicMessagesStream:
    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response_basic(self, mock_build):
        endpoint = AnthropicMessagesStream(model_id="test-model")
        events = _make_stream_events()

        start_t = time.perf_counter()
        result = endpoint.parse_response(iter(events), start_t)

        assert isinstance(result, InvocationResponse)
        assert result.id == "msg_stream123"
        assert result.response_text == "Hello, world!"
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 5
        assert result.time_to_first_token is not None
        assert result.time_to_last_token is not None
        assert result.time_to_first_token <= result.time_to_last_token

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response_with_cache(self, mock_build):
        endpoint = AnthropicMessagesStream(model_id="test-model")
        events = _make_stream_events(cache_read_input_tokens=7)

        result = endpoint.parse_response(iter(events), time.perf_counter())

        assert result.num_tokens_input_cached == 7

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response_empty_stream(self, mock_build):
        endpoint = AnthropicMessagesStream(model_id="test-model")

        start_t = time.perf_counter()
        result = endpoint.parse_response(iter([]), start_t)

        assert isinstance(result, InvocationResponse)
        assert result.response_text == ""
        assert result.id is None
        assert result.time_to_first_token == result.time_to_last_token

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response_no_text_deltas(self, mock_build):
        """Stream with message_start and message_delta but no text content."""
        endpoint = AnthropicMessagesStream(model_id="test-model")
        events = _make_stream_events(text_chunks=[])

        result = endpoint.parse_response(iter(events), time.perf_counter())

        assert result.response_text == ""
        # TTFT should equal TTLT when no text was received
        assert result.time_to_first_token == result.time_to_last_token

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response_timing(self, mock_build):
        """Verify TTFT is captured on the first text delta."""
        endpoint = AnthropicMessagesStream(model_id="test-model")
        events = _make_stream_events(text_chunks=["First", " Second"])

        start_t = time.perf_counter()
        result = endpoint.parse_response(iter(events), start_t)

        assert result.time_to_first_token is not None
        assert result.time_to_last_token is not None
        assert result.time_to_first_token > 0
        assert result.time_to_last_token >= result.time_to_first_token

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_prepare_payload_sets_stream(self, mock_build):
        endpoint = AnthropicMessagesStream(model_id="test-model")
        payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 256}

        prepared = endpoint.prepare_payload(payload)

        assert prepared["stream"] is True
        assert prepared["model"] == "test-model"

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_invoke_success(self, mock_build):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        mock_client.messages.create.return_value = iter(
            _make_stream_events(text_chunks=["Hi!"])
        )

        endpoint = AnthropicMessagesStream(model_id="test-model")
        result = endpoint.invoke(
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}
        )

        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hi!"
        assert result.error is None

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_invoke_api_error(self, mock_build):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("Stream error")

        endpoint = AnthropicMessagesStream(model_id="test-model")
        result = endpoint.invoke(
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 256}
        )

        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "Stream error" in result.error

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_parse_response_ignores_non_text_deltas(self, mock_build):
        """Verify that non-text deltas (e.g. thinking, input_json) are ignored."""
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

        # A thinking delta (should be ignored)
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

        result = endpoint.parse_response(iter(events), time.perf_counter())

        assert result.response_text == "Answer"
        assert result.num_tokens_output == 3


# ---------------------------------------------------------------------------
# Tests: Endpoint initialization
# ---------------------------------------------------------------------------


class TestEndpointInit:
    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_default_provider(self, mock_build):
        endpoint = AnthropicMessages(model_id="claude-opus-4-7")
        assert endpoint.provider == "anthropic"
        assert endpoint.model_id == "claude-opus-4-7"
        assert endpoint.endpoint_name == "anthropic-messages"

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_bedrock_provider(self, mock_build):
        endpoint = AnthropicMessages(
            model_id="global.anthropic.claude-opus-4-6-v1",
            provider="bedrock",
            aws_region="us-west-2",
        )
        assert endpoint.provider == "bedrock"
        assert endpoint.aws_region == "us-west-2"
        mock_build.assert_called_once_with(
            provider="bedrock",
            api_key=None,
            aws_region="us-west-2",
        )

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_bedrock_mantle_provider(self, mock_build):
        endpoint = AnthropicMessages(
            model_id="anthropic.claude-opus-4-7",
            provider="bedrock-mantle",
            aws_region="us-east-1",
        )
        assert endpoint.provider == "bedrock-mantle"
        mock_build.assert_called_once_with(
            provider="bedrock-mantle",
            api_key=None,
            aws_region="us-east-1",
        )

    @patch("llmeter.endpoints.anthropic_messages._build_anthropic_client")
    def test_custom_endpoint_name(self, mock_build):
        endpoint = AnthropicMessages(
            model_id="test-model", endpoint_name="my-custom-endpoint"
        )
        assert endpoint.endpoint_name == "my-custom-endpoint"
