# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.litellm import LiteLLM, LiteLLMBase, LiteLLMStreaming


class TestLiteLLMBase:
    """Test the LiteLLMBase class using concrete implementations."""

    @patch("llmeter.endpoints.litellm.get_llm_provider")
    def test_init_with_model_id(self, mock_get_provider):
        """Test initialization with explicit model_id using concrete LiteLLM class."""
        mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)

        endpoint = LiteLLM(litellm_model="gpt-3.5-turbo", model_id="custom-model-id")

        assert endpoint.litellm_model == "gpt-3.5-turbo"
        assert endpoint.model_id == "custom-model-id"
        assert endpoint.provider == "openai"
        assert endpoint.endpoint_name == "gpt-3.5-turbo"
        mock_get_provider.assert_called_once_with("gpt-3.5-turbo")

    @patch("llmeter.endpoints.litellm.get_llm_provider")
    def test_init_without_model_id(self, mock_get_provider):
        """Test initialization without explicit model_id using concrete LiteLLM class."""
        mock_get_provider.return_value = ("claude-3", "anthropic", None, None)

        endpoint = LiteLLM(litellm_model="claude-3")

        assert endpoint.litellm_model == "claude-3"
        assert endpoint.model_id == "claude-3"
        assert endpoint.provider == "anthropic"
        assert endpoint.endpoint_name == "claude-3"

    @patch("llmeter.endpoints.litellm.get_llm_provider")
    def test_parse_payload(self, mock_get_provider):
        """Test _parse_payload method."""
        mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)
        endpoint = LiteLLM(litellm_model="gpt-3.5-turbo")

        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }

        result = endpoint._parse_payload(payload)
        expected = '[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]'
        assert result == expected

    @patch("llmeter.endpoints.litellm.get_llm_provider")
    def test_parse_payload_empty_messages(self, mock_get_provider):
        """Test _parse_payload with empty messages."""
        mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)
        endpoint = LiteLLM(litellm_model="gpt-3.5-turbo")

        payload = {"messages": []}
        result = endpoint._parse_payload(payload)
        assert result == "[]"

    def test_create_payload_single_message(self):
        """Test create_payload with single string message."""
        result = LiteLLMBase.create_payload("Hello world")

        expected = {
            "messages": [{"role": "user", "content": "Hello world"}],
            "max_tokens": 256,
        }
        assert result == expected

    def test_create_payload_multiple_messages(self):
        """Test create_payload with sequence of messages."""
        messages = ["Hello", "How are you?"]
        result = LiteLLMBase.create_payload(messages)

        expected = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ],
            "max_tokens": 256,
        }
        assert result == expected

    def test_create_payload_with_system_message(self):
        """Test create_payload with system message."""
        result = LiteLLMBase.create_payload(
            "Hello", system_message="You are a helpful assistant"
        )

        expected = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "You are a helpful assistant"},
            ],
            "max_tokens": 256,
        }
        assert result == expected

    def test_create_payload_with_custom_max_tokens(self):
        """Test create_payload with custom max_tokens."""
        result = LiteLLMBase.create_payload("Hello", max_tokens=512)

        expected = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512,
        }
        assert result == expected

    def test_create_payload_with_kwargs(self):
        """Test create_payload with additional kwargs."""
        result = LiteLLMBase.create_payload("Hello", temperature=0.7, top_p=0.9)

        expected = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        assert result == expected


class TestLiteLLM:
    """Test the LiteLLM class."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("llmeter.endpoints.litellm.get_llm_provider") as mock_get_provider:
            mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)
            self.endpoint = LiteLLM(litellm_model="gpt-3.5-turbo")

    @patch("llmeter.endpoints.litellm.completion")
    def test_invoke_success(self, mock_completion):
        """Test successful invoke."""
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello there!"

        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 10
        usage_mock.completion_tokens = 5
        mock_response.usage = usage_mock

        mock_completion.return_value = mock_response

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.endpoint.invoke(payload)

        assert isinstance(result, InvocationResponse)
        assert result.id == "test-id"
        assert result.response_text == "Hello there!"
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 5
        assert result.input_prompt == '[{"role": "user", "content": "Hello"}]'
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-3.5-turbo"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    @patch("llmeter.endpoints.litellm.completion")
    def test_invoke_success_no_usage(self, mock_completion):
        """Test successful invoke without usage information."""
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello there!"
        # Remove usage attribute to simulate AttributeError
        del mock_response.usage
        mock_completion.return_value = mock_response

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.endpoint.invoke(payload)

        assert isinstance(result, InvocationResponse)
        assert result.id == "test-id"
        assert result.response_text == "Hello there!"
        assert result.num_tokens_input is None
        assert result.num_tokens_output is None

    @patch("llmeter.endpoints.litellm.completion")
    def test_invoke_with_kwargs_in_payload(self, mock_completion):
        """Test invoke with additional kwargs passed via payload."""
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"

        # Create usage mock separately and assign it
        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 5
        usage_mock.completion_tokens = 3
        mock_response.usage = usage_mock

        mock_completion.return_value = mock_response

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        self.endpoint.invoke(payload)

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-3.5-turbo"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9

    @patch("llmeter.endpoints.litellm.completion")
    def test_invoke_exception(self, mock_completion):
        """Test invoke with exception."""
        mock_completion.side_effect = Exception("API Error")

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.endpoint.invoke(payload)

        assert isinstance(result, InvocationResponse)
        assert result.error == "API Error"
        assert result.input_prompt == '[{"role": "user", "content": "Hello"}]'
        assert result.id is not None and len(result.id) == 32  # UUID hex length

    def test_process_raw_response(self):
        """Test process_raw_response method."""
        mock_response = MagicMock()
        mock_response.id = "response-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"

        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 15
        usage_mock.completion_tokens = 8
        mock_response.usage = usage_mock

        result = InvocationResponse(id=None, response_text=None)
        self.endpoint.process_raw_response(mock_response, 0.0, result)

        assert isinstance(result, InvocationResponse)
        assert result.id == "response-id"
        assert result.response_text == "Test response"
        assert result.num_tokens_input == 15
        assert result.num_tokens_output == 8
        # Non-streaming: neither first-token metric is measurable
        assert result.time_to_first_token is None
        assert result.time_to_first_content_token is None

    def test_process_raw_response_no_usage(self):
        """Test process_raw_response without usage info."""
        mock_response = MagicMock()
        mock_response.id = "response-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        del mock_response.usage

        result = InvocationResponse(id=None, response_text=None)
        self.endpoint.process_raw_response(mock_response, 0.0, result)

        assert isinstance(result, InvocationResponse)
        assert result.id == "response-id"
        assert result.response_text == "Test response"
        assert result.num_tokens_input is None
        assert result.num_tokens_output is None


class TestLiteLLMStreaming:
    """Test the LiteLLMStreaming class."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("llmeter.endpoints.litellm.get_llm_provider") as mock_get_provider:
            mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)
            self.endpoint = LiteLLMStreaming(litellm_model="gpt-3.5-turbo")

    @patch("llmeter.endpoints.litellm.completion")
    @patch("time.perf_counter")
    def test_invoke_success(self, mock_time, mock_completion):
        """Test successful streaming invoke."""
        # perf_counter calls: wrapper start, chunk1, chunk2, chunk3, wrapper end
        mock_time.side_effect = [0.0, 0.1, 0.15, 0.2, 0.25]

        from litellm import CustomStreamWrapper

        mock_stream = MagicMock(spec=CustomStreamWrapper)
        mock_chunks = []

        # First chunk (with first token)
        chunk1 = MagicMock()
        chunk1.id = "stream-id"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.usage = None
        mock_chunks.append(chunk1)

        # Second chunk (continuation)
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.usage = None
        mock_chunks.append(chunk2)

        # Final chunk (with usage)
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = "!"
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 10
        chunk3.usage.completion_tokens = 5
        mock_chunks.append(chunk3)

        mock_stream.__iter__ = lambda self: iter(mock_chunks)
        mock_completion.return_value = mock_stream

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.endpoint.invoke(payload)

        assert isinstance(result, InvocationResponse)
        assert result.id == "stream-id"
        assert result.response_text == "Hello world!"
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 5
        assert result.time_to_first_token == 0.1
        assert result.time_to_last_token == 0.2
        # time_per_output_token is computed by the runner, not the endpoint
        assert result.time_per_output_token is None

        # Check that stream options were set
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    @patch("llmeter.endpoints.litellm.completion")
    def test_invoke_with_stream_in_payload(self, mock_completion):
        """Test invoke when stream is already in payload."""
        from litellm import CustomStreamWrapper

        mock_stream = MagicMock(spec=CustomStreamWrapper)
        mock_stream.__iter__ = lambda self: iter([])
        mock_completion.return_value = mock_stream

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,  # This should be overridden
        }
        self.endpoint.invoke(payload)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["stream"] is True

    @patch("llmeter.endpoints.litellm.completion")
    def test_invoke_with_stream_options_in_payload(self, mock_completion):
        """Test invoke when stream_options is already in payload."""
        from litellm import CustomStreamWrapper

        mock_stream = MagicMock(spec=CustomStreamWrapper)
        mock_stream.__iter__ = lambda self: iter([])
        mock_completion.return_value = mock_stream

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream_options": {
                "custom": "value"
            },  # This should be merged with include_usage
        }
        self.endpoint.invoke(payload)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["stream_options"] == {
            "custom": "value",
            "include_usage": True,
        }

    @patch("llmeter.endpoints.litellm.completion")
    def test_invoke_exception(self, mock_completion):
        """Test invoke with exception during completion call."""
        mock_completion.side_effect = Exception("Stream error")

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.endpoint.invoke(payload)

        assert isinstance(result, InvocationResponse)
        assert result.error == "Stream error"
        assert result.input_prompt == '[{"role": "user", "content": "Hello"}]'

    @patch("time.perf_counter")
    def test_process_raw_response(self, mock_time):
        """Test process_raw_response method."""
        mock_time.side_effect = [0.15, 0.4]
        start_t = 0.0

        chunk1 = MagicMock()
        chunk1.id = "test-id"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "First"
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " second"
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 8
        chunk2.usage.completion_tokens = 3

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([chunk1, chunk2])

        result = InvocationResponse(id=None, response_text=None)
        self.endpoint.process_raw_response(mock_stream, start_t, result)

        assert isinstance(result, InvocationResponse)
        assert result.id == "test-id"
        assert result.response_text == "First second"
        assert result.num_tokens_input == 8
        assert result.num_tokens_output == 3
        assert result.time_to_first_token == 0.15
        assert result.time_to_last_token == 0.4
        assert result.time_per_output_token is None

    @patch("time.perf_counter")
    def test_process_raw_response_no_usage(self, mock_time):
        """Test process_raw_response with no usage information."""
        mock_time.side_effect = [0.1]
        start_t = 0.0

        chunk = MagicMock()
        chunk.id = "test-id"
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Content"
        del chunk.usage

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([chunk])

        result = InvocationResponse(id=None, response_text=None)
        self.endpoint.process_raw_response(mock_stream, start_t, result)

        assert result.num_tokens_input is None
        assert result.num_tokens_output is None
        assert result.time_per_output_token is None

    @patch("time.perf_counter")
    def test_process_raw_response_empty_content(self, mock_time):
        """Test process_raw_response with None content in chunks."""
        mock_time.side_effect = [0.1, 0.2]
        start_t = 0.0

        chunk1 = MagicMock()
        chunk1.id = "test-id"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = None
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "Real content"
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 5
        chunk2.usage.completion_tokens = 2

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([chunk1, chunk2])

        result = InvocationResponse(id=None, response_text=None)
        self.endpoint.process_raw_response(mock_stream, start_t, result)

        assert result.response_text == "Real content"

    @patch("time.perf_counter")
    def test_process_raw_response_single_token_output(self, mock_time):
        """Test process_raw_response with single token output (edge case for time_per_output_token)."""
        mock_time.side_effect = [0.1]
        start_t = 0.0

        chunk = MagicMock()
        chunk.id = "test-id"
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hi"
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = 5
        chunk.usage.completion_tokens = 1

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([chunk])

        result = InvocationResponse(id=None, response_text=None)
        self.endpoint.process_raw_response(mock_stream, start_t, result)

        # With 1 token, (num_tokens_output - 1) = 0, so time_per_output_token should be None
        assert result.time_per_output_token is None


# ---------------------------------------------------------------------------
# Tests: reasoning models via LiteLLM
# ---------------------------------------------------------------------------


def _ns_chunk(chunk_id="c1", content=None, usage=None, **delta_attrs):
    """Build a LiteLLM-shaped streaming chunk.

    Uses SimpleNamespace rather than MagicMock so only the attributes set here exist -- a
    MagicMock would auto-create `reasoning_content` on every delta.
    """
    delta = SimpleNamespace(content=content, **delta_attrs)
    return SimpleNamespace(
        id=chunk_id, choices=[SimpleNamespace(delta=delta)], usage=usage
    )


def _stream(chunks):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(chunks)
    return stream


class TestLiteLLMStreamingFirstTokenMetrics:
    def setup_method(self):
        with patch("llmeter.endpoints.litellm.get_llm_provider") as mock_get_provider:
            mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)
            self.endpoint = LiteLLMStreaming(litellm_model="gpt-3.5-turbo")

    @patch("time.perf_counter")
    def test_reasoning_content_sets_ttft_only(self, mock_time):
        """LiteLLM's normalized `reasoning_content` sets TTFT but not the content TTFT."""
        mock_time.side_effect = [100.2, 100.6]

        response = InvocationResponse(response_text=None)
        self.endpoint.process_raw_response(
            _stream(
                [
                    _ns_chunk(reasoning_content="thinking..."),
                    _ns_chunk(content="Answer"),
                ]
            ),
            100.0,
            response,
        )

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.6)
        assert response.response_text == "Answer"

    @patch("time.perf_counter")
    def test_thinking_blocks_set_ttft(self, mock_time):
        mock_time.side_effect = [100.3, 100.9]

        response = InvocationResponse(response_text=None)
        self.endpoint.process_raw_response(
            _stream(
                [
                    _ns_chunk(
                        thinking_blocks=[{"type": "thinking", "thinking": "hmm"}]
                    ),
                    _ns_chunk(content="Answer"),
                ]
            ),
            100.0,
            response,
        )

        assert response.time_to_first_token == pytest.approx(0.3)
        assert response.time_to_first_content_token == pytest.approx(0.9)

    def test_reasoning_excluded_from_response_text(self):
        response = InvocationResponse(response_text=None)
        self.endpoint.process_raw_response(
            _stream(
                [
                    _ns_chunk(reasoning_content="internal"),
                    _ns_chunk(content="Visible"),
                ]
            ),
            time.perf_counter(),
            response,
        )

        assert response.response_text == "Visible"

    def test_metrics_equal_without_reasoning(self):
        response = InvocationResponse(response_text=None)
        self.endpoint.process_raw_response(
            _stream([_ns_chunk(content="Hello")]), time.perf_counter(), response
        )

        assert response.time_to_first_token is not None
        assert response.time_to_first_token == response.time_to_first_content_token

    def test_usage_only_chunk_with_empty_choices(self):
        """A final usage-only chunk with no choices must not raise IndexError."""
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
        final = SimpleNamespace(id="c1", choices=[], usage=usage)

        response = InvocationResponse(response_text=None)
        self.endpoint.process_raw_response(
            _stream([_ns_chunk(content="Hi"), final]), time.perf_counter(), response
        )

        assert response.error is None
        assert response.response_text == "Hi"
        assert response.num_tokens_input == 7
        assert response.num_tokens_output == 3


# ---------------------------------------------------------------------------
# Tests: reasoning_type resolution, across both LiteLLM endpoints
# ---------------------------------------------------------------------------


def _sync_response(**message_attrs):
    """A non-streaming LiteLLM response whose message may carry reasoning fields."""
    message = SimpleNamespace(content="Answer", **message_attrs)
    return SimpleNamespace(
        id="resp-1",
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


def _stream_for_shape(shape: str):
    chunks = {
        "reasoning_content": [_ns_chunk(reasoning_content="thinking")],
        "thinking_blocks": [
            _ns_chunk(thinking_blocks=[{"type": "thinking", "thinking": "hmm"}])
        ],
        "none": [],
    }[shape]
    return _stream([*chunks, _ns_chunk(content="Answer")])


def _sync_for_shape(shape: str):
    attrs = {
        "reasoning_content": {"reasoning_content": "thinking"},
        "thinking_blocks": {
            "thinking_blocks": [{"type": "thinking", "thinking": "hmm"}]
        },
        "none": {},
    }[shape]
    return _sync_response(**attrs)


#: Parametrising over the transport is the point: LiteLLM normalizes reasoning onto the same fields
#: for both, so streaming and non-streaming must resolve equivalent content identically.
_MODES = {
    "streaming": (LiteLLMStreaming, _stream_for_shape),
    "non-streaming": (LiteLLM, _sync_for_shape),
}

_SHAPES = ("reasoning_content", "thinking_blocks", "none")


def _resolve(mode: str, shape: str, litellm_model="openai/gpt-oss-120b", declared=None):
    endpoint_cls, build = _MODES[mode]
    with patch("llmeter.endpoints.litellm.get_llm_provider") as mock_provider:
        mock_provider.return_value = (litellm_model, "openai", None, None)
        endpoint = endpoint_cls(
            litellm_model=litellm_model, default_reasoning_visibility=declared
        )
    response = InvocationResponse(response_text=None)
    endpoint.process_raw_response(build(shape), time.perf_counter(), response)
    return response


class TestLiteLLMReasoningTypeResolution:
    """LiteLLM exposes no fidelity marker, so resolution comes from the model string or a declaration.

    Covers the non-streaming endpoint too: it records `reasoning_type` from the response message,
    which was previously untested even though the library sets it.
    """

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "shape,expected",
        [
            ("reasoning_content", "verbatim"),
            ("thinking_blocks", "verbatim"),
            # No reasoning at all must stay unset, *not* take the endpoint's default
            ("none", None),
        ],
    )
    def test_resolution_by_content_shape(self, mode, shape, expected):
        assert _resolve(mode, shape).reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "litellm_model,expected",
        [
            ("anthropic/claude-sonnet-4-6", "summary"),
            ("bedrock/anthropic.claude-opus-4-6", "summary"),
            ("deepseek/deepseek-reasoner", "verbatim"),
            ("openai/gpt-oss-120b", "verbatim"),
        ],
    )
    def test_inferred_from_provider_prefix(self, mode, litellm_model, expected):
        response = _resolve(mode, "reasoning_content", litellm_model=litellm_model)
        assert response.reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "declared,expected",
        [
            ("verbatim", "verbatim"),
            ("summary", "summary"),
            # Declining is explicit, and must still record that reasoning happened
            ("unknown", "unknown"),
        ],
    )
    def test_declared_visibility_overrides_inference(self, mode, declared, expected):
        response = _resolve(
            mode,
            "reasoning_content",
            litellm_model="anthropic/claude-3-7-sonnet",
            declared=declared,
        )
        assert response.reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize("shape", _SHAPES)
    def test_reasoning_never_leaks_into_response_text(self, mode, shape):
        assert _resolve(mode, shape).response_text == "Answer"

    def test_non_streaming_records_no_first_token_metrics(self):
        response = _resolve("non-streaming", "reasoning_content")
        assert response.time_to_first_token is None
        assert response.time_to_first_content_token is None


# ---------------------------------------------------------------------------
# Tests: reasoning-token accounting, and reasoning that is billed but never streamed
# ---------------------------------------------------------------------------


def _usage(completion_tokens=20, reasoning_tokens=None, cached_tokens=None):
    """A LiteLLM `Usage`-shaped object, optionally carrying the token breakdowns."""
    return SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=completion_tokens,
        completion_tokens_details=(
            SimpleNamespace(reasoning_tokens=reasoning_tokens)
            if reasoning_tokens is not None
            else None
        ),
        prompt_tokens_details=(
            SimpleNamespace(cached_tokens=cached_tokens)
            if cached_tokens is not None
            else None
        ),
    )


def _endpoint(endpoint_cls, litellm_model="anthropic/claude-opus-4-7"):
    with patch("llmeter.endpoints.litellm.get_llm_provider") as mock_provider:
        mock_provider.return_value = (litellm_model, "anthropic", None, None)
        return endpoint_cls(litellm_model=litellm_model)


class TestLiteLLMReasoningTokenCounts:
    """LiteLLM normalizes provider reasoning-token counts onto `completion_tokens_details`.

    Without reading it, `num_tokens_output_reasoning` stayed `None`, which made the answer-only TPOT
    pairing uncomputable -- so every summarized/withheld-reasoning run through LiteLLM reported no
    TPOT at all.
    """

    @pytest.mark.parametrize(
        "reasoning_tokens,expected", [(7, 7), (0, 0), (None, None)]
    )
    def test_streaming_extracts_the_breakdown(self, reasoning_tokens, expected):
        endpoint = _endpoint(LiteLLMStreaming)
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(
            _stream(
                [
                    _ns_chunk(content="Answer"),
                    _ns_chunk(usage=_usage(reasoning_tokens=reasoning_tokens)),
                ]
            ),
            time.perf_counter(),
            response,
        )

        assert response.num_tokens_output_reasoning == expected

    @pytest.mark.parametrize(
        "reasoning_tokens,expected", [(7, 7), (0, 0), (None, None)]
    )
    def test_non_streaming_extracts_the_breakdown(self, reasoning_tokens, expected):
        endpoint = _endpoint(LiteLLM)
        raw = _sync_response()
        raw.usage = _usage(reasoning_tokens=reasoning_tokens)
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.num_tokens_output_reasoning == expected

    def test_details_as_mapping_supported(self):
        """Some LiteLLM versions/providers deliver the details as a plain dict."""
        endpoint = _endpoint(LiteLLM)
        raw = _sync_response()
        raw.usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            completion_tokens_details={"reasoning_tokens": 5},
        )
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.num_tokens_output_reasoning == 5

    def test_placeholder_details_are_not_mistaken_for_a_count(self):
        endpoint = _endpoint(LiteLLM)
        raw = _sync_response()
        raw.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.num_tokens_output_reasoning is None

    def test_usage_without_details_attribute(self):
        endpoint = _endpoint(LiteLLM)
        raw = _sync_response()
        raw.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.num_tokens_output_reasoning is None


class TestLiteLLMHiddenReasoning:
    """A gateway can pass usage through while stripping (or renaming) the reasoning fields."""

    def _invoke(
        self, endpoint_cls, chunks_or_raw, litellm_model="anthropic/claude-opus-4-7"
    ):
        endpoint = _endpoint(endpoint_cls, litellm_model=litellm_model)
        with patch("llmeter.endpoints.litellm.completion") as completion:
            completion.return_value = chunks_or_raw
            return endpoint.invoke({"messages": [{"role": "user", "content": "Hi"}]})

    def test_streaming_resolves_unknown_from_token_count(self):
        response = self._invoke(
            LiteLLMStreaming,
            _stream(
                [
                    _ns_chunk(content="Answer"),
                    _ns_chunk(usage=_usage(reasoning_tokens=8)),
                ]
            ),
        )

        assert response.num_tokens_output_reasoning == 8
        assert response.reasoning_type == "unknown"

    def test_streaming_stays_unset_without_reasoning_tokens(self):
        response = self._invoke(
            LiteLLMStreaming,
            _stream(
                [
                    _ns_chunk(content="Answer"),
                    _ns_chunk(usage=_usage(reasoning_tokens=0)),
                ]
            ),
        )

        assert response.reasoning_type is None

    def test_streamed_reasoning_still_wins(self):
        """`"summary"` from the Anthropic model prefix is more precise than `"unknown"`."""
        response = self._invoke(
            LiteLLMStreaming,
            _stream(
                [
                    _ns_chunk(reasoning_content="thinking"),
                    _ns_chunk(content="Answer"),
                    _ns_chunk(usage=_usage(reasoning_tokens=8)),
                ]
            ),
        )

        assert response.reasoning_type == "summary"

    def test_non_streaming_resolves_unknown_from_token_count(self):
        raw = _sync_response()
        raw.usage = _usage(reasoning_tokens=8)

        response = self._invoke(LiteLLM, raw)

        assert response.reasoning_type == "unknown"


class TestLiteLLMCachedInputTokens:
    """LiteLLM normalizes prompt-cache hits onto `prompt_tokens_details.cached_tokens`.

    Prompt caching dominates TTFT, so a benchmark that silently reports `None` here can't explain
    its own latency distribution -- and it makes LLMeter's cache reporting inconsistent between
    LiteLLM and the Bedrock/OpenAI connectors, which have always populated it.
    """

    @pytest.mark.parametrize("cached_tokens,expected", [(6, 6), (0, 0), (None, None)])
    def test_streaming_extracts_cached_tokens(self, cached_tokens, expected):
        endpoint = _endpoint(LiteLLMStreaming)
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(
            _stream(
                [
                    _ns_chunk(content="Answer"),
                    _ns_chunk(usage=_usage(cached_tokens=cached_tokens)),
                ]
            ),
            time.perf_counter(),
            response,
        )

        assert response.num_tokens_input_cached == expected

    @pytest.mark.parametrize("cached_tokens,expected", [(6, 6), (0, 0), (None, None)])
    def test_non_streaming_extracts_cached_tokens(self, cached_tokens, expected):
        endpoint = _endpoint(LiteLLM)
        raw = _sync_response()
        raw.usage = _usage(cached_tokens=cached_tokens)
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.num_tokens_input_cached == expected

    def test_details_as_mapping_supported(self):
        endpoint = _endpoint(LiteLLM)
        raw = _sync_response()
        raw.usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            prompt_tokens_details={"cached_tokens": 4},
        )
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.num_tokens_input_cached == 4

    def test_placeholder_details_are_not_mistaken_for_a_count(self):
        endpoint = _endpoint(LiteLLM)
        raw = _sync_response()
        raw.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.num_tokens_input_cached is None

    def test_both_breakdowns_coexist(self):
        """Reading one must not clobber the other."""
        endpoint = _endpoint(LiteLLMStreaming)
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(
            _stream(
                [
                    _ns_chunk(content="Answer"),
                    _ns_chunk(usage=_usage(reasoning_tokens=8, cached_tokens=6)),
                ]
            ),
            time.perf_counter(),
            response,
        )

        assert response.num_tokens_output_reasoning == 8
        assert response.num_tokens_input_cached == 6


class TestLiteLLMUsageAgainstRealSdkTypes:
    """Pin the field paths against the installed `litellm` types, not just hand-built stubs.

    A stub can't catch LiteLLM renaming a wrapper field; constructing the real `Usage` object can.
    """

    def test_real_usage_object_is_parsed(self):
        from litellm.types.utils import (
            CompletionTokensDetailsWrapper,
            PromptTokensDetailsWrapper,
            Usage,
        )

        endpoint = _endpoint(LiteLLM)
        raw = _sync_response()
        raw.usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=8
            ),
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=6),
        )
        response = InvocationResponse(response_text=None)

        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 20
        assert response.num_tokens_output_reasoning == 8
        assert response.num_tokens_input_cached == 6
