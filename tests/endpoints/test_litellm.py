# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import MagicMock, patch, Mock
from uuid import uuid4
import time

from llmeter.endpoints.litellm import LiteLLM, LiteLLMStreaming, LiteLLMBase
from llmeter.endpoints.base import InvocationResponse


class TestLiteLLMBase:
    """Test the LiteLLMBase class using concrete implementations."""

    @patch('llmeter.endpoints.litellm.get_llm_provider')
    def test_init_with_model_id(self, mock_get_provider):
        """Test initialization with explicit model_id using concrete LiteLLM class."""
        mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)
        
        endpoint = LiteLLM(
            litellm_model="gpt-3.5-turbo",
            model_id="custom-model-id"
        )
        
        assert endpoint.litellm_model == "gpt-3.5-turbo"
        assert endpoint.model_id == "custom-model-id"
        assert endpoint.provider == "openai"
        assert endpoint.endpoint_name == "gpt-3.5-turbo"
        mock_get_provider.assert_called_once_with("gpt-3.5-turbo")

    @patch('llmeter.endpoints.litellm.get_llm_provider')
    def test_init_without_model_id(self, mock_get_provider):
        """Test initialization without explicit model_id using concrete LiteLLM class."""
        mock_get_provider.return_value = ("claude-3", "anthropic", None, None)
        
        endpoint = LiteLLM(litellm_model="claude-3")
        
        assert endpoint.litellm_model == "claude-3"
        assert endpoint.model_id == "claude-3"
        assert endpoint.provider == "anthropic"
        assert endpoint.endpoint_name == "claude-3"

    @patch('llmeter.endpoints.litellm.get_llm_provider')
    def test_parse_payload(self, mock_get_provider):
        """Test _parse_payload method."""
        mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)
        endpoint = LiteLLM(litellm_model="gpt-3.5-turbo")
        
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }
        
        result = endpoint._parse_payload(payload)
        expected = '[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]'
        assert result == expected

    @patch('llmeter.endpoints.litellm.get_llm_provider')
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
            "max_tokens": 256
        }
        assert result == expected

    def test_create_payload_multiple_messages(self):
        """Test create_payload with sequence of messages."""
        messages = ["Hello", "How are you?"]
        result = LiteLLMBase.create_payload(messages)
        
        expected = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"}
            ],
            "max_tokens": 256
        }
        assert result == expected

    def test_create_payload_with_system_message(self):
        """Test create_payload with system message."""
        result = LiteLLMBase.create_payload(
            "Hello",
            system_message="You are a helpful assistant"
        )
        
        expected = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "You are a helpful assistant"}
            ],
            "max_tokens": 256
        }
        assert result == expected

    def test_create_payload_with_custom_max_tokens(self):
        """Test create_payload with custom max_tokens."""
        result = LiteLLMBase.create_payload("Hello", max_tokens=512)
        
        expected = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512
        }
        assert result == expected

    def test_create_payload_with_kwargs(self):
        """Test create_payload with additional kwargs."""
        result = LiteLLMBase.create_payload(
            "Hello",
            temperature=0.7,
            top_p=0.9
        )
        
        expected = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9
        }
        assert result == expected


class TestLiteLLM:
    """Test the LiteLLM class."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('llmeter.endpoints.litellm.get_llm_provider') as mock_get_provider:
            mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)
            self.endpoint = LiteLLM(litellm_model="gpt-3.5-turbo")

    @patch('llmeter.endpoints.litellm.completion')
    def test_invoke_success(self, mock_completion):
        """Test successful invoke."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello there!"
        
        # Create usage mock separately and assign it
        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 10
        usage_mock.completion_tokens = 5
        mock_response.usage = usage_mock
        
        mock_completion.return_value = mock_response

        # Patch isinstance to make the type check pass
        with patch('llmeter.endpoints.litellm.isinstance') as mock_isinstance:
            mock_isinstance.return_value = True
            
            payload = {"messages": [{"role": "user", "content": "Hello"}]}
            result = self.endpoint.invoke(payload)

            assert isinstance(result, InvocationResponse)
            assert result.id == "test-id"
            assert result.response_text == "Hello there!"
            assert result.num_tokens_input == 10
            assert result.num_tokens_output == 5
            assert result.input_prompt == '[{"role": "user", "content": "Hello"}]'
            mock_completion.assert_called_once_with(model="gpt-3.5-turbo", **payload)

    @patch('llmeter.endpoints.litellm.completion')
    def test_invoke_success_no_usage(self, mock_completion):
        """Test successful invoke without usage information."""
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello there!"
        # Remove usage attribute to simulate AttributeError
        del mock_response.usage
        mock_completion.return_value = mock_response

        # Patch isinstance to make the type check pass
        with patch('llmeter.endpoints.litellm.isinstance') as mock_isinstance:
            mock_isinstance.return_value = True
            
            payload = {"messages": [{"role": "user", "content": "Hello"}]}
            result = self.endpoint.invoke(payload)

            assert isinstance(result, InvocationResponse)
            assert result.id == "test-id"
            assert result.response_text == "Hello there!"
            assert result.num_tokens_input is None
            assert result.num_tokens_output is None

    @patch('llmeter.endpoints.litellm.completion')
    def test_invoke_with_kwargs(self, mock_completion):
        """Test invoke with additional kwargs."""
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

        # Patch isinstance to make the type check pass
        with patch('llmeter.endpoints.litellm.isinstance') as mock_isinstance:
            mock_isinstance.return_value = True
            
            payload = {"messages": [{"role": "user", "content": "Hello"}]}
            result = self.endpoint.invoke(payload, temperature=0.7, top_p=0.9)

            mock_completion.assert_called_once_with(
                model="gpt-3.5-turbo",
                temperature=0.7,
                top_p=0.9,
                **payload
            )

    @patch('llmeter.endpoints.litellm.completion')
    def test_invoke_exception(self, mock_completion):
        """Test invoke with exception."""
        mock_completion.side_effect = Exception("API Error")

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.endpoint.invoke(payload)

        assert isinstance(result, InvocationResponse)
        assert result.error == "API Error"
        assert result.input_prompt == '[{"role": "user", "content": "Hello"}]'
        assert result.id is not None and len(result.id) == 32  # UUID hex length

    def test_parse_converse_response(self):
        """Test _parse_converse_response method."""
        mock_response = MagicMock()
        mock_response.id = "response-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        
        # Create usage mock separately and assign it
        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 15
        usage_mock.completion_tokens = 8
        mock_response.usage = usage_mock

        result = self.endpoint._parse_converse_response(mock_response)

        assert isinstance(result, InvocationResponse)
        assert result.id == "response-id"
        assert result.response_text == "Test response"
        assert result.num_tokens_input == 15
        assert result.num_tokens_output == 8

    def test_parse_converse_response_no_usage(self):
        """Test _parse_converse_response without usage info."""
        mock_response = MagicMock()
        mock_response.id = "response-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        # Remove usage attribute to simulate AttributeError
        del mock_response.usage

        result = self.endpoint._parse_converse_response(mock_response)

        assert isinstance(result, InvocationResponse)
        assert result.id == "response-id"
        assert result.response_text == "Test response"
        assert result.num_tokens_input is None
        assert result.num_tokens_output is None


class TestLiteLLMStreaming:
    """Test the LiteLLMStreaming class."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('llmeter.endpoints.litellm.get_llm_provider') as mock_get_provider:
            mock_get_provider.return_value = ("gpt-3.5-turbo", "openai", None, None)
            self.endpoint = LiteLLMStreaming(litellm_model="gpt-3.5-turbo")

    @patch('llmeter.endpoints.litellm.completion')
    @patch('time.perf_counter')
    def test_invoke_success(self, mock_time, mock_completion):
        """Test successful streaming invoke."""
        # Mock time progression: start, first token, final time
        mock_time.side_effect = [0.0, 0.1, 0.2]  # start, first token, end

        # Mock streaming response with proper CustomStreamWrapper type
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
        assert result.time_per_output_token == 0.025  # (0.2-0.1)/(5-1)

        # Check that stream options were set
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    @patch('llmeter.endpoints.litellm.completion')
    def test_invoke_with_stream_in_payload(self, mock_completion):
        """Test invoke when stream is already in payload."""
        from litellm import CustomStreamWrapper
        mock_stream = MagicMock(spec=CustomStreamWrapper)
        mock_stream.__iter__ = lambda self: iter([])
        mock_completion.return_value = mock_stream

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False  # This should be overridden
        }
        self.endpoint.invoke(payload)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["stream"] is True

    @patch('llmeter.endpoints.litellm.completion')
    def test_invoke_with_stream_options_in_payload(self, mock_completion):
        """Test invoke when stream_options is already in payload."""
        from litellm import CustomStreamWrapper
        mock_stream = MagicMock(spec=CustomStreamWrapper)
        mock_stream.__iter__ = lambda self: iter([])
        mock_completion.return_value = mock_stream

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream_options": {"custom": "value"}  # This should be merged with include_usage
        }
        self.endpoint.invoke(payload)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["stream_options"] == {"custom": "value", "include_usage": True}

    @patch('llmeter.endpoints.litellm.completion')
    def test_invoke_exception(self, mock_completion):
        """Test invoke with exception during completion call."""
        mock_completion.side_effect = Exception("Stream error")

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.endpoint.invoke(payload)

        assert isinstance(result, InvocationResponse)
        assert result.error == "Stream error"
        assert result.input_prompt == '[{"role": "user", "content": "Hello"}]'

    @patch('time.perf_counter')
    def test_parse_stream(self, mock_time):
        """Test _parse_stream method."""
        mock_time.side_effect = [0.15, 0.4]  # first token time, last token time
        start_t = 0.0

        # Create mock chunks
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

        result = self.endpoint._parse_stream(mock_stream, start_t)  # type: ignore

        assert isinstance(result, InvocationResponse)
        assert result.id == "test-id"
        assert result.response_text == "First second"
        assert result.num_tokens_input == 8
        assert result.num_tokens_output == 3
        assert result.time_to_first_token == 0.15
        assert result.time_to_last_token == 0.4
        assert result.time_per_output_token == 0.125  # (0.4-0.15)/(3-1)

    @patch('time.perf_counter')
    def test_parse_stream_no_usage(self, mock_time):
        """Test _parse_stream with no usage information."""
        mock_time.side_effect = [0.0, 0.1, 0.2]
        start_t = 0.0

        chunk = MagicMock()
        chunk.id = "test-id"
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Content"
        # Remove usage attribute to simulate AttributeError
        del chunk.usage

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([chunk])

        result = self.endpoint._parse_stream(mock_stream, start_t)  # type: ignore

        assert result.num_tokens_input is None
        assert result.num_tokens_output is None
        assert result.time_per_output_token is None

    @patch('time.perf_counter')
    def test_parse_stream_empty_content(self, mock_time):
        """Test _parse_stream with None content in chunks."""
        mock_time.side_effect = [0.0, 0.1, 0.2]
        start_t = 0.0

        chunk1 = MagicMock()
        chunk1.id = "test-id"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = None  # Empty content
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "Real content"
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 5
        chunk2.usage.completion_tokens = 2

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([chunk1, chunk2])

        result = self.endpoint._parse_stream(mock_stream, start_t)  # type: ignore

        assert result.response_text == "Real content"

    @patch('time.perf_counter')
    def test_parse_stream_single_token_output(self, mock_time):
        """Test _parse_stream with single token output (edge case for time_per_output_token)."""
        mock_time.side_effect = [0.0, 0.1, 0.2]
        start_t = 0.0

        chunk = MagicMock()
        chunk.id = "test-id"
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hi"
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = 5
        chunk.usage.completion_tokens = 1  # Single token

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([chunk])

        result = self.endpoint._parse_stream(mock_stream, start_t)  # type: ignore

        # With 1 token, (num_tokens_output - 1) = 0, so time_per_output_token should be None
        assert result.time_per_output_token is None
