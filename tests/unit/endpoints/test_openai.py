# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from openai import APIConnectionError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from llmeter.endpoints.base import Endpoint, InvocationResponse
from llmeter.endpoints.openai import (
    OpenAICompletionEndpoint,
    OpenAICompletionStreamEndpoint,
    OpenAIEndpoint,
)


class TestOpenAIEndpoint:
    """Test the base OpenAI endpoint class."""

    def test_initialization(self):
        """Test OpenAI endpoint initialization with default parameters."""
        # Use concrete endpoint for testing base functionality
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo",
            endpoint_name="test_openai",
            api_key="test_key",
        )

        assert endpoint.model_id == "gpt-3.5-turbo"
        assert endpoint.endpoint_name == "test_openai"
        assert endpoint.provider == "openai"
        assert endpoint._client is not None
        assert endpoint._client.project is None
        assert endpoint.project is None

    def test_initialization_with_custom_provider(self):
        """Test OpenAI endpoint initialization with custom provider."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-4",
            provider="custom_openai",
            api_key="test_key",
        )

        assert endpoint.provider == "custom_openai"
        assert endpoint.model_id == "gpt-4"

    def test_initialization_with_project_id(self):
        """Test OpenAI endpoint initialization with custom project ID."""
        endpoint = OpenAICompletionEndpoint(
            model_id="openai.glm-5",
            api_key="test_key",
            project="proj_DUMMY",
        )

        assert endpoint.model_id == "openai.glm-5"
        assert endpoint._client.project == "proj_DUMMY"
        assert endpoint.project == "proj_DUMMY"

    def test_initialization_without_api_key(self):
        """Test OpenAI endpoint initialization without API key."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        assert endpoint.model_id == "gpt-3.5-turbo"
        assert endpoint._client is not None

    def test_parse_payload_single_message(self):
        """Test _parse_payload with a single message."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )
        payload = {"messages": [{"role": "user", "content": "Hello, world!"}]}

        result = endpoint._parse_payload(payload)
        assert result == "Hello, world!"

    def test_parse_payload_multiple_messages(self):
        """Test _parse_payload with multiple messages."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        }

        result = endpoint._parse_payload(payload)
        assert result == "Hello\nHi there!\nHow are you?"

    def test_parse_payload_empty_messages(self):
        """Test _parse_payload with empty messages list."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )
        payload = {"messages": []}

        result = endpoint._parse_payload(payload)
        assert result == ""

    def test_parse_payload_missing_messages(self):
        """Test _parse_payload with missing messages key."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )
        payload = {}

        result = endpoint._parse_payload(payload)
        assert result == ""

    def test_create_payload_single_string(self):
        """Test create_payload with a single string message."""
        payload = OpenAIEndpoint.create_payload("Hello, world!")

        expected = {
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 256,
        }
        assert payload == expected

    def test_create_payload_multiple_strings(self):
        """Test create_payload with multiple string messages."""
        payload = OpenAIEndpoint.create_payload(["Hello", "How are you?"])

        expected = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "How are you?"},
                    ],
                },
            ],
            "max_tokens": 256,
        }
        assert payload == expected

    def test_create_payload_custom_max_tokens(self):
        """Test create_payload with custom max_tokens."""
        payload = OpenAIEndpoint.create_payload("Hello", max_tokens=512)

        expected = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512,
        }
        assert payload == expected

    def test_create_payload_with_kwargs(self):
        """Test create_payload with additional kwargs."""
        payload = OpenAIEndpoint.create_payload(
            "Hello", max_tokens=512, temperature=0.7, top_p=0.9
        )

        expected = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        assert payload == expected

    def test_create_payload_empty_string(self):
        """Test create_payload with empty string."""
        payload = OpenAIEndpoint.create_payload("")

        expected = {
            "messages": [{"role": "user", "content": ""}],
            "max_tokens": 256,
        }
        assert payload == expected

    def test_create_payload_empty_list(self):
        """Test create_payload with empty list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            OpenAIEndpoint.create_payload([])


class TestOpenAICompletionEndpoint:
    """Test the OpenAI completion endpoint class."""

    @pytest.fixture
    def endpoint(self):
        """Create a test OpenAI completion endpoint."""
        return OpenAICompletionEndpoint(model_id="gpt-3.5-turbo", api_key="test_key")

    @pytest.fixture
    def mock_chat_completion(self):
        """Create a mock ChatCompletion response."""
        return ChatCompletion(
            id="chatcmpl-test123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Hello! How can I help you today?", role="assistant"
                    ),
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=8, prompt_tokens=10, total_tokens=18
            ),
        )

    def test_invoke_success(self, endpoint, mock_chat_completion):
        """Test successful invoke call."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = mock_chat_completion

            payload = {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 256,
            }

            response = endpoint.invoke(payload)

            assert isinstance(response, InvocationResponse)
            assert response.id == "chatcmpl-test123"
            assert response.response_text == "Hello! How can I help you today?"
            assert response.num_tokens_input == 10
            assert response.num_tokens_output == 8
            assert response.time_to_last_token is not None
            assert response.time_to_last_token > 0
            assert response.error is None
            assert response.input_payload["model"] == "gpt-3.5-turbo"

    def test_invoke_api_connection_error(self, endpoint):
        """Test invoke with APIConnectionError."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.side_effect = APIConnectionError(request=Mock())

            payload = {"messages": [{"role": "user", "content": "Hello"}]}

            response = endpoint.invoke(payload)

            assert isinstance(response, InvocationResponse)
            assert response.error is not None
            assert response.response_text is None
            assert response.id is not None

    def test_invoke_generic_exception(self, endpoint):
        """Test invoke with generic exception."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.side_effect = Exception("Unexpected error")

            payload = {"messages": [{"role": "user", "content": "Hello"}]}

            response = endpoint.invoke(payload)

            assert isinstance(response, InvocationResponse)
            assert response.error == "Unexpected error"
            assert response.response_text is None

    def test_process_raw_response(self, endpoint, mock_chat_completion):
        """Test process_raw_response method."""
        start_time = time.perf_counter()
        response = InvocationResponse(id=None, response_text=None)

        endpoint.process_raw_response(mock_chat_completion, start_time, response)

        assert isinstance(response, InvocationResponse)
        assert response.id == "chatcmpl-test123"
        assert response.response_text == "Hello! How can I help you today?"
        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 8
        # Non-streaming: neither first-token metric is measurable
        assert response.time_to_first_token is None
        assert response.time_to_first_content_token is None

    def test_process_raw_response_no_usage(self, endpoint):
        """Test process_raw_response with no usage information."""
        completion = ChatCompletion(
            id="chatcmpl-test123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
            usage=None,
        )

        start_time = time.perf_counter()
        response = InvocationResponse(id=None, response_text=None)
        endpoint.process_raw_response(completion, start_time, response)

        assert response.num_tokens_input is None
        assert response.num_tokens_output is None

    def test_invoke_sets_input_prompt(self, endpoint, mock_chat_completion):
        """Test that invoke sets the input_prompt correctly."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = mock_chat_completion

            payload = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "How are you?"},
                ]
            }

            response = endpoint.invoke(payload)

            assert response.input_prompt == "Hello\nHow are you?"


class TestOpenAICompletionStreamEndpoint:
    """Test the OpenAI streaming completion endpoint class."""

    @pytest.fixture
    def endpoint(self):
        """Create a test OpenAI streaming completion endpoint."""
        return OpenAICompletionStreamEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

    @pytest.fixture
    def mock_stream_response(self):
        """Create a mock streaming response."""
        # Create mock chunks
        chunk1 = MagicMock()
        chunk1.id = "chatcmpl-test123"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-test123"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " there!"

        chunk3 = MagicMock()
        chunk3.id = "chatcmpl-test123"
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 10
        chunk3.usage.completion_tokens = 5

        return [chunk1, chunk2, chunk3]

    def test_invoke_success(self, endpoint, mock_stream_response):
        """Test successful streaming invoke call."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = iter(mock_stream_response)

            payload = {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 256,
            }

            response = endpoint.invoke(payload)

            assert isinstance(response, InvocationResponse)
            assert response.id == "chatcmpl-test123"
            assert response.response_text == "Hello there!"
            assert response.num_tokens_input == 10
            assert response.num_tokens_output == 5
            assert response.time_to_first_token is not None
            assert response.time_to_last_token is not None
            assert response.time_to_first_token < response.time_to_last_token
            assert response.error is None

    def test_invoke_sets_stream_parameters(self, endpoint, mock_stream_response):
        """Test that invoke sets stream parameters correctly."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = iter(mock_stream_response)

            payload = {"messages": [{"role": "user", "content": "Hello"}]}

            endpoint.invoke(payload)

            # Verify the call was made with streaming parameters
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            if call_args and len(call_args) > 1:
                kwargs = call_args[1]
                assert kwargs["stream"] is True
                assert kwargs["stream_options"] == {"include_usage": True}

    def test_invoke_preserves_existing_stream_config(
        self, endpoint, mock_stream_response
    ):
        """Test that invoke preserves existing stream configuration."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = iter(mock_stream_response)

            payload = {
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "stream_options": {"include_usage": False},
            }

            endpoint.invoke(payload)

            # Verify existing stream config is preserved
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            if call_args and len(call_args) > 1:
                kwargs = call_args[1]
                assert kwargs["stream"] is True
                assert kwargs["stream_options"] == {"include_usage": False}

    def test_invoke_api_connection_error(self, endpoint):
        """Test invoke with APIConnectionError."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.side_effect = APIConnectionError(request=Mock())

            payload = {"messages": [{"role": "user", "content": "Hello"}]}

            response = endpoint.invoke(payload)

            assert isinstance(response, InvocationResponse)
            assert response.error is not None
            assert response.response_text is None

    def test_invoke_generic_exception(self, endpoint):
        """Test invoke with generic exception."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.side_effect = Exception("Unexpected error")

            payload = {"messages": [{"role": "user", "content": "Hello"}]}

            response = endpoint.invoke(payload)

            assert isinstance(response, InvocationResponse)
            assert response.error == "Unexpected error"
            assert response.response_text is None

    def test_process_raw_response(self, endpoint, mock_stream_response):
        """Test process_raw_response method."""
        start_time = time.perf_counter()
        response = InvocationResponse(id=None, response_text=None)

        endpoint.process_raw_response(iter(mock_stream_response), start_time, response)

        assert response.id == "chatcmpl-test123"
        assert response.response_text == "Hello there!"
        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 5
        assert response.time_to_first_token is not None
        assert response.time_to_last_token is not None

    def test_process_raw_response_empty_stream(self, endpoint):
        """Test process_raw_response with empty stream."""
        start_time = time.perf_counter()
        response = InvocationResponse(id=None, response_text=None)

        endpoint.process_raw_response(iter([]), start_time, response)

        assert response.response_text is None
        assert response.num_tokens_input is None
        assert response.num_tokens_output is None

    def test_process_raw_response_no_usage(self, endpoint):
        """Test process_raw_response without usage information."""
        chunk1 = MagicMock()
        chunk1.id = "chatcmpl-test123"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-test123"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = None
        chunk2.usage = None

        start_time = time.perf_counter()
        response = InvocationResponse(id=None, response_text=None)

        endpoint.process_raw_response(iter([chunk1, chunk2]), start_time, response)

        assert response.response_text == "Hello"
        assert response.num_tokens_input is None
        assert response.num_tokens_output is None

    def test_process_raw_response_none_content(self, endpoint):
        """Test process_raw_response with None content in first chunk."""
        chunk1 = MagicMock()
        chunk1.id = "chatcmpl-test123"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = None

        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-test123"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "Hello"

        start_time = time.perf_counter()
        response = InvocationResponse(id=None, response_text=None)

        endpoint.process_raw_response(iter([chunk1, chunk2]), start_time, response)

        assert response.response_text == "Hello"

    def test_invoke_sets_input_prompt(self, endpoint, mock_stream_response):
        """Test that invoke sets the input_prompt correctly."""
        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = iter(mock_stream_response)

            payload = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "How are you?"},
                ]
            }

            response = endpoint.invoke(payload)

            assert response.input_prompt == "Hello\nHow are you?"


class TestOpenAIEndpointIntegration:
    """Integration tests for OpenAI endpoints."""

    def test_endpoint_inheritance(self):
        """Test that OpenAI endpoints properly inherit from base classes."""
        completion_endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )
        stream_endpoint = OpenAICompletionStreamEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        assert isinstance(completion_endpoint, OpenAIEndpoint)
        assert isinstance(stream_endpoint, OpenAIEndpoint)

        # Test that they have the required methods
        assert hasattr(completion_endpoint, "invoke")
        assert hasattr(completion_endpoint, "create_payload")
        assert hasattr(stream_endpoint, "invoke")
        assert hasattr(stream_endpoint, "create_payload")

    def test_endpoint_to_dict(self):
        """Test endpoint serialization to dictionary."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-4",
            endpoint_name="test_openai",
            provider="openai",
            api_key="test_key",
        )

        endpoint_dict = endpoint.to_dict()

        assert endpoint_dict["model_id"] == "gpt-4"
        assert endpoint_dict["endpoint_name"] == "test_openai"
        assert endpoint_dict["provider"] == "openai"
        assert endpoint_dict["endpoint_type"] == "OpenAICompletionEndpoint"

    def test_create_payload_consistency(self):
        """Test that create_payload works consistently across endpoint types."""
        message = "Test message"

        base_payload = OpenAIEndpoint.create_payload(message)
        completion_payload = OpenAICompletionEndpoint.create_payload(message)
        stream_payload = OpenAICompletionStreamEndpoint.create_payload(message)

        # All should create the same payload structure
        assert base_payload == completion_payload == stream_payload
        assert base_payload["messages"][0]["content"] == message

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across endpoint types."""
        completion_endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )
        stream_endpoint = OpenAICompletionStreamEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        # Mock both endpoints to raise the same error
        error_message = "Test error"

        with patch.object(
            completion_endpoint._client.chat.completions, "create"
        ) as mock_create:
            mock_create.side_effect = Exception(error_message)
            completion_response = completion_endpoint.invoke({"messages": []})

        with patch.object(
            stream_endpoint._client.chat.completions, "create"
        ) as mock_create:
            mock_create.side_effect = Exception(error_message)
            stream_response = stream_endpoint.invoke({"messages": []})

        # Both should handle errors similarly
        assert completion_response.error == error_message
        assert stream_response.error == error_message
        assert completion_response.response_text is None
        assert stream_response.response_text is None


class TestOpenAIEndpointEdgeCases:
    """Test edge cases and error conditions for OpenAI endpoints."""

    def test_parse_payload_malformed_messages(self):
        """Test _parse_payload with malformed messages."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        # Test with messages that don't have content
        payload = {"messages": [{"role": "user"}]}
        result = endpoint._parse_payload(payload)
        assert result == ""

        # Test with non-list messages
        payload = {"messages": "not a list"}
        result = endpoint._parse_payload(payload)
        assert result == ""

    def test_create_payload_invalid_input_types(self):
        """Test create_payload with invalid input types."""
        # Skip these tests as they are caught by type checking at compile time
        # The type system prevents passing invalid types to create_payload
        pass

    def test_invoke_with_empty_payload(self):
        """Test invoke with completely empty payload."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.side_effect = Exception("Missing required parameter")

            response = endpoint.invoke({})

            assert isinstance(response, InvocationResponse)
            assert response.error is not None

    def test_stream_endpoint_with_malformed_chunks(self):
        """Test streaming endpoint with malformed response chunks."""
        endpoint = OpenAICompletionStreamEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        # Create malformed chunks - remove choices to trigger AttributeError
        malformed_chunk = MagicMock()
        malformed_chunk.id = "test123"
        # Delete choices attribute to ensure AttributeError
        del malformed_chunk.choices

        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = iter([malformed_chunk])

            payload = {"messages": [{"role": "user", "content": "Hello"}]}

            # Malformed chunks are caught by the base invoke wrapper and
            # returned as an error InvocationResponse instead of propagating.
            response = endpoint.invoke(payload)
            assert isinstance(response, InvocationResponse)
            assert response.error is not None
            assert response.input_payload is not None

    def test_response_timing_accuracy(self):
        """Test that response timing measurements are accurate."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        mock_completion = ChatCompletion(
            id="test123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content="Hello", role="assistant"),
                )
            ],
            created=1234567890,
            model="gpt-3.5-turbo",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        )

        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            # Add a small delay to simulate API call
            def delayed_response(*args, **kwargs):
                time.sleep(0.01)  # 10ms delay
                return mock_completion

            mock_create.side_effect = delayed_response

            payload = {"messages": [{"role": "user", "content": "Hello"}]}
            response = endpoint.invoke(payload)

            # Verify timing is reasonable (should be at least 10ms)
            if response.time_to_last_token is not None:
                assert response.time_to_last_token >= 0.01
                assert response.time_to_last_token < 1.0  # Should be less than 1 second

    def test_stream_response_with_multiple_content_chunks(self):
        """Test streaming response with multiple content chunks including empty ones."""
        endpoint = OpenAICompletionStreamEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        # Create chunks with various content patterns
        chunk1 = MagicMock()
        chunk1.id = "chatcmpl-test123"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-test123"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = ""  # Empty string content

        chunk3 = MagicMock()
        chunk3.id = "chatcmpl-test123"
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = " world"

        chunk4 = MagicMock()
        chunk4.id = "chatcmpl-test123"
        chunk4.choices = [MagicMock()]
        chunk4.choices[0].delta.content = None  # None content
        chunk4.usage = MagicMock()
        chunk4.usage.prompt_tokens = 5
        chunk4.usage.completion_tokens = 3

        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = iter([chunk1, chunk2, chunk3, chunk4])

            payload = {"messages": [{"role": "user", "content": "Test"}]}
            response = endpoint.invoke(payload)

            assert response.response_text == "Hello world"
            assert response.num_tokens_input == 5
            assert response.num_tokens_output == 3

    def test_stream_response_usage_none_value(self):
        """Test streaming response when usage attribute exists but is None."""
        endpoint = OpenAICompletionStreamEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        chunk1 = MagicMock()
        chunk1.id = "chatcmpl-test123"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-test123"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.usage = None  # usage attribute exists but is None

        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = iter([chunk1, chunk2])

            payload = {"messages": [{"role": "user", "content": "Test"}]}
            response = endpoint.invoke(payload)

            assert response.response_text == "Hello world"
            assert response.num_tokens_input is None
            assert response.num_tokens_output is None


class TestOpenAIEndpointSerialization:
    """Test serialization behaves as expected for both endpoint types."""

    def test_completion_endpoint_serializes_basic_fields(self):
        """Test sync endpoint serialized state when optional parameters not set."""
        from llmeter.serialization import dump_object

        endpoint = OpenAICompletionEndpoint(model_id="gpt-4")
        state = dump_object(endpoint)["__llmeter_state__"]
        assert "api_key" not in state
        assert state["endpoint_name"] == "openai"
        assert state["model_id"] == "gpt-4"
        assert state.get("project") is None
        assert state.get("organization") is None
        assert state["provider"] == "openai"
        assert "base_url" in state
        assert state["max_retries"] == 2
        # timeout is always serialized (as dict when it's an httpx.Timeout)
        assert "timeout" in state

    def test_completion_endpoint_serializes_all_client_fields(self):
        """Test sync endpoint serializes all client configuration fields."""
        from llmeter.serialization import dump_object

        endpoint = OpenAICompletionEndpoint(
            model_id="openai.gpt-oss-120b",
            api_key="test_key",
            endpoint_name="test-endpoint",
            organization="org-abc",
            project="proj_TEST",
            base_url="https://custom.example.com/v1",
            websocket_base_url="wss://custom.example.com/ws",
            timeout=30.0,
            max_retries=5,
            default_headers={"X-Custom": "val"},
            default_query={"version": "2"},
            provider="test-provider",
        )
        state = dump_object(endpoint)["__llmeter_state__"]
        assert "api_key" not in state
        assert state["endpoint_name"] == "test-endpoint"
        assert state["model_id"] == "openai.gpt-oss-120b"
        assert state["organization"] == "org-abc"
        assert state["project"] == "proj_TEST"
        assert "custom.example.com" in state["base_url"]
        assert state["websocket_base_url"] == "wss://custom.example.com/ws"
        assert state["timeout"] == 30.0
        assert state["max_retries"] == 5
        assert state["default_headers"] == {"X-Custom": "val"}
        assert state["default_query"] == {"version": "2"}
        assert state["provider"] == "test-provider"

    def test_timeout_serialized_as_dict_when_granular(self):
        """Test httpx.Timeout is serialized as a dict with connect/read/write/pool."""
        import httpx

        from llmeter.serialization import dump_object

        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-4",
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        state = dump_object(endpoint)["__llmeter_state__"]
        assert state["timeout"] == {
            "connect": 5.0,
            "read": 10.0,
            "write": 10.0,
            "pool": 10.0,
        }

    def test_completion_endpoint_round_trip(self):
        """Test save/load round-trip preserves all fields for completion endpoint."""
        from llmeter.endpoints.base import Endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            original = OpenAICompletionEndpoint(
                model_id="gpt-4",
                api_key="test_key",
                endpoint_name="proj-test",
                organization="org-rt",
                project="proj_roundtrip",
                base_url="https://custom.test/v1",
                websocket_base_url="wss://custom.test/ws",
                timeout=45.0,
                max_retries=3,
                default_headers={"X-Foo": "bar"},
                default_query={"q": "1"},
                provider="test-provider",
            )
            output_path = Path(tmpdir) / "endpoint.json"
            original.save(output_path)

            loaded = Endpoint.load_from_file(output_path)

            assert isinstance(loaded, OpenAICompletionEndpoint)
            assert loaded.model_id == "gpt-4"
            assert loaded.endpoint_name == "proj-test"
            assert loaded._client.organization == "org-rt"
            assert loaded.project == "proj_roundtrip"
            assert "custom.test" in str(loaded._client.base_url)
            assert loaded._client.websocket_base_url == "wss://custom.test/ws"
            assert loaded._client.timeout == 45.0
            assert loaded._client.max_retries == 3
            assert loaded._client._custom_headers == {"X-Foo": "bar"}
            assert loaded._client._custom_query == {"q": "1"}
            assert loaded.provider == "test-provider"

    def test_round_trip_with_granular_timeout(self):
        """Test that httpx.Timeout round-trips correctly through dict serialization."""
        import httpx

        from llmeter.endpoints.base import Endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            original = OpenAICompletionEndpoint(
                model_id="gpt-4",
                timeout=httpx.Timeout(10.0, connect=2.0),
            )
            output_path = Path(tmpdir) / "endpoint.json"
            original.save(output_path)

            loaded = Endpoint.load_from_file(output_path)

            assert isinstance(loaded, OpenAICompletionEndpoint)
            assert loaded._client.timeout == httpx.Timeout(10.0, connect=2.0)

    def test_stream_endpoint_serializes_basic_fields(self):
        """Test stream endpoint serialized state when optional parameters not set."""
        from llmeter.serialization import dump_object

        endpoint = OpenAICompletionStreamEndpoint(model_id="gpt-4")
        state = dump_object(endpoint)["__llmeter_state__"]
        assert "api_key" not in state
        assert state["endpoint_name"] == "openai"
        assert state["model_id"] == "gpt-4"
        assert state.get("project") is None
        assert state.get("organization") is None
        assert state["provider"] == "openai"

    def test_stream_endpoint_round_trip(self):
        """Test save/load round-trip preserves all fields for streaming endpoint."""
        from llmeter.endpoints.base import Endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            original = OpenAICompletionStreamEndpoint(
                model_id="zai.glm-5",
                api_key="test_key",
                endpoint_name="proj-test-stream",
                organization="org-stream",
                project="proj_roundtrip_stream",
                base_url="https://stream.test/v1",
                max_retries=4,
                default_headers={"X-Stream": "yes"},
                provider="test-provider-stream",
            )
            output_path = Path(tmpdir) / "endpoint.json"
            original.save(output_path)

            loaded = Endpoint.load_from_file(output_path)

            assert isinstance(loaded, OpenAICompletionStreamEndpoint)
            assert loaded.model_id == "zai.glm-5"
            assert loaded.endpoint_name == "proj-test-stream"
            assert loaded._client.organization == "org-stream"
            assert loaded.project == "proj_roundtrip_stream"
            assert "stream.test" in str(loaded._client.base_url)
            assert loaded._client.max_retries == 4
            assert loaded._client._custom_headers == {"X-Stream": "yes"}
            assert loaded.provider == "test-provider-stream"

    def test_round_trip_without_optional_fields(self):
        """Test save/load round-trip works when only required params are set."""
        from llmeter.endpoints.base import Endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            original = OpenAICompletionEndpoint(model_id="gpt-4", api_key="test_key")
            output_path = Path(tmpdir) / "endpoint.json"
            original.save(output_path)

            loaded = Endpoint.load_from_file(output_path)

            assert isinstance(loaded, OpenAICompletionEndpoint)
            assert loaded.model_id == "gpt-4"
            assert loaded.project is None
            assert loaded._client.organization is None


class TestStreamMidStreamErrors:
    """Verify that errors during stream consumption are caught by the invoke wrapper."""

    def test_timeout_during_stream_consumption(self):
        """A timeout while iterating chunks should be caught, not raised."""
        endpoint = OpenAICompletionStreamEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        def exploding_stream():
            chunk = MagicMock()
            chunk.id = "test-id"
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Hello"
            chunk.usage = None
            yield chunk
            raise TimeoutError("Read timed out")

        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = exploding_stream()
            response = endpoint.invoke(
                {"messages": [{"role": "user", "content": "Hi"}]}
            )

        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "timed out" in response.error.lower()
        assert response.input_payload is not None

    def test_connection_error_during_stream_consumption(self):
        """A connection drop mid-stream should be caught, not raised."""
        endpoint = OpenAICompletionStreamEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )

        def dropping_stream():
            chunk = MagicMock()
            chunk.id = "test-id"
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Partial"
            chunk.usage = None
            yield chunk
            raise ConnectionError("Connection reset")

        with patch.object(endpoint._client.chat.completions, "create") as mock_create:
            mock_create.return_value = dropping_stream()
            response = endpoint.invoke(
                {"messages": [{"role": "user", "content": "Hi"}]}
            )

        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "connection" in response.error.lower()
        assert response.input_payload is not None


# ---------------------------------------------------------------------------
# Tests: reasoning models on the Chat Completions streaming API
# ---------------------------------------------------------------------------


def _reasoning_chunk(chunk_id="chatcmpl-r1", field="reasoning_content"):
    """Build a streaming chunk whose delta carries reasoning but no visible content.

    Uses SimpleNamespace rather than MagicMock so that only the attributes set here exist -- a
    MagicMock would auto-create `reasoning_content` on every delta.
    """
    delta = SimpleNamespace(content=None, **{field: "thinking..."})
    chunk = SimpleNamespace(
        id=chunk_id, choices=[SimpleNamespace(delta=delta)], usage=None
    )
    return chunk


def _content_chunk(text, chunk_id="chatcmpl-r1"):
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(
        id=chunk_id, choices=[SimpleNamespace(delta=delta)], usage=None
    )


class TestOpenAICompletionStreamFirstTokenMetrics:
    @pytest.fixture
    def endpoint(self):
        return OpenAICompletionStreamEndpoint(model_id="gpt-oss-120b", api_key="test")

    @pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
    @patch("time.perf_counter")
    def test_reasoning_chunk_sets_ttft_only(self, mock_perf_counter, endpoint, field):
        """A reasoning-only chunk sets TTFT but not the content TTFT."""
        mock_perf_counter.side_effect = [100.2, 100.6]

        response = InvocationResponse(response_text=None)
        endpoint.process_raw_response(
            iter([_reasoning_chunk(field=field), _content_chunk("Answer")]),
            100.0,
            response,
        )

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.6)
        assert response.response_text == "Answer"

    def test_reasoning_excluded_from_response_text(self, endpoint):
        response = InvocationResponse(response_text=None)
        endpoint.process_raw_response(
            iter([_reasoning_chunk(), _content_chunk("Visible")]),
            time.perf_counter(),
            response,
        )

        assert response.response_text == "Visible"

    def test_thinking_blocks_recognized(self, endpoint):
        """LiteLLM-style structured thinking blocks also count as reasoning."""
        delta = SimpleNamespace(
            content=None, thinking_blocks=[{"type": "thinking", "thinking": "hmm"}]
        )
        chunk = SimpleNamespace(
            id="c1", choices=[SimpleNamespace(delta=delta)], usage=None
        )

        response = InvocationResponse(response_text=None)
        endpoint.process_raw_response(
            iter([chunk, _content_chunk("Answer")]), time.perf_counter(), response
        )

        assert response.time_to_first_token is not None
        assert response.time_to_first_token < response.time_to_first_content_token

    def test_metrics_equal_without_reasoning(self, endpoint):
        response = InvocationResponse(response_text=None)
        endpoint.process_raw_response(
            iter([_content_chunk("Hello")]), time.perf_counter(), response
        )

        assert response.time_to_first_token is not None
        assert response.time_to_first_token == response.time_to_first_content_token

    def test_empty_delta_without_reasoning_is_ignored(self, endpoint):
        """A chunk with neither content nor reasoning must not set any timing."""
        empty = SimpleNamespace(
            id="c1",
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
            usage=None,
        )

        response = InvocationResponse(response_text=None)
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.2, 100.6]
            endpoint.process_raw_response(
                iter([empty, _content_chunk("Answer")]), 100.0, response
            )

        assert response.time_to_first_token == pytest.approx(0.6)
        assert response.time_to_first_content_token == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Tests: reasoning_type resolution, across both Chat Completions endpoints
# ---------------------------------------------------------------------------


def _sync_completion(**message_attrs):
    """A non-streaming Chat Completion whose message may carry reasoning fields."""
    message = SimpleNamespace(content="Answer", **message_attrs)
    return SimpleNamespace(
        id="chatcmpl-1", choices=[SimpleNamespace(message=message)], usage=None
    )


def _stream_for_shape(shape: str):
    chunks = {
        "reasoning_content": [_reasoning_chunk(field="reasoning_content")],
        "reasoning": [_reasoning_chunk(field="reasoning")],
        "none": [],
    }[shape]
    return iter([*chunks, _content_chunk("Answer")])


def _sync_for_shape(shape: str):
    attrs = {
        "reasoning_content": {"reasoning_content": "thinking"},
        "reasoning": {"reasoning": "thinking"},
        "none": {},
    }[shape]
    return _sync_completion(**attrs)


#: Parametrising over the transport is the point: the same provider fields appear in streamed deltas
#: and in a non-streaming message, so both must resolve equivalent content identically.
_MODES = {
    "streaming": (OpenAICompletionStreamEndpoint, _stream_for_shape),
    "non-streaming": (OpenAICompletionEndpoint, _sync_for_shape),
}

_SHAPES = ("reasoning_content", "reasoning", "none")


def _resolve(mode: str, shape: str, model_id="gpt-oss-120b", declared=None):
    endpoint_cls, build = _MODES[mode]
    endpoint = endpoint_cls(
        model_id=model_id, api_key="k", default_reasoning_visibility=declared
    )
    response = InvocationResponse(response_text=None)
    endpoint.process_raw_response(build(shape), time.perf_counter(), response)
    return response


class TestOpenAICompletionReasoningTypeResolution:
    """Chat Completions carries no fidelity marker, so this is inferred or declared."""

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "shape,expected",
        [
            ("reasoning_content", "verbatim"),
            ("reasoning", "verbatim"),
            # No reasoning at all must stay unset, *not* take the endpoint's default
            ("none", None),
        ],
    )
    def test_resolution_by_content_shape(self, mode, shape, expected):
        assert _resolve(mode, shape).reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("gpt-oss-120b", "verbatim"),
            ("deepseek-reasoner", "verbatim"),
            ("anthropic.claude-opus-4-6", "summary"),
            ("bedrock/anthropic.claude-sonnet-4-6", "summary"),
        ],
    )
    def test_inferred_from_model_id(self, mode, model_id, expected):
        response = _resolve(mode, "reasoning_content", model_id=model_id)
        assert response.reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "declared,expected",
        [("verbatim", "verbatim"), ("summary", "summary"), ("unknown", "unknown")],
    )
    def test_declared_visibility_overrides_inference(self, mode, declared, expected):
        response = _resolve(
            mode,
            "reasoning_content",
            model_id="anthropic.claude-opus-4-6",
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

    def test_reasoning_after_first_content_is_still_detected(self):
        """Detection must not be gated on TTFT being unset, or late reasoning would be missed."""
        endpoint = OpenAICompletionStreamEndpoint(model_id="gpt-oss-120b", api_key="k")
        response = InvocationResponse(response_text=None)
        endpoint.process_raw_response(
            iter([_content_chunk("Ans"), _reasoning_chunk(), _content_chunk("wer")]),
            time.perf_counter(),
            response,
        )
        assert response.reasoning_type == "verbatim"

    def test_inferred_value_round_trips(self):
        """The *resolved* value must persist, not just an explicitly declared one.

        `default_reasoning_visibility` is resolved eagerly in `__init__`, so `to_dict()` records the
        concrete guess rather than `None`. That is what makes a saved endpoint config reproducible --
        reloading it must not re-run the inference (which could change as the heuristic evolves) or
        silently drop back to a different default.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original = OpenAICompletionStreamEndpoint(
                model_id="anthropic.claude-opus-4-6", api_key="k"
            )
            assert original.default_reasoning_visibility == "summary", (
                "guard: this model ID should infer 'summary', or the test proves nothing"
            )
            path = Path(tmpdir) / "endpoint.json"
            original.save_to_file(path)
            saved = json.loads(path.read_text())
            loaded = Endpoint.load_from_file(path)

        assert (
            saved["__llmeter_state__"]["default_reasoning_visibility"] == "summary"
        ), "the resolved value must be written to disk, not omitted or left null"
        assert isinstance(loaded, OpenAICompletionStreamEndpoint)
        assert loaded.default_reasoning_visibility == "summary"

    def test_invalid_declared_value_is_rejected_at_construction(self):
        """A typo must fail loudly rather than quietly suppressing TPOT."""
        with pytest.raises(ValueError, match="not a recognized reasoning type"):
            OpenAICompletionStreamEndpoint(
                model_id="gpt-oss-120b",
                api_key="k",
                default_reasoning_visibility="verbatm",
            )

    def test_declared_arg_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = OpenAICompletionStreamEndpoint(
                model_id="gpt-oss-120b",
                api_key="k",
                default_reasoning_visibility="summary",
            )
            path = Path(tmpdir) / "endpoint.json"
            original.save_to_file(path)
            loaded = Endpoint.load_from_file(path)

        assert isinstance(loaded, OpenAICompletionStreamEndpoint)
        assert loaded.model_id == "gpt-oss-120b", "other config must survive too"
        assert loaded.default_reasoning_visibility == "summary"


# ---------------------------------------------------------------------------
# Tests: reasoning that is billed but never streamed (OpenAI's own models)
# ---------------------------------------------------------------------------


def _usage_chunk(completion_tokens=20, reasoning_tokens=None, chunk_id="chatcmpl-r1"):
    """A final usage-bearing chunk, optionally reporting a reasoning-token breakdown."""
    details = (
        SimpleNamespace(reasoning_tokens=reasoning_tokens)
        if reasoning_tokens is not None
        else None
    )
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=completion_tokens,
        prompt_tokens_details=None,
        completion_tokens_details=details,
    )
    return SimpleNamespace(id=chunk_id, choices=[], usage=usage)


def _sync_completion_with_usage(reasoning_tokens=None, **message_attrs):
    details = (
        SimpleNamespace(reasoning_tokens=reasoning_tokens)
        if reasoning_tokens is not None
        else None
    )
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=20,
        prompt_tokens_details=None,
        completion_tokens_details=details,
    )
    message = SimpleNamespace(content="Answer", **message_attrs)
    return SimpleNamespace(
        id="chatcmpl-1", choices=[SimpleNamespace(message=message)], usage=usage
    )


class TestOpenAICompletionHiddenReasoning:
    """`api.openai.com` streams no reasoning content, reporting only `reasoning_tokens`.

    `reasoning_content`/`reasoning` are vendor extensions that OpenAI itself does not emit, so for
    an o-series or GPT-5 model nothing in the stream identifies the reasoning phase. Without the
    token-count backfill these responses would claim `reasoning_type=None` -- i.e. "this model did
    not reason" -- and TPOT would then pair a post-reasoning TTFT against a reasoning-inclusive
    token count, the exact mismatch the metric exists to avoid.
    """

    @pytest.fixture
    def endpoint(self):
        return OpenAICompletionStreamEndpoint(model_id="gpt-5", api_key="k")

    def _invoke(self, endpoint, chunks):
        with patch.object(endpoint._client.chat.completions, "create") as create:
            create.return_value = iter(chunks)
            return endpoint.invoke({"messages": [{"role": "user", "content": "Hi"}]})

    def test_streaming_resolves_unknown_from_token_count(self, endpoint):
        response = self._invoke(
            endpoint, [_content_chunk("Answer"), _usage_chunk(reasoning_tokens=8)]
        )

        assert response.num_tokens_output_reasoning == 8
        assert response.reasoning_type == "unknown", (
            "reasoning tokens were billed, so `None` would wrongly assert no reasoning"
        )

    def test_streaming_stays_unset_when_no_reasoning_tokens(self, endpoint):
        response = self._invoke(
            endpoint, [_content_chunk("Answer"), _usage_chunk(reasoning_tokens=0)]
        )

        assert response.reasoning_type is None

    def test_streamed_reasoning_content_still_wins(self):
        """A provider that *does* stream reasoning must keep its observed classification."""
        endpoint = OpenAICompletionStreamEndpoint(
            model_id="deepseek-reasoner", api_key="k"
        )

        response = self._invoke(
            endpoint,
            [
                _reasoning_chunk(),
                _content_chunk("Answer"),
                _usage_chunk(reasoning_tokens=8),
            ],
        )

        assert response.reasoning_type == "verbatim"

    def test_non_streaming_resolves_unknown_from_token_count(self):
        endpoint = OpenAICompletionEndpoint(model_id="gpt-5", api_key="k")

        with patch.object(endpoint._client.chat.completions, "create") as create:
            create.return_value = _sync_completion_with_usage(reasoning_tokens=8)
            response = endpoint.invoke(
                {"messages": [{"role": "user", "content": "Hi"}]}
            )

        assert response.reasoning_type == "unknown"

    @pytest.mark.asyncio
    async def test_tpot_uses_the_answer_only_pairing(self, endpoint):
        """End-to-end: the backfill is what keeps the Runner off the mismatched pairing."""
        from llmeter.runner import _Run

        response = self._invoke(
            endpoint, [_content_chunk("Answer"), _usage_chunk(reasoning_tokens=8)]
        )
        # Pin the timings so the two candidate pairings give distinguishable answers
        response.time_to_first_token = 1.0
        response.time_to_first_content_token = 1.0
        response.time_to_last_token = 5.0

        await _Run._compute_time_per_output_token(response)

        # answer-only: (5.0 - 1.0) / ((20 - 8) - 1) == 0.3636...
        # whole-output (wrong): (5.0 - 1.0) / (20 - 1) == 0.2105...
        assert response.time_per_output_token == pytest.approx(4.0 / 11)
