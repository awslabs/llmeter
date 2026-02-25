# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ResponseEndpoint class.

This module tests endpoint initialization, payload creation, response parsing,
error handling, and timing measurements for the ResponseEndpoint class.
"""

from unittest.mock import Mock, patch

from llmeter.endpoints.openai_response import ResponseEndpoint


class TestResponseEndpointInitialization:
    """Test endpoint initialization.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    """

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_init_with_model_id(self, mock_openai_class):
        """Test endpoint initialization with model_id parameter."""
        endpoint = ResponseEndpoint(model_id="gpt-4")

        assert endpoint.model_id == "gpt-4"
        assert endpoint.endpoint_name == "openai-response"
        assert endpoint.provider == "openai"
        mock_openai_class.assert_called_once()

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_init_with_api_key(self, mock_openai_class):
        """Test endpoint initialization with optional api_key parameter."""
        endpoint = ResponseEndpoint(model_id="gpt-4", api_key="test-key-123")

        assert endpoint.model_id == "gpt-4"
        # Verify OpenAI client was initialized with api_key
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key-123"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_init_with_custom_endpoint_name(self, mock_openai_class):
        """Test endpoint initialization with custom endpoint_name."""
        endpoint = ResponseEndpoint(
            model_id="gpt-4", endpoint_name="custom-response-endpoint"
        )

        assert endpoint.endpoint_name == "custom-response-endpoint"
        assert endpoint.model_id == "gpt-4"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_init_default_provider_value(self, mock_openai_class):
        """Test that provider defaults to 'openai'."""
        endpoint = ResponseEndpoint(model_id="gpt-4")

        assert endpoint.provider == "openai"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_init_with_custom_provider(self, mock_openai_class):
        """Test endpoint initialization with custom provider."""
        endpoint = ResponseEndpoint(model_id="gpt-4", provider="custom-provider")

        assert endpoint.provider == "custom-provider"


class TestResponseEndpointPayloadCreation:
    """Test payload creation helper method.

    **Validates: Requirements 8.2, 8.3, 8.4, 8.5, 8.6**
    """

    def test_create_payload_with_single_string(self):
        """Test create_payload with single string message."""
        payload = ResponseEndpoint.create_payload(
            user_message="Hello, how are you?", max_tokens=256
        )

        assert isinstance(payload, dict)
        assert payload["input"] == "Hello, how are you?"
        assert payload["max_tokens"] == 256

    def test_create_payload_with_sequence_of_strings(self):
        """Test create_payload with sequence of messages."""
        messages = ["First message", "Second message", "Third message"]
        payload = ResponseEndpoint.create_payload(user_message=messages, max_tokens=512)

        assert isinstance(payload, dict)
        assert "input" in payload
        assert isinstance(payload["input"], list)
        assert len(payload["input"]) == 3

        # Verify message format
        for i, msg in enumerate(payload["input"]):
            assert msg["role"] == "user"
            assert msg["content"] == messages[i]

    def test_create_payload_with_max_tokens(self):
        """Test create_payload with max_tokens parameter."""
        payload = ResponseEndpoint.create_payload(
            user_message="Test message", max_tokens=1024
        )

        assert payload["max_tokens"] == 1024

    def test_create_payload_default_max_tokens(self):
        """Test that max_tokens defaults to 256."""
        payload = ResponseEndpoint.create_payload(user_message="Test")

        assert payload["max_tokens"] == 256

    def test_create_payload_with_instructions(self):
        """Test create_payload with instructions parameter."""
        payload = ResponseEndpoint.create_payload(
            user_message="Write a story",
            max_tokens=256,
            instructions="You are a creative storyteller",
        )

        assert "instructions" in payload
        assert payload["instructions"] == "You are a creative storyteller"

    def test_create_payload_without_instructions(self):
        """Test create_payload without instructions parameter."""
        payload = ResponseEndpoint.create_payload(
            user_message="Test message", max_tokens=256
        )

        assert "instructions" not in payload

    def test_create_payload_with_temperature_kwarg(self):
        """Test create_payload with temperature in kwargs."""
        payload = ResponseEndpoint.create_payload(
            user_message="Test", max_tokens=256, temperature=0.8
        )

        assert payload["temperature"] == 0.8

    def test_create_payload_with_top_p_kwarg(self):
        """Test create_payload with top_p in kwargs."""
        payload = ResponseEndpoint.create_payload(
            user_message="Test", max_tokens=256, top_p=0.9
        )

        assert payload["top_p"] == 0.9

    def test_create_payload_with_multiple_kwargs(self):
        """Test create_payload with multiple additional kwargs."""
        payload = ResponseEndpoint.create_payload(
            user_message="Test",
            max_tokens=256,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.95
        assert payload["frequency_penalty"] == 0.5
        assert payload["presence_penalty"] == 0.3


class TestResponseEndpointResponseParsing:
    """Test response parsing with mocked responses.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    """

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_parse_complete_response(self, mock_openai_class):
        """Test parsing complete response with all fields."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock complete response
        mock_response = Mock()
        mock_response.id = "resp_abc123"
        mock_response.output_text = "I'm doing well, thank you for asking!"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=15)

        mock_client.responses.create.return_value = mock_response

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Hello, how are you?", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify all fields are extracted correctly
        assert response.id == "resp_abc123"
        assert response.response_text == "I'm doing well, thank you for asking!"
        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 15
        assert response.time_to_last_token is not None
        assert response.time_to_last_token > 0
        assert response.error is None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_parse_response_missing_usage(self, mock_openai_class):
        """Test parsing response with missing usage data."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response without usage
        mock_response = Mock()
        mock_response.id = "resp_xyz789"
        mock_response.output_text = "Response without usage data"
        mock_response.usage = None

        mock_client.responses.create.return_value = mock_response

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify response is parsed correctly with None token counts
        assert response.id == "resp_xyz789"
        assert response.response_text == "Response without usage data"
        assert response.num_tokens_input is None
        assert response.num_tokens_output is None
        assert response.time_to_last_token is not None
        assert response.error is None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_parse_response_empty_content(self, mock_openai_class):
        """Test parsing response with empty content."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response with empty text
        mock_response = Mock()
        mock_response.id = "resp_empty123"
        mock_response.output_text = ""
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=0)

        mock_client.responses.create.return_value = mock_response

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify empty content is handled correctly
        assert response.id == "resp_empty123"
        assert response.response_text == ""
        assert response.num_tokens_input == 5
        assert response.num_tokens_output == 0
        assert response.error is None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_extract_response_id(self, mock_openai_class):
        """Test extracting response ID."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_test_id_12345"
        mock_response.output_text = "Test response"
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=5)

        mock_client.responses.create.return_value = mock_response

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify response ID is extracted correctly
        assert response.id == "resp_test_id_12345"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_extract_token_counts(self, mock_openai_class):
        """Test extracting token counts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_tokens"
        mock_response.output_text = "Token count test"
        mock_response.usage = Mock(prompt_tokens=25, completion_tokens=50)

        mock_client.responses.create.return_value = mock_response

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify token counts are extracted correctly
        assert response.num_tokens_input == 25
        assert response.num_tokens_output == 50


class TestResponseEndpointErrorHandling:
    """Test error handling with mocked errors.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_api_connection_error(self, mock_openai_class):
        """Test handling of APIConnectionError."""
        from openai import APIConnectionError

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_request = Mock()
        mock_client.responses.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock_request
        )

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify error response
        assert response.error is not None
        assert "Connection failed" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_authentication_error(self, mock_openai_class):
        """Test handling of AuthenticationError."""
        from openai import AuthenticationError

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_client.responses.create.side_effect = AuthenticationError(
            "Invalid API key", response=Mock(), body=None
        )

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify error response
        assert response.error is not None
        assert "Invalid API key" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_rate_limit_error(self, mock_openai_class):
        """Test handling of RateLimitError."""
        from openai import RateLimitError

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_client.responses.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=Mock(), body=None
        )

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify error response
        assert response.error is not None
        assert "Rate limit exceeded" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_bad_request_error(self, mock_openai_class):
        """Test handling of BadRequestError."""
        from openai import BadRequestError

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_client.responses.create.side_effect = BadRequestError(
            "Invalid request", response=Mock(), body=None
        )

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify error response
        assert response.error is not None
        assert "Invalid request" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_generic_exception(self, mock_openai_class):
        """Test handling of generic Exception."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_client.responses.create.side_effect = Exception("Unexpected error")

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify error response
        assert response.error is not None
        assert "Unexpected error" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None


class TestResponseEndpointTimingMeasurements:
    """Test timing measurements.

    **Validates: Requirements 2.3, 3.4**
    """

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_time_to_last_token_positive(self, mock_openai_class):
        """Test that time_to_last_token is positive."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_timing"
        mock_response.output_text = "Timing test response"
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=5)

        mock_client.responses.create.return_value = mock_response

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify time_to_last_token is positive
        assert response.time_to_last_token is not None
        assert response.time_to_last_token > 0

    @patch("llmeter.endpoints.openai.OpenAI")
    @patch("time.perf_counter")
    def test_timing_accuracy(self, mock_perf_counter, mock_openai_class):
        """Test timing calculation accuracy."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock time.perf_counter to return predictable values
        mock_perf_counter.side_effect = [1.0, 1.5]  # start_t=1.0, end_t=1.5

        mock_response = Mock()
        mock_response.id = "resp_timing_accuracy"
        mock_response.output_text = "Timing accuracy test"
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=5)

        mock_client.responses.create.return_value = mock_response

        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify timing calculation (1.5 - 1.0 = 0.5)
        assert response.time_to_last_token == 0.5
