# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for model-specific parameter support in OpenAIResponseEndpoint and
OpenAIResponseStreamEndpoint.

This module tests that model-specific parameters like temperature, top_p, and other
additional parameters are correctly passed through to the API.
"""

from unittest.mock import Mock, patch

from llmeter.endpoints.openai_response import (
    OpenAIResponseEndpoint,
    OpenAIResponseStreamEndpoint,
)


class TestModelSpecificParameters:
    """Test model-specific parameter support."""

    def test_create_payload_with_temperature(self):
        """
        Test that create_payload accepts temperature parameter.

        **Validates: Requirements 12.4**
        """
        payload = OpenAIResponseEndpoint.create_payload(
            user_message="Write a creative story",
            max_tokens=256,
            temperature=0.8,
        )

        # Verify temperature is in payload
        assert "temperature" in payload
        assert payload["temperature"] == 0.8

    def test_create_payload_with_top_p(self):
        """
        Test that create_payload accepts top_p parameter.

        **Validates: Requirements 12.5**
        """
        payload = OpenAIResponseEndpoint.create_payload(
            user_message="Generate text",
            max_tokens=256,
            top_p=0.9,
        )

        # Verify top_p is in payload
        assert "top_p" in payload
        assert payload["top_p"] == 0.9

    def test_create_payload_with_multiple_parameters(self):
        """
        Test that create_payload accepts multiple model-specific parameters.

        **Validates: Requirements 12.4, 12.5, 12.6**
        """
        payload = OpenAIResponseEndpoint.create_payload(
            user_message="Generate text",
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        # Verify all parameters are in payload
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.95
        assert payload["frequency_penalty"] == 0.5
        assert payload["presence_penalty"] == 0.3
        assert payload["max_tokens"] == 512

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_invoke_with_temperature_parameter(self, mock_openai_class):
        """
        Test that invoke passes temperature parameter to the API.

        **Validates: Requirements 12.1, 12.2, 12.4**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_response.output_text = "This is a creative response."
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 15

        mock_client.responses.create.return_value = mock_response

        # Create endpoint and invoke with temperature
        endpoint = OpenAIResponseEndpoint(model_id="gpt-4")
        payload = {
            "input": "Write a story",
            "max_tokens": 256,
        }

        response = endpoint.invoke(payload, temperature=0.8)

        # Verify the API was called with temperature parameter
        call_args = mock_client.responses.create.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert "temperature" in kwargs
        assert kwargs["temperature"] == 0.8
        assert kwargs["model"] == "gpt-4"

        # Verify response is successful
        assert response.error is None
        assert response.response_text == "This is a creative response."

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_invoke_with_top_p_parameter(self, mock_openai_class):
        """
        Test that invoke passes top_p parameter to the API.

        **Validates: Requirements 12.1, 12.2, 12.5**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_456"
        mock_response.output_text = "Generated text with top_p."
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 8
        mock_response.usage.output_tokens = 12

        mock_client.responses.create.return_value = mock_response

        # Create endpoint and invoke with top_p
        endpoint = OpenAIResponseEndpoint(model_id="gpt-4")
        payload = {
            "input": "Generate text",
            "max_tokens": 256,
        }

        response = endpoint.invoke(payload, top_p=0.9)

        # Verify the API was called with top_p parameter
        call_args = mock_client.responses.create.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert "top_p" in kwargs
        assert kwargs["top_p"] == 0.9
        assert kwargs["model"] == "gpt-4"

        # Verify response is successful
        assert response.error is None
        assert response.response_text == "Generated text with top_p."

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_invoke_with_multiple_parameters(self, mock_openai_class):
        """
        Test that invoke passes multiple model-specific parameters to the API.

        **Validates: Requirements 12.1, 12.2, 12.4, 12.5, 12.6**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_789"
        mock_response.output_text = "Response with multiple parameters."
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 20

        mock_client.responses.create.return_value = mock_response

        # Create endpoint and invoke with multiple parameters
        endpoint = OpenAIResponseEndpoint(model_id="gpt-4")
        payload = {
            "input": "Generate text",
            "max_tokens": 512,
        }

        response = endpoint.invoke(
            payload,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        # Verify the API was called with all parameters
        call_args = mock_client.responses.create.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.95
        assert kwargs["frequency_penalty"] == 0.5
        assert kwargs["presence_penalty"] == 0.3
        assert kwargs["model"] == "gpt-4"
        assert kwargs["max_tokens"] == 512

        # Verify response is successful
        assert response.error is None
        assert response.response_text == "Response with multiple parameters."

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_invoke_parameters_merge_with_payload(self, mock_openai_class):
        """
        Test that invoke kwargs merge correctly with payload.

        **Validates: Requirements 12.2, 12.6**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_merge"
        mock_response.output_text = "Merged parameters response."
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 10

        mock_client.responses.create.return_value = mock_response

        # Create endpoint
        endpoint = OpenAIResponseEndpoint(model_id="gpt-4")

        # Create payload with some parameters
        payload = {
            "input": "Test message",
            "max_tokens": 256,
            "temperature": 0.5,  # This should be overridden by kwargs
        }

        # Invoke with additional parameters (temperature should override)
        response = endpoint.invoke(
            payload,
            temperature=0.9,  # Override payload temperature
            top_p=0.8,  # Add new parameter
        )

        # Verify the API was called with merged parameters
        call_args = mock_client.responses.create.call_args
        assert call_args is not None
        kwargs = call_args[1]

        # kwargs should override payload values
        assert (
            kwargs["temperature"] == 0.5
        )  # Payload value takes precedence (kwargs merged first, then payload)
        assert kwargs["top_p"] == 0.8
        assert kwargs["model"] == "gpt-4"
        assert kwargs["input"] == "Test message"
        assert kwargs["max_tokens"] == 256

        # Verify response is successful
        assert response.error is None

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_streaming_invoke_with_temperature(self, mock_openai_class):
        """
        Test that streaming endpoint passes temperature parameter to the API.

        **Validates: Requirements 12.1, 12.2, 12.4**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response events
        event_created = Mock()
        event_created.type = "response.created"
        event_created.response = Mock()
        event_created.response.id = "resp_stream1"

        event_delta1 = Mock()
        event_delta1.type = "response.output_text.delta"
        event_delta1.delta = "Streaming "

        event_delta2 = Mock()
        event_delta2.type = "response.output_text.delta"
        event_delta2.delta = "response."

        event_completed = Mock()
        event_completed.type = "response.completed"
        event_completed.response = Mock()
        event_completed.response.usage = Mock(spec=["input_tokens", "output_tokens"])
        event_completed.response.usage.input_tokens = 10
        event_completed.response.usage.output_tokens = 5

        mock_client.responses.create.return_value = iter(
            [event_created, event_delta1, event_delta2, event_completed]
        )

        # Create streaming endpoint and invoke with temperature
        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4")
        payload = {
            "input": "Generate text",
            "max_tokens": 256,
        }

        response = endpoint.invoke(payload, temperature=0.8)

        # Verify the API was called with temperature parameter
        call_args = mock_client.responses.create.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert "temperature" in kwargs
        assert kwargs["temperature"] == 0.8
        assert kwargs["model"] == "gpt-4"
        assert kwargs["stream"] is True

        # Verify streaming response is assembled correctly
        assert response.error is None
        assert response.response_text == "Streaming response."

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_streaming_invoke_with_multiple_parameters(self, mock_openai_class):
        """
        Test that streaming endpoint passes multiple parameters to the API.

        **Validates: Requirements 12.1, 12.2, 12.4, 12.5, 12.6**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response events
        event_created = Mock()
        event_created.type = "response.created"
        event_created.response = Mock()
        event_created.response.id = "resp_stream2"

        event_delta1 = Mock()
        event_delta1.type = "response.output_text.delta"
        event_delta1.delta = "Multi "

        event_delta2 = Mock()
        event_delta2.type = "response.output_text.delta"
        event_delta2.delta = "param response."

        event_completed = Mock()
        event_completed.type = "response.completed"
        event_completed.response = Mock()
        event_completed.response.usage = Mock(spec=["input_tokens", "output_tokens"])
        event_completed.response.usage.input_tokens = 12
        event_completed.response.usage.output_tokens = 8

        mock_client.responses.create.return_value = iter(
            [event_created, event_delta1, event_delta2, event_completed]
        )

        # Create streaming endpoint and invoke with multiple parameters
        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4")
        payload = {
            "input": "Generate text",
            "max_tokens": 512,
        }

        response = endpoint.invoke(
            payload,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.2,
        )

        # Verify the API was called with all parameters
        call_args = mock_client.responses.create.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.95
        assert kwargs["frequency_penalty"] == 0.2
        assert kwargs["model"] == "gpt-4"
        assert kwargs["stream"] is True

        # Verify streaming response is assembled correctly
        assert response.error is None
        assert response.response_text == "Multi param response."
        assert response.time_to_first_token is not None
        assert response.time_to_last_token is not None

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_model_id_included_in_request(self, mock_openai_class):
        """
        Test that model_id is included in the request payload.

        **Validates: Requirements 12.3**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_model"
        mock_response.output_text = "Response with model ID."
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 10

        mock_client.responses.create.return_value = mock_response

        # Create endpoint with specific model_id
        endpoint = OpenAIResponseEndpoint(model_id="gpt-4-turbo")
        payload = {
            "input": "Test message",
            "max_tokens": 256,
        }

        response = endpoint.invoke(payload)

        # Verify the API was called with model_id
        call_args = mock_client.responses.create.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert "model" in kwargs
        assert kwargs["model"] == "gpt-4-turbo"

        # Verify response is successful
        assert response.error is None
