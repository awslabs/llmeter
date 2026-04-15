# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for error handling in OpenAIResponse{Stream}Endpoint classes."""

from unittest.mock import Mock, patch

from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from llmeter.endpoints.openai_response import OpenAIResponseEndpoint, OpenAIResponseStreamEndpoint


class TestOpenAIResponseEndpointErrorHandling:
    """Test error handling for OpenAIResponseEndpoint."""

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_api_connection_error(self, mock_openai_class):
        """Test handling of APIConnectionError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Create a mock request for APIConnectionError
        mock_request = Mock()
        mock_client.responses.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock_request
        )

        endpoint = OpenAIResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Connection failed" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_authentication_error(self, mock_openai_class):
        """Test handling of AuthenticationError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.responses.create.side_effect = AuthenticationError(
            "Invalid API key", response=Mock(), body=None
        )

        endpoint = OpenAIResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Invalid API key" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_rate_limit_error(self, mock_openai_class):
        """Test handling of RateLimitError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.responses.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=Mock(), body=None
        )

        endpoint = OpenAIResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Rate limit exceeded" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_bad_request_error(self, mock_openai_class):
        """Test handling of BadRequestError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.responses.create.side_effect = BadRequestError(
            "Invalid request", response=Mock(), body=None
        )

        endpoint = OpenAIResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Invalid request" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_generic_exception(self, mock_openai_class):
        """Test handling of generic Exception."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.responses.create.side_effect = Exception("Unexpected error")

        endpoint = OpenAIResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Unexpected error" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None


class TestResponseStreamEndpointErrorHandling:
    """Test error handling for ResponseStreamEndpoint."""

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_api_connection_error(self, mock_openai_class):
        """Test handling of APIConnectionError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Create a mock request for APIConnectionError
        mock_request = Mock()
        mock_client.responses.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock_request
        )

        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Connection failed" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_authentication_error(self, mock_openai_class):
        """Test handling of AuthenticationError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.responses.create.side_effect = AuthenticationError(
            "Invalid API key", response=Mock(), body=None
        )

        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Invalid API key" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_rate_limit_error(self, mock_openai_class):
        """Test handling of RateLimitError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.responses.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=Mock(), body=None
        )

        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Rate limit exceeded" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_bad_request_error(self, mock_openai_class):
        """Test handling of BadRequestError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.responses.create.side_effect = BadRequestError(
            "Invalid request", response=Mock(), body=None
        )

        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Invalid request" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_generic_exception(self, mock_openai_class):
        """Test handling of generic Exception."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.responses.create.side_effect = Exception("Unexpected error")

        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello"}
        response = endpoint.invoke(payload)

        assert response.error is not None
        assert "Unexpected error" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None
