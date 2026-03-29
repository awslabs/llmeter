# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for OpenAI Response API response format configuration.

This module tests that response format parameters (text.format) pass through
correctly and that structured outputs are parsed properly.
"""

from unittest.mock import Mock, patch


from llmeter.endpoints.openai_response import ResponseEndpoint, ResponseStreamEndpoint


class TestResponseFormatConfiguration:
    """Test response format configuration support."""

    def test_create_payload_with_text_format_json_schema(self):
        """
        Test that create_payload accepts text.format parameter for structured outputs.

        **Validates: Requirements 7.1, 7.2**
        """
        # Create payload with JSON schema format
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        }

        payload = ResponseEndpoint.create_payload(
            user_message="Generate a person with name and age",
            max_tokens=256,
            text={"format": json_schema},
        )

        # Verify text.format is in payload
        assert "text" in payload
        assert "format" in payload["text"]
        assert payload["text"]["format"] == json_schema

    def test_create_payload_with_text_format_simple(self):
        """
        Test that create_payload accepts simple text.format parameter.

        **Validates: Requirements 7.1, 7.2**
        """
        payload = ResponseEndpoint.create_payload(
            user_message="Generate JSON output",
            max_tokens=256,
            text={"format": {"type": "json_object"}},
        )

        # Verify text.format is in payload
        assert "text" in payload
        assert "format" in payload["text"]
        assert payload["text"]["format"]["type"] == "json_object"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_invoke_passes_text_format_to_api(self, mock_openai_class):
        """
        Test that invoke passes text.format parameter to the API.

        **Validates: Requirements 7.1, 7.2**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_response.output_text = '{"name": "John", "age": 30}'
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 15

        mock_client.responses.create.return_value = mock_response

        # Create endpoint and invoke with text.format
        endpoint = ResponseEndpoint(model_id="gpt-4")
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        }

        payload = {
            "input": "Generate a person",
            "max_tokens": 256,
            "text": {"format": json_schema},
        }

        response = endpoint.invoke(payload)

        # Verify the API was called with text.format parameter
        call_args = mock_client.responses.create.call_args
        assert call_args is not None
        assert "text" in call_args[1]
        assert "format" in call_args[1]["text"]
        assert call_args[1]["text"]["format"] == json_schema

        # Verify response is successful
        assert response.error is None
        assert response.response_text == '{"name": "John", "age": 30}'

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_parse_structured_output_response(self, mock_openai_class):
        """
        Test that structured output responses parse correctly using output_text helper.

        **Validates: Requirements 7.3, 7.4, 7.5**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock structured output response (JSON)
        mock_response = Mock()
        mock_response.id = "resp_456"
        mock_response.output_text = '{"name": "Alice", "age": 25}'
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 12
        mock_response.usage.output_tokens = 8

        mock_client.responses.create.return_value = mock_response

        # Create endpoint and invoke
        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {
            "input": "Generate a person",
            "max_tokens": 256,
            "text": {"format": {"type": "json_object"}},
        }

        response = endpoint.invoke(payload)

        # Verify structured output is parsed correctly
        assert response.error is None
        assert response.response_text == '{"name": "Alice", "age": 25}'
        assert response.id == "resp_456"
        assert response.num_tokens_input == 12
        assert response.num_tokens_output == 8
        assert response.time_to_last_token is not None
        assert response.time_to_last_token > 0

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_streaming_with_text_format(self, mock_openai_class):
        """
        Test that streaming endpoint passes text.format parameter correctly.

        **Validates: Requirements 7.1, 7.2**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response events (Response API uses typed events)
        # ResponseCreatedEvent with response ID
        event_created = Mock()
        event_created.type = "response.created"
        event_created.response = Mock()
        event_created.response.id = "resp_789"

        # Text delta events
        event_delta1 = Mock()
        event_delta1.type = "response.output_text.delta"
        event_delta1.delta = '{"name": "Bob"'

        event_delta2 = Mock()
        event_delta2.type = "response.output_text.delta"
        event_delta2.delta = ', "age": 35}'

        # Completed event with usage
        event_completed = Mock()
        event_completed.type = "response.completed"
        event_completed.response = Mock()
        event_completed.response.usage = Mock(spec=["input_tokens", "output_tokens"])
        event_completed.response.usage.input_tokens = 10
        event_completed.response.usage.output_tokens = 12

        mock_client.responses.create.return_value = iter(
            [event_created, event_delta1, event_delta2, event_completed]
        )

        # Create streaming endpoint and invoke with text.format
        endpoint = ResponseStreamEndpoint(model_id="gpt-4")
        json_schema = {"type": "json_object"}

        payload = {
            "input": "Generate a person",
            "max_tokens": 256,
            "text": {"format": json_schema},
        }

        response = endpoint.invoke(payload)

        # Verify the API was called with text.format parameter
        call_args = mock_client.responses.create.call_args
        assert call_args is not None
        assert "text" in call_args[1]
        assert "format" in call_args[1]["text"]
        assert call_args[1]["text"]["format"] == json_schema

        # Verify streaming response is assembled correctly
        assert response.error is None
        assert response.response_text == '{"name": "Bob", "age": 35}'
        assert response.id == "resp_789"
        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 12
        assert response.time_to_first_token is not None
        assert response.time_to_first_token > 0
        assert response.time_to_last_token is not None
        assert response.time_to_last_token > response.time_to_first_token

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_text_modality_response(self, mock_openai_class):
        """
        Test that text modality responses are handled correctly.

        **Validates: Requirements 7.4**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_text123"
        mock_response.output_text = "This is a plain text response."
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 7

        mock_client.responses.create.return_value = mock_response

        # Create endpoint and invoke without text.format (default text modality)
        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {
            "input": "Say hello",
            "max_tokens": 256,
        }

        response = endpoint.invoke(payload)

        # Verify text response is parsed correctly
        assert response.error is None
        assert response.response_text == "This is a plain text response."
        assert response.id == "resp_text123"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_json_modality_response(self, mock_openai_class):
        """
        Test that JSON modality responses are handled correctly.

        **Validates: Requirements 7.5**
        """
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock JSON response
        mock_response = Mock()
        mock_response.id = "resp_json123"
        mock_response.output_text = '{"status": "success", "data": [1, 2, 3]}'
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 8
        mock_response.usage.output_tokens = 10

        mock_client.responses.create.return_value = mock_response

        # Create endpoint and invoke with JSON format
        endpoint = ResponseEndpoint(model_id="gpt-4")
        payload = {
            "input": "Generate JSON data",
            "max_tokens": 256,
            "text": {"format": {"type": "json_object"}},
        }

        response = endpoint.invoke(payload)

        # Verify JSON response is parsed correctly
        assert response.error is None
        assert response.response_text == '{"status": "success", "data": [1, 2, 3]}'
        assert response.id == "resp_json123"
        assert response.num_tokens_input == 8
        assert response.num_tokens_output == 10
