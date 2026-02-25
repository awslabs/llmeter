# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ResponseEndpoint._parse_payload method."""

from llmeter.endpoints.openai_response import ResponseEndpoint, ResponseStreamEndpoint


class TestResponseEndpointParsePayload:
    """Test _parse_payload method for ResponseEndpoint."""

    def test_parse_payload_with_string_input(self):
        """Test parsing payload with string input (return as-is)."""
        endpoint = ResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello, how are you?"}

        result = endpoint._parse_payload(payload)

        assert result == "Hello, how are you?"

    def test_parse_payload_with_message_array(self):
        """Test parsing payload with message array (concatenate contents)."""
        endpoint = ResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {
            "input": [
                {"role": "user", "content": "First message"},
                {"role": "user", "content": "Second message"},
            ]
        }

        result = endpoint._parse_payload(payload)

        assert result == "First message\nSecond message"

    def test_parse_payload_with_empty_messages(self):
        """Test parsing payload with no messages (return empty string)."""
        endpoint = ResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": []}

        result = endpoint._parse_payload(payload)

        assert result == ""

    def test_parse_payload_with_missing_input(self):
        """Test parsing payload with missing input field (return empty string)."""
        endpoint = ResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {}

        result = endpoint._parse_payload(payload)

        assert result == ""

    def test_parse_payload_with_none_input(self):
        """Test parsing payload with None input (return empty string)."""
        endpoint = ResponseEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": None}

        result = endpoint._parse_payload(payload)

        assert result == ""


class TestResponseStreamEndpointParsePayload:
    """Test _parse_payload method for ResponseStreamEndpoint."""

    def test_parse_payload_with_string_input(self):
        """Test parsing payload with string input (return as-is)."""
        endpoint = ResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": "Hello, how are you?"}

        result = endpoint._parse_payload(payload)

        assert result == "Hello, how are you?"

    def test_parse_payload_with_message_array(self):
        """Test parsing payload with message array (concatenate contents)."""
        endpoint = ResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {
            "input": [
                {"role": "user", "content": "First message"},
                {"role": "user", "content": "Second message"},
            ]
        }

        result = endpoint._parse_payload(payload)

        assert result == "First message\nSecond message"

    def test_parse_payload_with_empty_messages(self):
        """Test parsing payload with no messages (return empty string)."""
        endpoint = ResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": []}

        result = endpoint._parse_payload(payload)

        assert result == ""

    def test_parse_payload_with_missing_input(self):
        """Test parsing payload with missing input field (return empty string)."""
        endpoint = ResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {}

        result = endpoint._parse_payload(payload)

        assert result == ""

    def test_parse_payload_with_none_input(self):
        """Test parsing payload with None input (return empty string)."""
        endpoint = ResponseStreamEndpoint(model_id="gpt-4", api_key="dummy-key")
        payload = {"input": None}

        result = endpoint._parse_payload(payload)

        assert result == ""
