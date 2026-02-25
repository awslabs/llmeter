# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError
from mock import MagicMock

from llmeter.endpoints.bedrock import (
    BedrockBase,
    BedrockConverse,
    BedrockConverseStream,
    InvocationResponse,
)


class TestBedrock:
    def test__parse_conversation_stream_1(self):
        """
        Test the _parse_conversation_stream method with a complete stream containing
        contentBlockDelta, contentBlockStop, and metadata.

        This test verifies that the method correctly parses the conversation stream,
        extracts the output text, calculates timing information, and processes metadata
        to create an accurate InvocationResponse object.
        """
        # Mock the UUID generation to ensure consistent test results
        uuid.uuid4 = Mock(return_value=Mock(hex="test_uuid"))

        # Create a mock client response
        client_response = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                {"contentBlockDelta": {"delta": {"text": " world!"}}},
                {"contentBlockStop": {}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 20}}},
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        # Create an instance of BedrockConverseStream
        bedrock_stream = BedrockConverseStream(model_id="test_model")

        # Call the method under test
        start_time = time.perf_counter()
        result = bedrock_stream._parse_conversation_stream(client_response, start_time)

        # Assert the result is an InvocationResponse object
        assert isinstance(result, InvocationResponse)

        # Check the parsed values
        # assert result.id == "test_uuid"
        assert result.response_text == "Hello world!"
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 20
        assert result.retries == 0

        # Check that timing information is set
        assert result.time_to_first_token is not None
        assert result.time_to_last_token is not None
        assert result.time_per_output_token is not None
        assert result.time_per_output_token < 0.01
        # Verify that time calculations are logical
        assert 0 < result.time_to_first_token < result.time_to_last_token

    def test__parse_conversation_stream_2(self):
        """
        Test the _parse_conversation_stream method when:
        - "contentBlockDelta" is in chunk, but time_flag is False
        - "contentBlockStop" is in chunk
        - "metadata" is in chunk
        - metadata is present
        - num_tokens_output, time_to_last_token, and time_to_first_token are all truthy

        This test verifies that the method correctly processes the stream,
        calculates timings, and sets all relevant fields in the InvocationResponse.
        """
        mock_client_response = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "Hello "}}},
                {"contentBlockDelta": {"delta": {"text": "world!"}}},
                {"contentBlockStop": {}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 20}}},
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }
        start_t = time.perf_counter()

        with patch("time.perf_counter") as mock_perf_counter:
            mock_perf_counter.side_effect = [start_t + 0.1, start_t + 0.2]

            bedrock_stream = BedrockConverseStream(model_id="test_model")
            result = bedrock_stream._parse_conversation_stream(
                mock_client_response, start_t
            )

        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hello world!"
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 20
        assert abs(result.time_to_first_token - 0.1) < 1e-5
        assert abs(result.time_to_last_token - 0.2) < 1e-5

    def test__parse_conversation_stream_3(self):
        """
        Test the _parse_conversation_stream method when:
        - There's no 'contentBlockDelta' in any chunk
        - There's a 'contentBlockStop' in a chunk
        - There's 'metadata' in a chunk
        - The metadata contains valid usage information
        - All necessary timing information is available

        This test verifies that the method correctly processes the stream,
        calculates timing information, and sets all relevant fields in the
        InvocationResponse object when these conditions are met.
        """
        # Setup
        bedrock_stream = BedrockConverseStream(model_id="test_model")
        start_t = time.perf_counter()

        # Mock client response
        client_response = {
            "stream": [
                {"contentBlockStop": {}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 20}}},
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        # Call the method
        result = bedrock_stream._parse_conversation_stream(client_response, start_t)

        # Assertions
        assert isinstance(result, InvocationResponse)
        assert result.id
        assert result.response_text == ""
        assert result.time_to_last_token is not None
        assert result.time_to_first_token is None
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 20
        assert result.time_per_output_token is None
        assert result.retries == 0

    def test__parse_conversation_stream_4(self):
        """
        Test the _parse_conversation_stream method when the response contains
        contentBlockDelta, metadata, and has valid token output and timing information.

        This test verifies that the method correctly processes a stream response
        containing a content block delta, metadata.
        """
        # Create an instance of BedrockConverseStream
        bedrock_stream = BedrockConverseStream(model_id="test_model")

        # Mock client response
        client_response = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                {"contentBlockStop": {}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        # Set start time
        start_t = time.perf_counter()

        # Call the method
        response = bedrock_stream._parse_conversation_stream(client_response, start_t)

        # Assertions
        assert isinstance(response, InvocationResponse)
        assert response.response_text == "Hello"
        assert response.num_tokens_input == 10
        assert response.num_tokens_output == 5
        assert response.time_to_first_token is not None
        assert response.time_to_last_token is not None
        assert response.time_per_output_token is not None
        assert response.retries == 0

    def test__parse_conversation_stream_6(self):
        """
        Test the _parse_conversation_stream method when:
        - The stream contains a contentBlockDelta
        - The time_flag is initially True
        - The stream contains a contentBlockStop
        - The stream contains metadata
        - The metadata is present
        - The conditions for calculating time_per_output_token are not met

        Expected behavior:
        - The method should return an InvocationResponse object
        - The response should contain the correct output text
        - The response should have time_to_first_token and time_to_last_token set
        - The response should have num_tokens_input and num_tokens_output set from metadata
        - The response should not have time_per_output_token set
        """
        # Create a mock BedrockConverseStream instance
        bedrock_stream = BedrockConverseStream(model_id="test_model")

        # Create a mock client response
        mock_response = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                {"contentBlockStop": {}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        # Set up the start time
        start_t = time.perf_counter()

        # Call the method under test
        result = bedrock_stream._parse_conversation_stream(mock_response, start_t)

        # Assertions
        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hello"
        assert result.time_to_first_token is not None
        assert result.time_to_last_token is not None
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 5
        assert result.time_per_output_token is not None
        assert result.retries == 0

    def test__parse_conversation_stream_7(self):
        """
        Test the _parse_conversation_stream method when the response contains
        contentBlockDelta, contentBlockStop, and metadata, but metadata is empty.
        This test verifies that the method correctly processes the stream chunks
        and returns an InvocationResponse with expected values.
        """
        # Create an instance of BedrockConverseStream
        bedrock_stream = BedrockConverseStream(model_id="test_model")

        # Mock client response
        client_response = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                {"contentBlockStop": {}},
                {"metadata": {}},
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        # Set start time
        start_t = time.perf_counter()

        # Call the method under test
        response = bedrock_stream._parse_conversation_stream(client_response, start_t)

        # Assert the response is an InvocationResponse
        assert isinstance(response, InvocationResponse)

        # Assert the response contains expected values
        assert response.id is not None
        assert response.response_text == "Hello"
        assert response.time_to_last_token is not None
        assert response.time_to_first_token is not None
        assert response.num_tokens_input is None
        assert response.num_tokens_output is None
        assert response.time_per_output_token is None
        assert response.retries == 0

    def test__parse_conversation_stream_empty_input(self):
        """
        Test _parse_conversation_stream with an empty input.
        """
        bedrock = BedrockConverseStream(model_id="test_model")
        empty_response = {"stream": [], "ResponseMetadata": {"RetryAttempts": 0}}
        start_t = 0.0

        result = bedrock._parse_conversation_stream(empty_response, start_t)

        assert isinstance(result, InvocationResponse)
        assert result.response_text == ""
        assert result.time_to_first_token is None
        assert result.time_to_last_token is None

    def test__parse_conversation_stream_incorrect_type(self):
        """
        Test _parse_conversation_stream with incorrect input type.
        """
        bedrock = BedrockConverseStream(model_id="test_model")
        incorrect_type_response = "Not a dictionary"
        start_t = 0.0

        result = bedrock._parse_conversation_stream(incorrect_type_response, start_t)

        assert isinstance(result, InvocationResponse)
        assert result.response_text is None
        assert result.time_to_first_token is None
        assert result.time_to_last_token is None

    def test__parse_conversation_stream_invalid_input(self):
        """
        Test _parse_conversation_stream with invalid input structure.
        """
        bedrock = BedrockConverseStream(model_id="test_model")
        invalid_response = {
            "invalid_key": "invalid_value",
            "ResponseMetadata": {"RetryAttempts": 0},
        }
        start_t = 0.0

        result = bedrock._parse_conversation_stream(invalid_response, start_t)

        assert isinstance(result, InvocationResponse)
        assert result.response_text is None
        assert result.time_to_first_token is None
        assert result.time_to_last_token is None

    def test__parse_conversation_stream_missing_metadata(self):
        """
        Test _parse_conversation_stream with missing metadata.
        """
        bedrock = BedrockConverseStream(model_id="test_model")
        response_without_metadata = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                {"contentBlockStop": {}},
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }
        start_t = 0.0

        result = bedrock._parse_conversation_stream(response_without_metadata, start_t)

        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hello"
        assert result.num_tokens_input is None
        assert result.num_tokens_output is None

    def test__parse_converse_response_invalid_output_structure(self):
        """
        Test _parse_converse_response with an invalid 'output' structure.
        This should return an InvocationResponse with error details.
        """
        bedrock_converse = BedrockConverse(model_id="test_model")
        invalid_response = {
            "output": {"invalid_key": "value"},
            "ResponseMetadata": {"RetryAttempts": 0},
        }
        result = bedrock_converse._parse_converse_response(invalid_response)
        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "missing required field" in str(result.error).lower()

    def test__parse_converse_response_invalid_usage_type(self):
        """
        Test _parse_converse_response with 'usage' as a non-dictionary type.
        This should not raise an exception but return None for token counts.
        """
        bedrock_converse = BedrockConverse(model_id="test_model")
        response = {
            "output": {"message": {"content": [{"text": "Test output"}]}},
            "usage": "invalid_usage",
            "ResponseMetadata": {"RetryAttempts": 0},
        }
        result = bedrock_converse._parse_converse_response(response)
        assert isinstance(result, InvocationResponse)
        assert result.num_tokens_input is None
        assert result.num_tokens_output is None

    def test__parse_converse_response_missing_output(self):
        """
        Test _parse_converse_response with a dictionary missing the 'output' key.
        This should return an InvocationResponse with an error.
        """
        bedrock_converse = BedrockConverse(model_id="test_model")
        invalid_response = {"ResponseMetadata": {"RetryAttempts": 0}}
        result = bedrock_converse._parse_converse_response(invalid_response)
        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "output" in str(result.error).lower()

    def test__parse_converse_response_missing_response_metadata(self):
        """
        Test _parse_converse_response with a dictionary missing the 'ResponseMetadata' key.
        This should return an InvocationResponse with an error.
        """
        bedrock_converse = BedrockConverse(model_id="test_model")
        invalid_response = {
            "output": {"message": {"content": [{"text": "Test output"}]}}
        }
        result = bedrock_converse._parse_converse_response(invalid_response)
        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "responsemetadata" in str(result.error).lower()

    def test__parse_payload_1(self):
        """
        Test that _parse_payload correctly extracts and concatenates text content from the payload.

        This test verifies that the _parse_payload method can handle a payload with multiple messages,
        each containing multiple content items with text. It ensures that the method correctly extracts
        all text content, filters out any None values, and joins the resulting texts with newline characters.
        """
        bedrock_converse = BedrockConverse(model_id="test_model")
        payload = {
            "messages": [
                {"content": [{"text": "Hello"}, {"text": "World"}]},
                {"content": [{"text": "Test"}, {"text": None}]},
                {"content": [{"text": ["Multiple", "Texts"]}]},
            ]
        }
        result = bedrock_converse._parse_payload(payload)
        expected = "Hello\nWorld\nTest\nMultiple\nTexts"
        assert result == expected

    def test__parse_payload_empty_input(self):
        """
        Test _parse_payload with an empty dictionary input.
        """
        bedrock_base = BedrockConverse(model_id="test_model")
        result = bedrock_base._parse_payload({})
        assert result == "", "Expected empty string for empty input"

    def test__parse_payload_empty_messages(self):
        """
        Test _parse_payload with empty messages list.
        """
        bedrock_base = BedrockConverse(model_id="test_model")
        result = bedrock_base._parse_payload({"messages": []})
        assert result == "", "Expected empty string for empty messages"

    def test__parse_payload_incorrect_format(self):
        """
        Test _parse_payload with incorrect format (messages is not a list).
        """
        bedrock_base = BedrockConverse(model_id="test_model")
        result = bedrock_base._parse_payload({"messages": "not a list"})
        assert result == "", "Expected empty string for incorrect format"

    def test__parse_payload_incorrect_type(self):
        """
        Test _parse_payload with incorrect input type (list instead of dict).
        """
        bedrock_base = BedrockConverse(model_id="test_model")
        result = bedrock_base._parse_payload([])
        assert result == "", "Expected empty string for incorrect input type"

    def test__parse_payload_invalid_input(self):
        """
        Test _parse_payload with invalid input (no 'messages' key).
        """
        bedrock_base = BedrockConverse(model_id="test_model")
        result = bedrock_base._parse_payload({"invalid_key": "value"})
        assert result == "", "Expected empty string for invalid input"

    def test__parse_payload_missing_content(self):
        """
        Test _parse_payload with messages missing 'content' key.
        """
        bedrock_base = BedrockConverse(model_id="test_model")
        payload = {"messages": [{"role": "user"}]}
        result = bedrock_base._parse_payload(payload)
        assert result == "", "Expected empty string for missing content"

    def test__parse_payload_multiple_empty_messages(self):
        """
        Test _parse_payload with multiple empty messages.
        """
        bedrock_base = BedrockConverse(model_id="test_model")
        payload = {"messages": [{}, {}, {}]}
        result = bedrock_base._parse_payload(payload)
        assert result == "", "Expected empty string for multiple empty messages"

    def test__parse_payload_nested_empty_content(self):
        """
        Test _parse_payload with nested empty content.
        """
        bedrock_base = BedrockConverse(model_id="test_model")
        payload = {"messages": [{"content": [{}]}]}
        result = bedrock_base._parse_payload(payload)
        assert result == "", "Expected empty string for nested empty content"

    def test__parse_payload_non_list_text(self):
        """
        Test _parse_payload with non-list 'text' in content.
        """
        bedrock_base = BedrockConverse(model_id="test_model")
        payload = {"messages": [{"content": [{"text": "single text"}]}]}
        result = bedrock_base._parse_payload(payload)
        assert result == "single text", "Expected 'single text' for non-list text"

    def test_create_payload_1(self):
        """
        Test create_payload method with a single string message and default max_tokens.

        This test verifies that:
        1. A string message is correctly converted to a list.
        2. The payload structure is correct.
        3. The inferenceConfig is added with the default max_tokens.
        """
        user_message = "Hello, world!"
        expected_payload = {
            "messages": [{"role": "user", "content": [{"text": "Hello, world!"}]}],
            "inferenceConfig": {"maxTokens": 256},
        }

        result = BedrockBase.create_payload(user_message)

        assert result == expected_payload

    def test_create_payload_2(self):
        """
        Test create_payload method when user_message is a list and inferenceConfig is not provided.

        This test verifies that:
        1. The method correctly handles a list of user messages.
        2. The inferenceConfig is properly initialized when not provided.
        3. The max_tokens parameter is correctly set in the inferenceConfig.
        """
        user_message = ["Hello", "How are you?"]
        max_tokens = 300

        payload = BedrockBase.create_payload(user_message, max_tokens)

        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["content"][0]["text"] == "Hello"
        assert payload["messages"][1]["content"][0]["text"] == "How are you?"
        assert "inferenceConfig" in payload
        assert payload["inferenceConfig"]["maxTokens"] == max_tokens

    def test_create_payload_3(self):
        """
        Test create_payload method when user_message is a string and inferenceConfig is provided.

        This test verifies that:
        1. The method correctly handles a single string message.
        2. The provided inferenceConfig is preserved in the payload.
        3. The max_tokens value is correctly set in the inferenceConfig.
        """
        user_message = "Hello, world!"
        max_tokens = 100
        inference_config = {"temperature": 0.7}

        payload = BedrockBase.create_payload(
            user_message, max_tokens, inferenceConfig=inference_config
        )

        assert isinstance(payload, dict)
        assert payload["messages"] == [
            {"role": "user", "content": [{"text": "Hello, world!"}]}
        ]
        assert payload["inferenceConfig"]["maxTokens"] == max_tokens
        assert payload["inferenceConfig"]["temperature"] == 0.7

    def test_create_payload_empty_input(self):
        """
        Test create_payload with an empty string input.
        This should result in a payload with an empty message.
        """
        result = BedrockBase.create_payload("")
        assert result == {
            "messages": [{"role": "user", "content": [{"text": ""}]}],
            "inferenceConfig": {"maxTokens": 256},
        }

    def test_create_payload_incorrect_type(self):
        """
        Test create_payload with an incorrect type for user_message (neither str nor list).
        This should raise a TypeError.
        """
        with pytest.raises(TypeError):
            BedrockBase.create_payload(123)

    def test_create_payload_invalid_max_tokens(self):
        """
        Test create_payload with an invalid max_tokens value (negative).
        This should raise a ValueError.
        """
        with pytest.raises(ValueError):
            BedrockBase.create_payload("test", max_tokens=-1)

    def test_create_payload_list_with_non_string(self):
        """
        Test create_payload with a list containing non-string elements.
        This should raise a TypeError when trying to create the payload.
        """
        with pytest.raises(TypeError):
            BedrockBase.create_payload(["valid", 123, "also valid"])

    def test_create_payload_very_large_max_tokens(self):
        """
        Test create_payload with a very large max_tokens value.
        This tests an edge case where the value might exceed normal limits.
        """
        result = BedrockBase.create_payload("test", max_tokens=1000000)
        assert result["inferenceConfig"]["maxTokens"] == 1000000

    def test_invoke_client_error(self):
        """
        Test the invoke method when the Bedrock client raises a ClientError.
        This should be caught and returned as an error response.
        """
        bedrock = BedrockConverse(model_id="test_model")
        bedrock._bedrock_client.converse = lambda **kwargs: (_ for _ in ()).throw(
            ClientError({"Error": {"Message": "Test error"}}, "converse")
        )
        response = bedrock.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )
        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "test error" in str(response.error).lower()

    def test_invoke_client_exception(self):
        """
        Test the invoke method when the client raises an exception.
        """
        bedrock_stream = BedrockConverseStream(model_id="test_model")
        bedrock_stream._bedrock_client = Mock()
        bedrock_stream._bedrock_client.converse_stream.side_effect = Exception(
            "Unexpected error"
        )

        response = bedrock_stream.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )

        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "Unexpected error" in str(response.error)

    def test_invoke_empty_payload(self):
        """
        Test the invoke method with an empty payload.
        This should result in an error as the payload is required.
        """
        bedrock = BedrockConverse(model_id="test_model")
        bedrock._bedrock_client = MagicMock()
        response = bedrock.invoke({})
        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "expected string for output text" in str(response.error).lower()

    # def test_invoke_empty_payload_2(self): #TODO: fix mocking of boto3 client
    #     """
    #     Test the invoke method with an empty payload.
    #     """
    #     bedrock_stream = BedrockConverseStream(model_id="test_model")
    #     bedrock_stream._bedrock_client = MagicMock()

    #     response = bedrock_stream.invoke({})

    #     assert isinstance(response, InvocationResponse)
    #     assert response.error is not None
    #     assert "expected string for output text" in str(response.error)

    def test_invoke_empty_stream_response(self):
        """
        Test the invoke method when the stream response is empty.
        """
        bedrock_stream = BedrockConverseStream(model_id="test_model")
        bedrock_stream._bedrock_client = Mock()
        bedrock_stream._bedrock_client.converse_stream.return_value = {"stream": []}

        response = bedrock_stream.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )

        assert isinstance(response, InvocationResponse)
        assert response.response_text is None
        assert response.time_to_first_token is None
        assert response.time_to_last_token is None

    def test_invoke_error_when_inference_config_is_none(self):
        """
        Test that invoke method returns an error response when inferenceConfig is None
        and an exception occurs during the API call.
        """
        # Arrange
        bedrock_converse = BedrockConverse(model_id="test_model")
        bedrock_converse._bedrock_client = Mock()
        bedrock_converse._bedrock_client.converse.side_effect = Exception("API error")

        payload = {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}

        # Act
        with patch(
            "llmeter.endpoints.bedrock.uuid4", return_value=Mock(hex="mock_uuid")
        ):
            result = bedrock_converse.invoke(payload)

        # Assert
        assert isinstance(result, InvocationResponse)
        assert result.id == "mock_uuid"
        assert result.error == "API error"
        assert result.input_payload == {
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
            "modelId": "test_model",
            "inferenceConfig": {},
        }

    def test_invoke_generic_exception(self):
        """
        Test the invoke method when a generic exception is raised.
        This should be caught and returned as an error response.
        """
        bedrock = BedrockConverse(model_id="test_model")
        bedrock._bedrock_client.converse = lambda **kwargs: (_ for _ in ()).throw(
            Exception("Generic error")
        )
        response = bedrock.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )
        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "generic error" in str(response.error).lower()

    def test_invoke_incorrect_payload_format(self):
        """
        Test the invoke method with an incorrect payload format.
        """
        bedrock_stream = BedrockConverseStream(model_id="test_model")
        bedrock_stream._bedrock_client = Mock()

        response = bedrock_stream.invoke({"wrong_key": "wrong_value"})

        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        # assert "payload" in str(response.error)

    def test_invoke_invalid_model_id(self):
        """
        Test the invoke method with an invalid model ID.
        """
        bedrock_stream = BedrockConverseStream(model_id="invalid_model")
        bedrock_stream._bedrock_client = Mock()
        bedrock_stream._bedrock_client.converse_stream.side_effect = ClientError(
            {"Error": {"Code": "ModelNotFound", "Message": "Model not found"}},
            "converse_stream",
        )

        response = bedrock_stream.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )

        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "Model not found" in str(response.error)

    def test_invoke_invalid_payload_format(self):
        """
        Test the invoke method with an invalid payload format.
        This should result in an error due to incorrect payload structure.
        """
        bedrock = BedrockConverse(model_id="test_model")
        response = bedrock.invoke({"invalid_key": "invalid_value"})
        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "unknown parameter" in str(response.error).lower()

    def test_invoke_missing_model_id(self):
        """
        Test the invoke method with a payload missing the model_id.
        This should result in an error as model_id is required.
        """
        bedrock = BedrockConverse(model_id="")
        response = bedrock.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )
        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "invalid length for parameter modelid" in str(response.error).lower()

    def test_invoke_when_inference_config_is_none_and_client_error_occurs(self):
        """
        Test the invoke method of BedrockConverseStream when inferenceConfig is None
        and a ClientError occurs during the API call.

        This test verifies that:
        1. The method correctly handles the case when inferenceConfig is None in the payload.
        2. It properly handles a ClientError raised by the Bedrock client.
        3. It returns an error InvocationResponse with the correct error information.
        """
        # Mock the Bedrock client and its converse_stream method
        mock_bedrock_client = Mock()
        mock_bedrock_client.converse_stream.side_effect = Exception("API Error")

        # Create an instance of BedrockConverseStream with the mocked client
        bedrock_stream = BedrockConverseStream(model_id="test_model")
        bedrock_stream._bedrock_client = mock_bedrock_client
        bedrock_stream._inference_config = None

        # Prepare test payload
        payload = {"messages": [{"role": "user", "content": "Hello"}]}

        # Call the method under test
        with patch("uuid.uuid4", return_value=Mock(hex="mock_uuid")):
            response = bedrock_stream.invoke(payload)

        # Assert the response is an error InvocationResponse
        assert isinstance(response, InvocationResponse)
        assert response.error == "API Error"
        # assert response.id == "mock_uuid"
        assert response.input_payload == {
            "messages": [{"role": "user", "content": "Hello"}],
            "inferenceConfig": {},
            "modelId": "test_model",
        }

        # Verify that the Bedrock client's converse_stream method was called with the correct payload
        mock_bedrock_client.converse_stream.assert_called_once_with(
            messages=[{"role": "user", "content": "Hello"}],
            inferenceConfig={},
            modelId="test_model",
        )

    def test_parse_converse_response_success(self):
        """
        Test that _parse_converse_response correctly parses a successful response
        and returns an InvocationResponse object with the expected attributes.
        """
        # Arrange
        bedrock_converse = BedrockConverse(model_id="test_model")
        mock_response = {
            "output": {"message": {"content": [{"text": "Test output"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20},
            "ResponseMetadata": {"RetryAttempts": 1},
        }

        # Act
        with patch("uuid.uuid4", return_value=Mock(hex="mock_uuid")):
            result = bedrock_converse._parse_converse_response(mock_response)

        # Assert
        assert isinstance(result, InvocationResponse)
        # assert result.id == "mock_uuid"  #TODO: fix uuid patching
        assert result.response_text == "Test output"
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 20
        assert result.retries == 1
