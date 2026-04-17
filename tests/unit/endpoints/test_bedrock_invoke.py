import json
import time
from contextlib import contextmanager
from io import BytesIO

import pytest
from botocore.exceptions import ClientError
from mock import MagicMock

from llmeter.endpoints.bedrock_invoke import (
    BedrockInvoke,
    BedrockInvokeStream,
    InvocationResponse,
)


@contextmanager
def _mock_invoke_model_response(
    body: dict,
    retries: int | None = 0,
    status_code: int | None = 200,
):
    with BytesIO(json.dumps(body).encode("utf-8")) as body_stream:
        yield {
            "body": body_stream,
            "ResponseMetadata": {
                "HTTPStatusCode": status_code,
                "RetryAttempts": retries,
            },
        }


class TestBedrockInvoke:
    def test__parse_payload_chat_completion(self):
        """
        ChatCompletions-like requests are parsed with default JMESPath params
        """
        endpoint = BedrockInvoke(model_id="test_model")
        msg = endpoint._parse_payload(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": "Howdy pardner"},
                    },
                ],
            }
        )
        assert msg == "Howdy pardner"

    def test__parse_payload_multi_message(self):
        """
        Multiple input message texts are joined with newlines
        """
        endpoint = BedrockInvoke(model_id="test_model")
        msg = endpoint._parse_payload(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": "Howdy"},
                    },
                    {
                        "role": "user",
                        "content": {"type": "text", "text": None},
                    },
                    {
                        "role": "user",
                        "content": {"type": "text", "text": "pardner"},
                    },
                ],
            }
        )
        assert msg == "Howdy\npardner"

    def test__parse_payload_no_content(self):
        """
        Payloads with no matching text raise an error during parsing.
        """
        endpoint = BedrockInvoke(model_id="test_model")
        with pytest.raises(TypeError, match="Failed to extract"):
            endpoint._parse_payload({})

    def test__parse_payload_custom_input_text_jmespath(self):
        """
        When text cannot be extracted from the input payload, an error should be returned
        """
        bedrock = BedrockInvoke(
            model_id="test_model", input_text_jmespath="test[].text"
        )
        with pytest.raises(TypeError, match="Failed to extract input text"):
            bedrock._parse_payload({"borf": "barf"})

        msg = bedrock._parse_payload({"test": [{"text": "Hi"}, {"text": "there"}]})
        # Arrays joined with newline:
        assert msg == "Hi\nthere"

    def test_parse_response_success(self):
        """
        Test that _parse_response correctly parses a successful ChatCompletions response
        and returns an InvocationResponse object with the expected attributes.
        """
        # Arrange
        endpoint = BedrockInvoke(model_id="test_model")
        with _mock_invoke_model_response(
            {
                "id": "test_response_id",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": "Test output",
                            "role": "assistant",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
            retries=1,
        ) as mock_response:
            # Act
            result = endpoint.parse_response(mock_response, 0.0)

        # Assert
        assert isinstance(result, InvocationResponse)
        assert result.id == "test_response_id"
        assert result.response_text == "Test output"
        assert result.num_tokens_input == 10
        assert result.num_tokens_output == 20
        assert result.retries == 1

    def test__parse_response_no_matching_jmespaths(self):
        """
        If JMESPaths are misconfigured, all outputs are empty but not an error:
        """
        endpoint = BedrockInvoke(model_id="test_model")
        with _mock_invoke_model_response({"wrong_key": "wrong_value"}) as mock_resp:
            result = endpoint.parse_response(mock_resp, 0.0)
        assert isinstance(result, InvocationResponse)
        assert result.error is None
        # id is back-filled by the invoke wrapper, not by parse_response
        assert result.input_prompt is None
        assert result.num_tokens_input is None
        assert result.num_tokens_output is None
        assert result.response_text is None

    def test__parse_response_invalid_json(self):
        """
        If response is invalid JSON, _parse_response raises JSONDecodeError.
        The base invoke wrapper converts this to an error response.
        """
        endpoint = BedrockInvoke(model_id="test_model")
        with BytesIO('{"Oops": '.encode("utf-8")) as invalid_body:
            with pytest.raises(json.JSONDecodeError):
                endpoint.parse_response({"body": invalid_body}, 0.0)

    def test__parse_response_custom_jmespaths(self):
        """
        Test that _parse_response correctly parses with custom JMESPath configs
        """
        # Arrange
        endpoint = BedrockInvoke(
            model_id="test_model",
            generated_text_jmespath="my.cool.nested.output",
            generated_token_count_jmespath="my.cool.tokensOut",
            input_token_count_jmespath="my.tokensIn",
        )
        with _mock_invoke_model_response(
            {
                "my": {
                    "cool": {
                        "nested": {"output": "Hi there"},
                        "tokensOut": 42,
                    },
                    "tokensIn": 24,
                }
            },
        ) as mock_response:
            # Act
            result = endpoint.parse_response(mock_response, 0.0)

        # Assert
        assert isinstance(result, InvocationResponse)
        assert result.response_text == "Hi there"
        assert result.num_tokens_input == 24
        assert result.num_tokens_output == 42

    def test_create_payload_message_only(self):
        """
        Test create_payload method with a single string message and default max_tokens.

        This test verifies that:
        1. A string message is correctly embedded to a message.
        2. The payload structure is correct.
        3. The default max_tokens is added to the request
        """
        user_message = "Hello, world!"
        expected_payload = {
            "messages": [
                {"role": "user", "content": [{"text": "Hello, world!", "type": "text"}]}
            ],
            "max_tokens": 256,
        }

        result = BedrockInvoke.create_payload(user_message)
        assert result == expected_payload

    def test_create_payload_overrides(self):
        """
        Test create_payload method when user_message is a list and inferenceConfig is not provided.

        This test verifies that:
        1. The method correctly handles a list of user messages.
        2. The max_tokens parameter is correctly overridden
        """
        user_message = ["Hello", "How are you?"]
        max_tokens = 300

        payload = BedrockInvoke.create_payload(user_message, max_tokens)
        expected_payload = {
            "messages": [
                {"role": "user", "content": [{"text": "Hello", "type": "text"}]},
                {"role": "user", "content": [{"text": "How are you?", "type": "text"}]},
            ],
            "max_tokens": 300,
        }
        assert payload == expected_payload

    def test_create_payload_empty_input(self):
        """
        Calling create_payload with an empty should generate an empty message.
        """
        result = BedrockInvoke.create_payload("")
        assert result == {
            "messages": [{"role": "user", "content": [{"text": "", "type": "text"}]}],
            "max_tokens": 256,
        }

    def test_create_payload_incorrect_type(self):
        """
        Calling create_payload with a non-string message should raise an error
        """
        with pytest.raises(TypeError):
            BedrockInvoke.create_payload(123)  # type: ignore

    def test_create_payload_list_with_non_string(self):
        """
        Mixed valid and invalid messages in create_payload should raise an error
        """
        with pytest.raises(TypeError):
            BedrockInvoke.create_payload(["valid", 123, "also valid"])  # type: ignore

    def test_invoke_client_error(self):
        """
        Test the invoke method when the Bedrock client raises a ClientError.
        This should be caught and returned as an error response.
        """
        endpoint = BedrockInvoke(model_id="test_model")
        endpoint._bedrock_client = MagicMock()
        endpoint._bedrock_client.invoke_model = MagicMock()
        endpoint._bedrock_client.invoke_model.side_effect = ClientError(
            {"Error": {"Message": "Test error"}}, "invoke_model"
        )
        response = endpoint.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )
        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "test error" in str(response.error).lower()

    def test_invoke_payload_no_text(self):
        """
        When text cannot be extracted from the input payload, the invocation
        should still succeed — _parse_payload failure is not an invocation error.
        """
        endpoint = BedrockInvoke(model_id="test_model")
        endpoint._bedrock_client = MagicMock()
        with _mock_invoke_model_response({}) as mock_resp:
            endpoint._bedrock_client.invoke_model = MagicMock(
                return_value=mock_resp,
            )
            response = endpoint.invoke({})
        assert isinstance(response, InvocationResponse)
        assert response.error is None
        # input_prompt falls back to None when _parse_payload can't extract text
        assert response.input_prompt is None

    def test_invoke_generic_exception(self):
        """
        Test the invoke method when a generic exception is raised.
        This should be caught and returned as an error response.
        """
        endpoint = BedrockInvoke(model_id="test_model")
        endpoint._bedrock_client = MagicMock()
        endpoint._bedrock_client.invoke_model = MagicMock(
            side_effect=Exception("Generic error")
        )
        response = endpoint.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )
        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "generic error" in str(response.error).lower()

    def test_invoke_missing_model_id(self):
        """
        Test the invoke method with a payload missing the model_id.
        This should result in an error as model_id is required.
        """
        endpoint = BedrockInvoke(model_id="")
        response = endpoint.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )
        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "invalid length for parameter modelid" in str(response.error).lower()

    def test_invoke_payload_no_llmeter_markers(self):
        """The request body sent to Bedrock InvokeModel must never contain
        ``__llmeter_bytes__`` marker objects — those are only for LLMeter's own
        persistence layer.  The API expects plain JSON (with base64 strings for
        binary data already encoded by the caller)."""
        endpoint = BedrockInvoke(model_id="test_model")
        endpoint._bedrock_client = MagicMock()

        with _mock_invoke_model_response(
            body={"output": {"message": {"content": [{"text": "Hi"}]}}},
        ) as mock_response:
            endpoint._bedrock_client.invoke_model = MagicMock(
                return_value=mock_response
            )

            payload = {
                "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
            }
            endpoint.invoke(payload)

        # Grab the body kwarg that was sent to the mock
        call_kwargs = endpoint._bedrock_client.invoke_model.call_args
        sent_body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body")
        sent_json = (
            sent_body.decode("utf-8") if isinstance(sent_body, bytes) else sent_body
        )
        assert "__llmeter_bytes__" not in sent_json


class TestBedrockInvokeStream:
    def test__parse_response_stream_standard(self):
        """
        Test the default _parse_response_stream method with ChatCompletions format events.
        """
        endpoint = BedrockInvokeStream(model_id="test_model")
        client_response = {
            "body": [
                {
                    "chunk": {
                        "bytes": json.dumps(
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "content": "Hi there ",
                                            "role": "assistant",
                                        },
                                        "finish_reason": None,
                                        "index": 0,
                                    }
                                ],
                                "id": "test_id",
                                "object": "chat.completion.chunk",
                            }
                        ).encode("utf-8")
                    }
                },
                {
                    "chunk": {
                        "bytes": json.dumps(
                            {
                                "choices": [
                                    {
                                        "delta": {"content": "fella"},
                                        "finish_reason": "stop",
                                        "index": 0,
                                    }
                                ],
                                "id": "test_id",
                                "object": "chat.completion.chunk",
                                "amazon-bedrock-invocationMetrics": {
                                    "inputTokenCount": 6,
                                    "outputTokenCount": 7,
                                    "invocationLatency": 100000,
                                    "firstByteLatency": 100000,
                                },
                            }
                        ).encode("utf-8")
                    }
                },
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        # Call the method under test
        start_time = time.perf_counter()
        result = endpoint.parse_response(client_response, start_time)

        # Assert the result is an InvocationResponse object
        assert isinstance(result, InvocationResponse)

        # Check the parsed values
        assert result.id == "test_id"
        assert result.response_text == "Hi there fella"
        assert result.num_tokens_input == 6
        assert result.num_tokens_output == 7
        assert result.retries == 0

        # Check that timing information is set
        assert result.time_to_first_token is not None
        assert result.time_to_first_token < 100000
        assert result.time_to_last_token is not None
        assert result.time_to_last_token < 100000
        # Verify that time calculations are logical
        assert 0 < result.time_to_first_token < result.time_to_last_token

    def test__parse_response_stream_no_content(self):
        """
        Test the default _parse_response_stream method when there's no content returned
        """
        endpoint = BedrockInvokeStream(model_id="test_model")
        client_response = {
            "body": [
                {
                    "chunk": {
                        "bytes": json.dumps(
                            {
                                "choices": [
                                    {
                                        "finish_reason": "stop",
                                        "index": 0,
                                    }
                                ],
                                "id": "test_id",
                                "object": "chat.completion.chunk",
                                "amazon-bedrock-invocationMetrics": {
                                    "inputTokenCount": 6,
                                    "outputTokenCount": 7,
                                    "invocationLatency": 100000,
                                    "firstByteLatency": 100000,
                                },
                            }
                        ).encode("utf-8")
                    }
                },
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        # Call the method under test
        start_time = time.perf_counter()
        result = endpoint.parse_response(client_response, start_time)

        # Assert the result is an InvocationResponse object
        assert isinstance(result, InvocationResponse)

        # Check the parsed values
        assert result.response_text == ""
        assert result.num_tokens_input == 6
        assert result.num_tokens_output == 7

        # Check that timing information is set
        assert result.time_to_first_token is None
        assert result.time_to_last_token is None

    def test__parse_response_stream_known_error(self):
        """
        Test parse_response records known Bedrock stream errors and returns partial data.
        """
        endpoint = BedrockInvokeStream(model_id="test_model")
        client_response = {
            "body": [
                {"throttlingException": {"message": "Test error"}},
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        start_time = time.perf_counter()
        result = endpoint.parse_response(client_response, start_time)

        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "test error" in result.error.lower()

    def test__parse_response_stream_known_error_with_partial_data(self):
        """
        Test that partial data collected before a stream error is preserved.
        """
        endpoint = BedrockInvokeStream(
            model_id="test_model",
            generated_text_jmespath="choices[0].delta.content",
        )
        client_response = {
            "body": [
                {
                    "chunk": {
                        "bytes": json.dumps(
                            {"choices": [{"delta": {"content": "Hello"}}]}
                        ).encode()
                    }
                },
                {
                    "chunk": {
                        "bytes": json.dumps(
                            {"choices": [{"delta": {"content": " world"}}]}
                        ).encode()
                    }
                },
                {"throttlingException": {"message": "Rate exceeded"}},
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        start_time = time.perf_counter()
        result = endpoint.parse_response(client_response, start_time)

        assert isinstance(result, InvocationResponse)
        assert result.error is not None
        assert "rate exceeded" in result.error.lower()
        # Partial data is preserved
        assert result.response_text == "Hello world"
        assert result.time_to_first_token is not None

    def test__parse_response_stream_unknown_event(self):
        """
        Test parse_response skips unknown event types without aborting.
        """
        endpoint = BedrockInvokeStream(
            model_id="test_model",
            generated_text_jmespath="choices[0].delta.content",
        )
        client_response = {
            "body": [
                {
                    "chunk": {
                        "bytes": json.dumps(
                            {"choices": [{"delta": {"content": "Hi"}}]}
                        ).encode()
                    }
                },
                {"myUnexpectedChunkType": {"hello": "world"}},
                {
                    "chunk": {
                        "bytes": json.dumps(
                            {"choices": [{"delta": {"content": "!"}}]}
                        ).encode()
                    }
                },
            ],
            "ResponseMetadata": {"RetryAttempts": 0},
        }

        start_time = time.perf_counter()
        result = endpoint.parse_response(client_response, start_time)

        assert isinstance(result, InvocationResponse)
        # Unknown events are skipped, not errors
        assert result.error is None
        assert result.response_text == "Hi!"

    def test__parse_response_stream_empty_input(self):
        """
        Test _parse_response_stream with an empty input.
        """
        endpoint = BedrockInvokeStream(model_id="test_model")
        empty_response = {"body": [], "ResponseMetadata": {"RetryAttempts": 0}}
        start_time = time.perf_counter()
        result = endpoint.parse_response(empty_response, start_time)

        assert isinstance(result, InvocationResponse)
        assert result.response_text == ""
        assert result.time_to_first_token is None
        assert result.time_to_last_token is None

    def test_invoke_client_error(self):
        """
        Test the invoke method when the Bedrock client raises a ClientError.
        This should be caught and returned as an error response.
        """
        endpoint = BedrockInvokeStream(model_id="test_model")
        endpoint._bedrock_client = MagicMock()
        endpoint._bedrock_client.invoke_model_with_response_stream = MagicMock()
        endpoint._bedrock_client.invoke_model_with_response_stream.side_effect = (
            ClientError(
                {"Error": {"Message": "Test error"}},
                "invoke_model_with_response_stream",
            )
        )
        response = endpoint.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        )
        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "test error" in str(response.error).lower()

    def test_invoke_stream_mid_stream_timeout(self):
        """Verify that a timeout during stream body iteration is caught by
        the invoke wrapper and returned as an error InvocationResponse."""
        endpoint = BedrockInvokeStream(model_id="test_model")
        endpoint._bedrock_client = MagicMock()

        def exploding_body():
            yield {
                "chunk": {
                    "bytes": json.dumps(
                        {"choices": [{"delta": {"content": "Hello"}}]}
                    ).encode()
                }
            }
            raise TimeoutError("Read timed out during streaming")

        endpoint._bedrock_client.invoke_model_with_response_stream.return_value = {
            "body": exploding_body(),
            "ResponseMetadata": {"RetryAttempts": 0, "RequestId": "req-789"},
        }

        response = endpoint.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hi"}]}]}
        )

        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "timed out" in response.error.lower()
        assert response.input_payload is not None

    def test_invoke_stream_connection_drop(self):
        """Verify that a connection error mid-stream is caught by the wrapper."""
        endpoint = BedrockInvokeStream(model_id="test_model")
        endpoint._bedrock_client = MagicMock()

        def dropping_body():
            yield {
                "chunk": {
                    "bytes": json.dumps(
                        {"choices": [{"delta": {"content": "Partial"}}]}
                    ).encode()
                }
            }
            raise ConnectionError("Connection reset by peer")

        endpoint._bedrock_client.invoke_model_with_response_stream.return_value = {
            "body": dropping_body(),
            "ResponseMetadata": {"RetryAttempts": 0, "RequestId": "req-abc"},
        }

        response = endpoint.invoke(
            {"messages": [{"role": "user", "content": [{"text": "Hi"}]}]}
        )

        assert isinstance(response, InvocationResponse)
        assert response.error is not None
        assert "connection" in response.error.lower()
        assert response.input_payload is not None
