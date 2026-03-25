# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Property-based tests for OpenAI Response API endpoints.

This module uses Hypothesis to verify universal properties across generated inputs.
Each test runs a minimum of 100 iterations to ensure correctness across a wide range
of inputs.

Feature: openai-response-api
"""

from unittest.mock import Mock, patch

from hypothesis import given, settings, strategies as st

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.openai_response import ResponseEndpoint, ResponseStreamEndpoint


# Feature: openai-response-api, Property 7: Payload Creation Format
# Validates: Requirements 8.2, 8.4, 8.5, 8.6
@settings(max_examples=100)
@given(
    user_message=st.one_of(
        st.text(min_size=1, max_size=500),
        st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=10),
    ),
    max_tokens=st.integers(min_value=1, max_value=4096),
    temperature=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0)),
    top_p=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
)
def test_property_payload_creation_format(user_message, max_tokens, temperature, top_p):
    """
    Property 7: Payload Creation Format

    **Validates: Requirements 8.2, 8.4, 8.5, 8.6**

    For any user_message (string or sequence of strings), optional instructions,
    and optional parameters, create_payload should return a dictionary containing
    an "input" field (string or message array), optional "instructions" field,
    and all provided parameters merged in.
    """
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p

    payload = ResponseEndpoint.create_payload(
        user_message=user_message, max_tokens=max_tokens, **kwargs
    )

    # Verify payload is a dictionary
    assert isinstance(payload, dict)

    # Verify input field exists and has correct format
    assert "input" in payload

    if isinstance(user_message, str):
        # String input should be preserved as-is
        assert payload["input"] == user_message
    else:
        # Sequence should be converted to message array
        assert isinstance(payload["input"], list)
        assert len(payload["input"]) == len(user_message)
        for i, msg in enumerate(payload["input"]):
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] == "user"
            assert msg["content"] == user_message[i]

    # Verify max_tokens is included
    assert "max_tokens" in payload
    assert payload["max_tokens"] == max_tokens

    # Verify optional parameters are included when provided
    if temperature is not None:
        assert "temperature" in payload
        assert payload["temperature"] == temperature

    if top_p is not None:
        assert "top_p" in payload
        assert payload["top_p"] == top_p


# Feature: openai-response-api, Property 1: Non-Streaming Response Parsing Completeness
# Validates: Requirements 2.4, 3.1, 3.2, 3.3, 3.4, 3.5
@settings(max_examples=100)
@given(
    response_id=st.text(min_size=1, max_size=50).map(lambda x: f"resp_{x}"),
    response_text=st.text(min_size=0, max_size=1000),
    prompt_tokens=st.one_of(st.none(), st.integers(min_value=1, max_value=10000)),
    completion_tokens=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
)
@patch("llmeter.endpoints.openai.OpenAI")
def test_property_non_streaming_response_parsing_completeness(
    mock_openai_class, response_id, response_text, prompt_tokens, completion_tokens
):
    """
    Property 1: Non-Streaming Response Parsing Completeness

    **Validates: Requirements 2.4, 3.1, 3.2, 3.3, 3.4, 3.5**

    For any valid Response object from the OpenAI Responses API, parsing should
    produce an InvocationResponse containing the response ID (format: resp_xxxxx),
    non-None response text extracted via output_text helper, token counts (when
    usage data is present), and a positive time_to_last_token value.
    """
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    # Create mock response
    mock_response = Mock()
    mock_response.id = response_id
    mock_response.output_text = response_text

    if prompt_tokens is not None and completion_tokens is not None:
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = prompt_tokens
        mock_response.usage.output_tokens = completion_tokens
    else:
        mock_response.usage = None

    mock_client.responses.create.return_value = mock_response

    endpoint = ResponseEndpoint(model_id="gpt-4")
    payload = {"input": "Test", "max_tokens": 256}
    result = endpoint.invoke(payload)

    # Verify response ID is present and has correct format
    assert result.id is not None
    assert result.id == response_id
    assert result.id.startswith("resp_")

    # Verify response text is present (can be empty string but not None)
    assert result.response_text is not None
    assert result.response_text == response_text

    # Verify token counts match when usage is present
    if prompt_tokens is not None and completion_tokens is not None:
        assert result.num_tokens_input == prompt_tokens
        assert result.num_tokens_output == completion_tokens
    else:
        assert result.num_tokens_input is None
        assert result.num_tokens_output is None

    # Verify time_to_last_token is positive
    assert result.time_to_last_token is not None
    assert result.time_to_last_token > 0

    # Verify no error
    assert result.error is None


# Feature: openai-response-api, Property 2: Streaming Response Assembly
# Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
@settings(max_examples=100)
@given(
    response_id=st.text(min_size=1, max_size=50).map(lambda x: f"resp_{x}"),
    text_chunks=st.lists(st.text(min_size=0, max_size=100), min_size=1, max_size=20),
    prompt_tokens=st.one_of(st.none(), st.integers(min_value=1, max_value=10000)),
    completion_tokens=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
)
@patch("llmeter.endpoints.openai.OpenAI")
def test_property_streaming_response_assembly(
    mock_openai_class, response_id, text_chunks, prompt_tokens, completion_tokens
):
    """
    Property 2: Streaming Response Assembly

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6**

    For any streaming response from the OpenAI Responses API, processing all chunks
    should produce an InvocationResponse containing the response ID from the first
    chunk (format: resp_xxxxx), concatenated response text from all output items
    with type "message" and content type "output_text", token counts from the final
    chunk (when available), positive time_to_first_token, and time_to_last_token
    greater than time_to_first_token.
    """
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    # Create mock streaming events (Response API uses typed events)
    mock_events = []

    # ResponseCreatedEvent with response ID
    event_created = Mock()
    event_created.type = "response.created"
    event_created.response = Mock()
    event_created.response.id = response_id
    mock_events.append(event_created)

    # Text delta events for each chunk
    for text in text_chunks:
        event_delta = Mock()
        event_delta.type = "response.output_text.delta"
        event_delta.delta = text
        mock_events.append(event_delta)

    # Completed event with usage if provided
    event_completed = Mock()
    event_completed.type = "response.completed"
    event_completed.response = Mock()
    if prompt_tokens is not None and completion_tokens is not None:
        event_completed.response.usage = Mock(spec=["input_tokens", "output_tokens"])
        event_completed.response.usage.input_tokens = prompt_tokens
        event_completed.response.usage.output_tokens = completion_tokens
    else:
        event_completed.response.usage = None
    mock_events.append(event_completed)

    # Mock the streaming response
    mock_client.responses.create.return_value = iter(mock_events)

    endpoint = ResponseStreamEndpoint(model_id="gpt-4")
    payload = {"input": "Test", "max_tokens": 256}
    result = endpoint.invoke(payload)

    # Verify response ID is present and has correct format
    assert result.id is not None
    assert result.id == response_id
    assert result.id.startswith("resp_")

    # Verify response text is concatenated correctly
    expected_text = "".join(text_chunks)
    assert result.response_text is not None
    assert result.response_text == expected_text

    # Verify token counts match when usage is present
    if prompt_tokens is not None and completion_tokens is not None:
        assert result.num_tokens_input == prompt_tokens
        assert result.num_tokens_output == completion_tokens
    else:
        assert result.num_tokens_input is None
        assert result.num_tokens_output is None

    # Verify time_to_first_token is positive
    assert result.time_to_first_token is not None
    assert result.time_to_first_token > 0

    # Verify time_to_last_token is positive
    assert result.time_to_last_token is not None
    assert result.time_to_last_token > 0

    # Verify time_to_last_token >= time_to_first_token
    assert result.time_to_last_token >= result.time_to_first_token

    # Verify no error
    assert result.error is None


# Feature: openai-response-api, Property 3: Payload Preservation in Responses
# Validates: Requirements 2.5, 4.3
@settings(max_examples=100)
@given(
    input_text=st.text(min_size=1, max_size=200),
    max_tokens=st.integers(min_value=1, max_value=4096),
    is_error=st.booleans(),
)
@patch("llmeter.endpoints.openai.OpenAI")
def test_property_payload_preservation(
    mock_openai_class, input_text, max_tokens, is_error
):
    """
    Property 3: Payload Preservation in Responses

    **Validates: Requirements 2.5, 4.3**

    For any invocation (successful or error), the returned InvocationResponse
    should contain the input_payload that was sent to the API.
    """
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    if is_error:
        # Simulate an error
        mock_client.responses.create.side_effect = Exception("Test error")
    else:
        # Simulate a successful response
        mock_response = Mock()
        mock_response.id = "resp_test123"
        mock_response.output_text = "Test response"
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.responses.create.return_value = mock_response

    endpoint = ResponseEndpoint(model_id="gpt-4")
    payload = {"input": input_text, "max_tokens": max_tokens}
    result = endpoint.invoke(payload)

    # Verify input_payload is preserved
    assert result.input_payload is not None
    assert isinstance(result.input_payload, dict)

    # Verify the payload contains the expected fields
    # Note: The implementation adds model_id, stream, etc. to the payload
    assert "input" in result.input_payload
    assert result.input_payload["input"] == input_text
    assert "max_tokens" in result.input_payload
    assert result.input_payload["max_tokens"] == max_tokens
    assert "model" in result.input_payload
    assert result.input_payload["model"] == "gpt-4"


# Feature: openai-response-api, Property 4: Input Prompt Extraction
# Validates: Requirements 2.6, 9.2, 9.4
@settings(max_examples=100)
@given(
    input_value=st.one_of(
        st.text(min_size=1, max_size=200),
        st.lists(
            st.fixed_dictionaries(
                {"role": st.just("user"), "content": st.text(min_size=1, max_size=100)}
            ),
            min_size=1,
            max_size=5,
        ),
    ),
)
@patch("llmeter.endpoints.openai.OpenAI")
def test_property_input_prompt_extraction(mock_openai_class, input_value):
    """
    Property 4: Input Prompt Extraction

    **Validates: Requirements 2.6, 9.2, 9.4**

    For any payload with messages, the extracted input_prompt should be the
    concatenation of all message contents, and this extracted prompt should
    appear in the InvocationResponse.
    """
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    # Mock successful response
    mock_response = Mock()
    mock_response.id = "resp_test123"
    mock_response.output_text = "Test response"
    mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_client.responses.create.return_value = mock_response

    endpoint = ResponseEndpoint(model_id="gpt-4")
    payload = {"input": input_value, "max_tokens": 256}
    result = endpoint.invoke(payload)

    # Verify input_prompt is extracted
    assert result.input_prompt is not None

    # Calculate expected prompt based on input type
    if isinstance(input_value, str):
        expected_prompt = input_value
    else:
        # For message array, concatenate contents with newlines
        expected_prompt = "\n".join(msg["content"] for msg in input_value)

    assert result.input_prompt == expected_prompt


# Feature: openai-response-api, Property 5: Error Handling Without Exceptions
# Validates: Requirements 4.1, 4.4, 4.5, 11.1, 11.5
@settings(max_examples=100)
@given(
    error_type=st.sampled_from(
        [
            "APIConnectionError",
            "AuthenticationError",
            "RateLimitError",
            "BadRequestError",
            "Exception",
        ]
    ),
    error_message=st.text(min_size=1, max_size=100),
    endpoint_type=st.sampled_from(["non_streaming", "streaming"]),
)
@patch("llmeter.endpoints.openai.OpenAI")
def test_property_error_handling_without_exceptions(
    mock_openai_class, error_type, error_message, endpoint_type
):
    """
    Property 5: Error Handling Without Exceptions

    **Validates: Requirements 4.1, 4.4, 4.5, 11.1, 11.5**

    For any invocation that encounters an error (connection error, API exception,
    invalid payload), the endpoint should return an InvocationResponse with a
    populated error field and a unique ID, without raising unhandled exceptions.
    """
    from openai import (
        APIConnectionError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
    )

    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    # Create the appropriate error type
    if error_type == "APIConnectionError":
        mock_request = Mock()
        error = APIConnectionError(message=error_message, request=mock_request)
    elif error_type == "AuthenticationError":
        error = AuthenticationError(error_message, response=Mock(), body=None)
    elif error_type == "RateLimitError":
        error = RateLimitError(error_message, response=Mock(), body=None)
    elif error_type == "BadRequestError":
        error = BadRequestError(error_message, response=Mock(), body=None)
    else:
        error = Exception(error_message)

    mock_client.responses.create.side_effect = error

    # Test with appropriate endpoint type
    if endpoint_type == "non_streaming":
        endpoint = ResponseEndpoint(model_id="gpt-4")
    else:
        endpoint = ResponseStreamEndpoint(model_id="gpt-4")

    payload = {"input": "Test", "max_tokens": 256}

    # This should NOT raise an exception
    result = endpoint.invoke(payload)

    # Verify error response structure
    assert result is not None
    assert isinstance(result, InvocationResponse)
    assert result.error is not None
    assert error_message in result.error
    assert result.id is not None
    assert len(result.id) > 0
    assert result.response_text is None
    assert result.input_payload is not None


# Feature: openai-response-api, Property 8: Parameter Forwarding
# Validates: Requirements 12.2, 12.3, 12.4, 12.5, 12.6
@settings(max_examples=100)
@given(
    temperature=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0)),
    top_p=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    frequency_penalty=st.one_of(st.none(), st.floats(min_value=-2.0, max_value=2.0)),
    presence_penalty=st.one_of(st.none(), st.floats(min_value=-2.0, max_value=2.0)),
)
@patch("llmeter.endpoints.openai.OpenAI")
def test_property_parameter_forwarding(
    mock_openai_class, temperature, top_p, frequency_penalty, presence_penalty
):
    """
    Property 8: Parameter Forwarding

    **Validates: Requirements 12.2, 12.3, 12.4, 12.5, 12.6**

    For any additional keyword arguments provided to invoke (temperature, top_p,
    text.format, etc.), they should be merged with the payload and included in
    the API request along with the model_id.
    """
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    # Mock successful response
    mock_response = Mock()
    mock_response.id = "resp_test123"
    mock_response.output_text = "Test response"
    mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_client.responses.create.return_value = mock_response

    endpoint = ResponseEndpoint(model_id="gpt-4-test")
    payload = {"input": "Test", "max_tokens": 256}

    # Build kwargs with non-None parameters
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty

    endpoint.invoke(payload, **kwargs)

    # Verify the call was made
    assert mock_client.responses.create.called

    # Get the actual payload sent to the API
    call_args = mock_client.responses.create.call_args
    actual_payload = call_args[1] if call_args[1] else call_args[0][0]

    # Verify model_id is included
    assert "model" in actual_payload
    assert actual_payload["model"] == "gpt-4-test"

    # Verify all provided parameters are forwarded
    if temperature is not None:
        assert "temperature" in actual_payload
        assert actual_payload["temperature"] == temperature

    if top_p is not None:
        assert "top_p" in actual_payload
        assert actual_payload["top_p"] == top_p

    if frequency_penalty is not None:
        assert "frequency_penalty" in actual_payload
        assert actual_payload["frequency_penalty"] == frequency_penalty

    if presence_penalty is not None:
        assert "presence_penalty" in actual_payload
        assert actual_payload["presence_penalty"] == presence_penalty

    # Verify original payload fields are preserved
    assert "input" in actual_payload
    assert actual_payload["input"] == "Test"
    assert "max_tokens" in actual_payload
    assert actual_payload["max_tokens"] == 256
