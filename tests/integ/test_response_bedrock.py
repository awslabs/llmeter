# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for OpenAI Response API with Bedrock endpoints.

This module contains integration tests that verify the OpenAI SDK can successfully
call Bedrock models via Bedrock's OpenAI-compatible endpoint (Bedrock Mantle) using
the Response API (not chat completions).

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    poetry run pytest -m integ

Required AWS Permissions:
    - bedrock:InvokeModel
    - bedrock:InvokeModelWithResponseStream

Estimated Cost:
    - ~$0.0001 per test run
    - ~$0.0002 total for all tests in this module
"""

import time

import pytest

try:
    from aws_bedrock_token_generator import provide_token
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_response_bedrock_non_streaming(
    aws_credentials, aws_region, bedrock_openai_test_model
):
    """
    Test OpenAI Response API with Bedrock non-streaming.

    This test validates that the OpenAI SDK can successfully:
    - Configure with Bedrock's Mantle endpoint for Response API
    - Authenticate using AWS temporary token
    - Make a non-streaming Response API request
    - Receive a response in OpenAI Response format
    - Extract response text using output_text helper
    - Extract token counts from usage
    - Measure response timing
    - Complete without errors

    Note: Response API uses bedrock-mantle endpoint, not bedrock-runtime.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_openai_test_model: Model ID for OpenAI SDK testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0001 per test run
    """
    # Generate temporary token for Bedrock authentication
    token = provide_token(region=aws_region)

    # Configure OpenAI client with Bedrock Mantle endpoint for Response API
    # Response API uses bedrock-mantle endpoint, not bedrock-runtime
    base_url = f"https://bedrock-mantle.{aws_region}.api.aws/v1"
    client = OpenAI(api_key=token, base_url=base_url)

    # Response API requires model ID without version suffix (e.g., openai.gpt-oss-120b)
    # Strip version suffix if present (e.g., -1:0)
    model_id = bedrock_openai_test_model.rsplit("-", 1)[0] if "-" in bedrock_openai_test_model and ":" in bedrock_openai_test_model else bedrock_openai_test_model

    # Create test payload
    test_input = [
        {
            "role": "user",
            "content": "Hello, this is a test message. Please respond with a brief greeting.",
        }
    ]

    # Measure timing and invoke
    start_time = time.perf_counter()
    response = client.responses.create(
        model=model_id,
        input=test_input,
        max_output_tokens=100,
    )
    response_time = time.perf_counter() - start_time

    # Verify response contains non-empty text using output_text helper
    assert response.output_text is not None, "Response output_text should not be None"
    assert len(response.output_text) > 0, "Response output_text should not be empty"
    assert isinstance(
        response.output_text, str
    ), "Response output_text should be a string"

    # Verify token counts are present and positive (usage may be None for some models)
    if response.usage is not None:
        assert (
            response.usage.input_tokens is not None
        ), "Input token count should not be None"
        assert response.usage.input_tokens > 0, "Input token count should be positive"
        assert (
            response.usage.output_tokens is not None
        ), "Output token count should not be None"
        assert (
            response.usage.output_tokens > 0
        ), "Output token count should be positive"

    # Verify response time is measured and positive
    assert response_time > 0, "Response time should be positive"

    # Verify response has an ID with correct format
    assert response.id is not None, "Response should have an ID"
    assert response.id.startswith("resp_"), "Response ID should start with 'resp_'"


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_response_bedrock_streaming(
    aws_credentials, aws_region, bedrock_openai_test_model
):
    """
    Test OpenAI Response API with Bedrock streaming.

    This test validates that the OpenAI SDK can successfully:
    - Configure with Bedrock's Mantle endpoint for Response API
    - Authenticate using AWS temporary token
    - Make a streaming Response API request
    - Receive response chunks in OpenAI Response format
    - Extract text from output array items
    - Assemble complete response text from chunks
    - Measure time to first token (TTFT) and time to last token (TTLT)
    - Extract token counts from the final chunk
    - Complete without errors

    Note: Response API uses bedrock-mantle endpoint, not bedrock-runtime.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_openai_test_model: Model ID for OpenAI SDK testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModelWithResponseStream

    Estimated Cost:
        ~$0.0001 per test run
    """
    # Generate temporary token for Bedrock authentication
    token = provide_token(region=aws_region)

    # Configure OpenAI client with Bedrock Mantle endpoint for Response API
    # Response API uses bedrock-mantle endpoint, not bedrock-runtime
    base_url = f"https://bedrock-mantle.{aws_region}.api.aws/v1"
    client = OpenAI(api_key=token, base_url=base_url)

    # Response API requires model ID without version suffix (e.g., openai.gpt-oss-120b)
    # Strip version suffix if present (e.g., -1:0)
    model_id = bedrock_openai_test_model.rsplit("-", 1)[0] if "-" in bedrock_openai_test_model and ":" in bedrock_openai_test_model else bedrock_openai_test_model

    # Create test payload
    test_input = [
        {
            "role": "user",
            "content": "Hello, this is a test message. Please respond with a brief greeting.",
        }
    ]

    # Measure timing and invoke with streaming
    start_time = time.perf_counter()
    stream = client.responses.create(
        model=model_id,
        input=test_input,
        max_output_tokens=100,
        stream=True,
        stream_options={"include_usage": True},
    )

    # Process streaming response
    response_text = ""
    response_id = None
    time_to_first_token = None
    prompt_tokens = None
    completion_tokens = None
    first_chunk = True

    for chunk in stream:
        if first_chunk:
            time_to_first_token = time.perf_counter() - start_time
            # Get response ID from the first chunk
            if hasattr(chunk, "response") and chunk.response:
                response_id = chunk.response.id
            first_chunk = False

        # Handle different event types
        # ResponseOutputTextDeltaEvent has delta attribute with text
        if hasattr(chunk, "delta") and chunk.delta:
            response_text += chunk.delta or ""
        
        # ResponseOutputItemAddedEvent and similar events have item attribute
        elif hasattr(chunk, "item") and chunk.item:
            if chunk.item.type == "message" and hasattr(chunk.item, "content"):
                for content_item in chunk.item.content:
                    if content_item.type == "output_text" and hasattr(content_item, "text"):
                        response_text += content_item.text or ""

        # ResponseCompletedEvent has response with full output
        elif hasattr(chunk, "response") and chunk.response:
            if hasattr(chunk.response, "usage") and chunk.response.usage:
                prompt_tokens = chunk.response.usage.input_tokens
                completion_tokens = chunk.response.usage.output_tokens

    time_to_last_token = time.perf_counter() - start_time

    # Verify response contains non-empty text
    assert response_text is not None, "Response text should not be None"
    assert len(response_text) > 0, "Response text should not be empty"
    assert isinstance(response_text, str), "Response text should be a string"

    # Verify token counts are present and positive (usage may be None for some models)
    if prompt_tokens is not None and completion_tokens is not None:
        assert prompt_tokens > 0, "Input token count should be positive"
        assert completion_tokens > 0, "Output token count should be positive"

    # Verify time to first token is measured and positive
    assert time_to_first_token is not None, "Time to first token should not be None"
    assert time_to_first_token > 0, "Time to first token should be positive"

    # Verify time to last token is measured and positive
    assert time_to_last_token is not None, "Time to last token should not be None"
    assert time_to_last_token > 0, "Time to last token should be positive"

    # Verify TTLT > TTFT (streaming should take time to complete)
    assert (
        time_to_last_token > time_to_first_token
    ), "Time to last token should be greater than time to first token"

    # Verify response has an ID with correct format
    assert response_id is not None, "Response should have an ID"
    assert response_id.startswith("resp_"), "Response ID should start with 'resp_'"
