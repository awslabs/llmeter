# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Bedrock Invoke API endpoints.

This module contains integration tests that verify BedrockInvoke and
BedrockInvokeStream endpoints work correctly with actual AWS Bedrock services.

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest -m integ

Required AWS Permissions:
    - bedrock:InvokeModel
    - bedrock:InvokeModelWithResponseStream

Estimated Cost:
    - ~$0.0001 per test run
    - ~$0.0002 total for all tests in this module
"""

import pytest

from llmeter.endpoints.bedrock_invoke import BedrockInvoke


@pytest.mark.integ
def test_bedrock_invoke_non_streaming(aws_credentials, aws_region, bedrock_test_model):
    """
    Test BedrockInvoke endpoint with actual Bedrock service.

    This test validates that the BedrockInvoke endpoint can successfully:
    - Create an endpoint instance with a valid model and JMESPath expressions
    - Invoke the endpoint with a test payload
    - Receive a non-streaming response
    - Extract response text via JMESPath
    - Extract token counts via JMESPath when available
    - Measure response timing
    - Complete without errors

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0001 per test run (using Claude 3.5 Sonnet v2 with minimal tokens)
    """
    # Create BedrockInvoke endpoint instance with JMESPath expressions for Claude models
    # Claude models via InvokeModel API use the native Claude Messages API format
    endpoint = BedrockInvoke(
        model_id=bedrock_test_model,
        region=aws_region,
        # JMESPath for Claude native Messages API format
        generated_text_jmespath="content[0].text",
        generated_token_count_jmespath="usage.output_tokens",
        input_token_count_jmespath="usage.input_tokens",
        input_text_jmespath="messages[0].content",
    )

    # Create payload in Claude native Messages API format
    # This is different from the Converse API format
    invoke_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello, this is a test message. Please respond with a brief greeting.",
            }
        ],
    }

    # Invoke the endpoint with test payload
    response = endpoint.invoke(invoke_payload)

    # Verify response contains non-empty text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str), "Response text should be a string"

    # Verify token counts are present and positive
    assert response.num_tokens_input is not None, "Input token count should not be None"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert (
        response.num_tokens_output is not None
    ), "Output token count should not be None"
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify response time is measured and positive
    assert response.time_to_last_token is not None, "Response time should not be None"
    assert response.time_to_last_token > 0, "Response time should be positive"

    # Verify no errors in response
    assert (
        response.error is None
    ), f"Response should not contain errors: {response.error}"

    # Verify response has an ID
    assert response.id is not None, "Response should have an ID"


@pytest.mark.integ
def test_bedrock_invoke_streaming(aws_credentials, aws_region, bedrock_test_model):
    """
    Test BedrockInvokeStream endpoint with actual Bedrock service.

    This test validates that the BedrockInvokeStream endpoint can successfully:
    - Create a streaming endpoint instance with a valid model and JMESPath expressions
    - Invoke the endpoint with a test payload
    - Receive a streaming response
    - Extract response text via JMESPath from stream chunks
    - Extract token counts via JMESPath when available
    - Measure time to first token (TTFT) and time to last token (TTLT)
    - Verify TTLT > TTFT
    - Complete without errors

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModelWithResponseStream

    Estimated Cost:
        ~$0.0001 per test run (using Claude 3.5 Sonnet v2 with minimal tokens)
    """
    from llmeter.endpoints.bedrock_invoke import BedrockInvokeStream

    # Create BedrockInvokeStream endpoint instance with JMESPath expressions for Claude models
    # Claude models via InvokeModelWithResponseStream API use the native Claude Messages API format
    endpoint = BedrockInvokeStream(
        model_id=bedrock_test_model,
        region=aws_region,
        # JMESPath for Claude native Messages API streaming format
        # delta.text extracts text from content_block_delta chunks
        generated_text_jmespath="delta.text",
        # message.usage.output_tokens extracts output tokens from message_delta chunks
        generated_token_count_jmespath="message.usage.output_tokens",
        # message.usage.input_tokens extracts input tokens from message_start chunk
        input_token_count_jmespath="message.usage.input_tokens",
        input_text_jmespath="messages[0].content",
    )

    # Create payload in Claude native Messages API format
    # This is different from the Converse API format
    invoke_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello, this is a test message. Please respond with a brief greeting.",
            }
        ],
    }

    # Invoke the endpoint with test payload
    response = endpoint.invoke(invoke_payload)

    # Verify response contains non-empty text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str), "Response text should be a string"

    # Verify token counts are present and positive
    assert response.num_tokens_input is not None, "Input token count should not be None"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert (
        response.num_tokens_output is not None
    ), "Output token count should not be None"
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify time to first token is measured and positive
    assert (
        response.time_to_first_token is not None
    ), "Time to first token should not be None"
    assert response.time_to_first_token > 0, "Time to first token should be positive"

    # Verify time to last token is measured and positive
    assert (
        response.time_to_last_token is not None
    ), "Time to last token should not be None"
    assert response.time_to_last_token > 0, "Time to last token should be positive"

    # Verify TTLT > TTFT (streaming should take time to complete)
    assert (
        response.time_to_last_token > response.time_to_first_token
    ), "Time to last token should be greater than time to first token"

    # Verify no errors in response
    assert (
        response.error is None
    ), f"Response should not contain errors: {response.error}"

    # Verify response has an ID
    assert response.id is not None, "Response should have an ID"
