# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Bedrock Converse API endpoints.

This module contains integration tests that verify BedrockConverse and
BedrockConverseStream endpoints work correctly with actual AWS Bedrock services.

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest -m integ

Required AWS Permissions:
    - bedrock:InvokeModel
    - bedrock:InvokeModelWithResponseStream

Estimated Cost:
    - ~$0.0001 per text-only test run
    - ~$0.0002 per image test run
    - ~$0.0006 total for all tests in this module
"""

import pytest

from llmeter.endpoints.bedrock import BedrockConverse


@pytest.mark.integ
def test_bedrock_converse_non_streaming(
    aws_credentials, aws_region, bedrock_test_model, test_payload
):
    """
    Test BedrockConverse endpoint with actual Bedrock service.

    This test validates that the BedrockConverse endpoint can successfully:
    - Create an endpoint instance with a valid model
    - Invoke the endpoint with a test payload
    - Receive a non-streaming response
    - Extract response text, token counts, and timing information
    - Complete without errors

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
        test_payload: Simple text test payload (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0001 per test run (using Claude Haiku 4.5 with minimal tokens)
    """
    # Create BedrockConverse endpoint instance
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    # Invoke the endpoint with test payload
    response = endpoint.invoke(test_payload)

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
def test_bedrock_converse_streaming(
    aws_credentials, aws_region, bedrock_test_model, test_payload
):
    """
    Test BedrockConverseStream endpoint with actual Bedrock service.

    This test validates that the BedrockConverseStream endpoint can successfully:
    - Create a streaming endpoint instance with a valid model
    - Invoke the endpoint with a test payload
    - Receive a streaming response
    - Extract response text, token counts, and timing information
    - Measure time to first token (TTFT) and time to last token (TTLT)
    - Verify TTLT > TTFT
    - Complete without errors

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
        test_payload: Simple text test payload (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModelWithResponseStream

    Estimated Cost:
        ~$0.0001 per test run (using Claude Haiku 4.5 with minimal tokens)
    """
    from llmeter.endpoints.bedrock import BedrockConverseStream

    # Create BedrockConverseStream endpoint instance
    endpoint = BedrockConverseStream(model_id=bedrock_test_model, region=aws_region)

    # Invoke the endpoint with test payload
    response = endpoint.invoke(test_payload)

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


@pytest.mark.integ
def test_bedrock_converse_with_image(
    aws_credentials, aws_region, bedrock_test_model, test_payload_with_image
):
    """
    Test BedrockConverse endpoint with image payload for multimodal capabilities.

    This test validates that the BedrockConverse endpoint can successfully:
    - Create an endpoint instance with a valid model
    - Invoke the endpoint with an image payload
    - Receive a non-streaming response
    - Extract response text describing the image
    - Verify token counts include image tokens
    - Measure response timing
    - Complete without errors

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5 (with multimodal content)**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
        test_payload_with_image: Test payload with image content (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0002 per test run (higher due to image processing)
    """
    # Create BedrockConverse endpoint instance
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    # Invoke the endpoint with image payload
    response = endpoint.invoke(test_payload_with_image)

    # Verify response contains non-empty text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str), "Response text should be a string"

    # Verify the response mentions the image content (should describe the red square)
    # The response should reference color or image content
    response_lower = response.response_text.lower()
    assert (
        "red" in response_lower
        or "square" in response_lower
        or "image" in response_lower
        or "color" in response_lower
    ), f"Response should describe the image content, got: {response.response_text}"

    # Verify token counts are present and positive
    assert response.num_tokens_input is not None, "Input token count should not be None"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    # Input tokens should be higher than text-only due to image tokens
    # Text-only payload typically uses ~20 tokens, image adds more
    assert (
        response.num_tokens_input > 30
    ), "Input token count should include image tokens (expected > 30)"

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
def test_bedrock_converse_streaming_with_image(
    aws_credentials, aws_region, bedrock_test_model, test_payload_with_image
):
    """
    Test BedrockConverseStream endpoint with image payload for multimodal capabilities.

    This test validates that the BedrockConverseStream endpoint can successfully:
    - Create a streaming endpoint instance with a valid model
    - Invoke the endpoint with an image payload
    - Receive a streaming response
    - Extract response text describing the image
    - Verify token counts include image tokens
    - Measure time to first token (TTFT) and time to last token (TTLT)
    - Verify TTLT > TTFT
    - Complete without errors

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6 (with multimodal content)**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
        test_payload_with_image: Test payload with image content (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModelWithResponseStream

    Estimated Cost:
        ~$0.0002 per test run (higher due to image processing)
    """
    from llmeter.endpoints.bedrock import BedrockConverseStream

    # Create BedrockConverseStream endpoint instance
    endpoint = BedrockConverseStream(model_id=bedrock_test_model, region=aws_region)

    # Invoke the endpoint with image payload
    response = endpoint.invoke(test_payload_with_image)

    # Verify response contains non-empty text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str), "Response text should be a string"

    # Verify the response mentions the image content (should describe the red square)
    # The response should reference color or image content
    response_lower = response.response_text.lower()
    assert (
        "red" in response_lower
        or "square" in response_lower
        or "image" in response_lower
        or "color" in response_lower
    ), f"Response should describe the image content, got: {response.response_text}"

    # Verify token counts are present and positive
    assert response.num_tokens_input is not None, "Input token count should not be None"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    # Input tokens should be higher than text-only due to image tokens
    # Text-only payload typically uses ~20 tokens, image adds more
    assert (
        response.num_tokens_input > 30
    ), "Input token count should include image tokens (expected > 30)"

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
