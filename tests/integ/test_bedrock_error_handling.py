# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for error handling in Bedrock endpoints.

This module contains integration tests that verify Bedrock endpoints handle
errors gracefully, including invalid model IDs and invalid payloads.

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest -m integ

Required AWS Permissions:
    - bedrock:InvokeModel

Estimated Cost:
    - ~$0.0001 per test run (most error tests fail fast with minimal cost)
    - ~$0.0002 total for all tests in this module
"""

import pytest

from llmeter.endpoints.bedrock import BedrockConverse


@pytest.mark.integ
def test_invalid_model_error(aws_credentials, aws_region):
    """
    Test that invalid model IDs produce appropriate errors.

    This test validates that endpoints handle invalid model IDs gracefully by:
    - Creating an endpoint with an invalid model ID
    - Attempting to invoke the endpoint
    - Verifying the error field is populated in the response
    - Verifying no unhandled exceptions are raised
    - Verifying the error message is descriptive

    **Validates: Requirements 8.1, 8.3, 8.4**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0001 per test run (error occurs before model invocation, minimal cost)
    """
    # Create endpoint with invalid model ID
    invalid_model_id = "invalid-model-id-that-does-not-exist"
    endpoint = BedrockConverse(model_id=invalid_model_id, region=aws_region)

    # Create a simple test payload
    test_payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": "Hello, this is a test message."}],
            }
        ],
        "inferenceConfig": {"maxTokens": 100},
    }

    # Invoke the endpoint - should not raise unhandled exception
    response = endpoint.invoke(test_payload)

    # Verify error field is populated
    assert (
        response.error is not None
    ), "Error field should be populated for invalid model"
    assert isinstance(response.error, str), "Error should be a string"
    assert len(response.error) > 0, "Error message should not be empty"

    # Verify error message is descriptive (should mention model or validation)
    error_lower = response.error.lower()
    assert (
        "model" in error_lower
        or "not found" in error_lower
        or "invalid" in error_lower
        or "validation" in error_lower
        or "access" in error_lower
    ), f"Error message should be descriptive, got: {response.error}"

    # Verify response text is None or empty when error occurs
    assert (
        response.response_text is None or len(response.response_text) == 0
    ), "Response text should be None or empty when error occurs"

    # Verify token counts are None or zero when error occurs
    # (endpoints may set these to None or 0 on error)
    if response.num_tokens_input is not None:
        assert response.num_tokens_input == 0, "Input tokens should be 0 on error"
    if response.num_tokens_output is not None:
        assert response.num_tokens_output == 0, "Output tokens should be 0 on error"


@pytest.mark.integ
def test_invalid_payload_error(aws_credentials, aws_region, bedrock_test_model):
    """
    Test that invalid payloads produce appropriate errors.

    This test validates that endpoints handle invalid payloads gracefully by:
    - Creating an endpoint with a valid model ID
    - Attempting to invoke the endpoint with an invalid payload
    - Verifying the error field is populated in the response
    - Verifying no unhandled exceptions are raised
    - Verifying the error message is descriptive and contains error details

    **Validates: Requirements 8.2, 8.3, 8.4**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Valid model ID for testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0001 per test run (error occurs during validation, minimal cost)
    """
    # Create endpoint with valid model ID
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    # Create an invalid payload - missing required fields
    # The Bedrock Converse API requires 'messages' field with proper structure
    invalid_payload = {
        "invalid_field": "this should not be here",
        "maxTokens": 100,  # Wrong location, should be in inferenceConfig
    }

    # Invoke the endpoint - should not raise unhandled exception
    response = endpoint.invoke(invalid_payload)

    # Verify error field is populated
    assert (
        response.error is not None
    ), "Error field should be populated for invalid payload"
    assert isinstance(response.error, str), "Error should be a string"
    assert len(response.error) > 0, "Error message should not be empty"

    # Verify error message is descriptive (should mention validation, required field, or messages)
    error_lower = response.error.lower()
    assert (
        "validation" in error_lower
        or "required" in error_lower
        or "messages" in error_lower
        or "invalid" in error_lower
        or "missing" in error_lower
        or "parameter" in error_lower
    ), f"Error message should be descriptive about the payload issue, got: {response.error}"

    # Verify response text is None or empty when error occurs
    assert (
        response.response_text is None or len(response.response_text) == 0
    ), "Response text should be None or empty when error occurs"

    # Verify token counts are None or zero when error occurs
    # (endpoints may set these to None or 0 on error)
    if response.num_tokens_input is not None:
        assert response.num_tokens_input == 0, "Input tokens should be 0 on error"
    if response.num_tokens_output is not None:
        assert response.num_tokens_output == 0, "Output tokens should be 0 on error"


@pytest.mark.integ
def test_error_response_structure(aws_credentials, aws_region, bedrock_test_model):
    """
    Test that error responses have consistent structure across different error types.

    This test validates that all error responses follow a consistent structure by:
    - Testing multiple error scenarios (invalid model, invalid payload)
    - Verifying each error response has the same structure
    - Verifying error field is always populated with a string
    - Verifying response_text is None or empty on errors
    - Verifying token counts are None or zero on errors
    - Verifying no unhandled exceptions are raised

    This ensures that error handling is consistent and predictable across all
    error types, making it easier for users to handle errors in their code.

    **Validates: Requirements 8.3, 8.4, 10.1, 10.2**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Valid model ID for testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0002 per test run (tests multiple error scenarios, minimal cost)
    """
    # Test scenario 1: Invalid model ID
    invalid_model_endpoint = BedrockConverse(
        model_id="invalid-model-id-that-does-not-exist", region=aws_region
    )
    test_payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": "Hello, this is a test message."}],
            }
        ],
        "inferenceConfig": {"maxTokens": 100},
    }
    response1 = invalid_model_endpoint.invoke(test_payload)

    # Test scenario 2: Invalid payload with valid model
    valid_model_endpoint = BedrockConverse(
        model_id=bedrock_test_model, region=aws_region
    )
    invalid_payload = {
        "invalid_field": "this should not be here",
        "maxTokens": 100,
    }
    response2 = valid_model_endpoint.invoke(invalid_payload)

    # Verify both responses have consistent error structure
    error_responses = [response1, response2]

    for i, response in enumerate(error_responses, 1):
        # Verify error field is populated with a non-empty string
        assert (
            response.error is not None
        ), f"Error response {i}: error field should be populated"
        assert isinstance(
            response.error, str
        ), f"Error response {i}: error should be a string"
        assert (
            len(response.error) > 0
        ), f"Error response {i}: error message should not be empty"

        # Verify response_text is None or empty
        assert (
            response.response_text is None or len(response.response_text) == 0
        ), f"Error response {i}: response_text should be None or empty on error"

        # Verify token counts are None or zero
        if response.num_tokens_input is not None:
            assert (
                response.num_tokens_input == 0
            ), f"Error response {i}: input tokens should be 0 on error"
        if response.num_tokens_output is not None:
            assert (
                response.num_tokens_output == 0
            ), f"Error response {i}: output tokens should be 0 on error"

        # Verify timing fields exist (may be 0 or None, but should be present)
        assert hasattr(
            response, "time_to_last_token"
        ), f"Error response {i}: should have time_to_last_token field"
        assert hasattr(
            response, "time_to_first_token"
        ), f"Error response {i}: should have time_to_first_token field"

    # Verify that different error types produce different error messages
    # (they should not be identical, as they represent different issues)
    assert (
        response1.error != response2.error
    ), "Different error types should produce different error messages"
