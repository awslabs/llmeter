# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Anthropic Messages API endpoints via Amazon Bedrock Mantle.

This module verifies that the llmeter AnthropicMessages and AnthropicMessagesStream
wrappers work correctly with the Anthropic Messages API served through the
Bedrock Mantle endpoint (``bedrock-mantle.{region}.api.aws/anthropic/v1/messages``).

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest tests/integ/test_anthropic_messages_bedrock.py -m integ

Required AWS Permissions:
    - bedrock-mantle:CreateInference (or equivalent Bedrock permissions)

Environment Variables:
    - BEDROCK_ANTHROPIC_MANTLE_REGION: AWS region (default: us-east-1)
    - BEDROCK_ANTHROPIC_MANTLE_TEST_MODEL: Model ID
      (default: anthropic.claude-opus-4-7)

Estimated Cost:
    - ~$0.001 per test run (Opus 4.7 pricing)
    - ~$0.003 total for all tests in this module
"""

from datetime import datetime

import pytest

from llmeter.endpoints.anthropic_messages import (
    AnthropicMessages,
    AnthropicMessagesEndpoint,
    AnthropicMessagesStream,
)


@pytest.mark.integ
def test_anthropic_messages_non_streaming(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
):
    """
    Test AnthropicMessages endpoint with Bedrock Mantle (non-streaming).

    Validates that the endpoint can:
    - Initialize with Bedrock Mantle provider and AWS credentials
    - Invoke the Anthropic Messages API via create_payload helper
    - Return an InvocationResponse with text, token counts, and timing
    - Complete without errors

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        bedrock_anthropic_mantle_region: AWS region for Bedrock Mantle.
        bedrock_anthropic_mantle_test_model: Anthropic model ID for Bedrock Mantle.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = AnthropicMessages(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )

    payload = AnthropicMessagesEndpoint.create_payload(
        user_message="Hello, this is a test. Please respond with a brief greeting.",
        max_tokens=100,
    )

    response = endpoint.invoke(payload)

    # No errors
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str)

    # Token counts
    assert response.num_tokens_input is not None, "Input token count should be present"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert response.num_tokens_output is not None, (
        "Output token count should be present"
    )
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # Timing (back-filled by base class for non-streaming)
    assert response.time_to_last_token is not None, "Response time should be present"
    assert response.time_to_last_token > 0, "Response time should be positive"

    # Response ID (Anthropic msg_ format)
    assert response.id is not None, "Response should have an ID"
    assert response.id.startswith("msg_"), (
        f"Response ID should be Anthropic format (msg_...), got: {response.id}"
    )

    # Metadata back-fill
    assert isinstance(response.request_time, datetime)
    assert response.input_payload is not None


@pytest.mark.integ
def test_anthropic_messages_streaming(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
):
    """
    Test AnthropicMessagesStream endpoint with Bedrock Mantle (streaming).

    Validates that the endpoint can:
    - Initialize with Bedrock Mantle provider and AWS credentials
    - Invoke the streaming Anthropic Messages API
    - Return an InvocationResponse with text, TTFT, TTLT, and token counts
    - Verify TTLT > TTFT
    - Complete without errors

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        bedrock_anthropic_mantle_region: AWS region for Bedrock Mantle.
        bedrock_anthropic_mantle_test_model: Anthropic model ID for Bedrock Mantle.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = AnthropicMessagesStream(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )

    payload = AnthropicMessagesEndpoint.create_payload(
        user_message="Hello, this is a test. Please respond with a brief greeting.",
        max_tokens=100,
    )

    response = endpoint.invoke(payload)

    # No errors
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str)

    # Token counts
    assert response.num_tokens_input is not None, "Input token count should be present"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert response.num_tokens_output is not None, (
        "Output token count should be present"
    )
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # TTFT
    assert response.time_to_first_token is not None, "TTFT should be present"
    assert response.time_to_first_token > 0, "TTFT should be positive"

    # TTLT
    assert response.time_to_last_token is not None, "TTLT should be present"
    assert response.time_to_last_token > 0, "TTLT should be positive"

    # TTLT > TTFT
    assert response.time_to_last_token > response.time_to_first_token, (
        "TTLT should be greater than TTFT"
    )

    # Response ID
    assert response.id is not None, "Response should have an ID"
    assert response.id.startswith("msg_"), (
        f"Response ID should be Anthropic format (msg_...), got: {response.id}"
    )

    # Metadata back-fill
    assert isinstance(response.request_time, datetime)


@pytest.mark.integ
def test_anthropic_messages_create_payload_roundtrip(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
):
    """
    Test that create_payload output works end-to-end with the endpoint.

    Validates the full flow: create_payload → invoke → valid response,
    including extra kwargs like system prompt and temperature.

    Args:
        aws_credentials: Boto3 session with valid AWS credentials.
        bedrock_anthropic_mantle_region: AWS region for Bedrock Mantle.
        bedrock_anthropic_mantle_test_model: Anthropic model ID for Bedrock Mantle.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = AnthropicMessages(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )

    payload = AnthropicMessagesEndpoint.create_payload(
        user_message="What is 2 + 2? Answer with just the number.",
        max_tokens=10,
        system="You are a calculator. Only output numbers.",
    )

    response = endpoint.invoke(payload)

    assert response.error is None, f"Unexpected error: {response.error}"
    assert response.response_text is not None
    assert "4" in response.response_text, (
        f"Expected '4' in response, got: {response.response_text}"
    )
