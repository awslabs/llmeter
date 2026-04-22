# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for reasoning token parsing via Bedrock Converse API.

This module tests that BedrockConverseStream correctly handles reasoning
(extended thinking) content when using a reasoning-capable model
(openai.gpt-oss-120b) through the Bedrock Converse API.

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest -m integ -k reasoning

Required AWS Permissions:
    - bedrock:InvokeModelWithResponseStream

Estimated Cost:
    - ~$0.001 per test run (reasoning models use more tokens)

Environment Variables:
    - AWS_REGION: AWS region for testing (default: us-east-1)
    - BEDROCK_REASONING_TEST_MODEL: Model ID for reasoning tests
      (default: openai.gpt-oss-120b-1:0)
"""

import os

import pytest

from llmeter.endpoints.bedrock import BedrockConverseStream


@pytest.fixture(scope="module")
def reasoning_model_id():
    """Get the reasoning-capable model ID for Converse API tests.

    Defaults to openai.gpt-oss-120b-1:0 which supports extended thinking
    via the Bedrock Converse API.
    """
    return os.environ.get(
        "BEDROCK_REASONING_TEST_MODEL", "openai.gpt-oss-120b-1:0"
    )


@pytest.mark.integ
def test_converse_stream_reasoning_visible_ttft(
    aws_credentials, aws_region, reasoning_model_id
):
    """Test streaming with a reasoning model and ttft_visible_tokens_only=True (default).

    Validates that:
    - The endpoint successfully invokes a reasoning-capable model
    - Response text is returned (reasoning content is not included)
    - TTFT is measured on the first visible text token
    - TTLT >= TTFT
    - Token counts are populated

    With ttft_visible_tokens_only=True, TTFT should reflect the time to the
    first *visible* text token, excluding any reasoning/thinking time.
    """
    endpoint = BedrockConverseStream(
        model_id=reasoning_model_id,
        region=aws_region,
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "What is 15 * 37? Reply with just the number."}
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 200},
    }

    response = endpoint.invoke(payload)

    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert "555" in response.response_text, (
        f"Expected '555' in response, got: {response.response_text}"
    )

    # Verify timing
    assert response.time_to_first_token is not None, (
        "Time to first token should not be None"
    )
    assert response.time_to_first_token > 0, "TTFT should be positive"
    assert response.time_to_last_token is not None, (
        "Time to last token should not be None"
    )
    assert response.time_to_last_token >= response.time_to_first_token, (
        "TTLT should be >= TTFT"
    )

    # Verify token counts
    if response.num_tokens_input is not None:
        assert response.num_tokens_input > 0, "Input token count should be positive"
    if response.num_tokens_output is not None:
        assert response.num_tokens_output > 0, "Output token count should be positive"


@pytest.mark.integ
def test_converse_stream_reasoning_inclusive_ttft(
    aws_credentials, aws_region, reasoning_model_id
):
    """Test streaming with a reasoning model and ttft_visible_tokens_only=False.

    Validates that:
    - TTFT is measured on the first token of any kind (including reasoning)
    - TTFT with reasoning included should be <= TTFT with visible-only
    - Response text still contains only visible content
    - TTLT >= TTFT

    This test creates two endpoints — one with visible-only TTFT and one with
    inclusive TTFT — and compares their timing to verify reasoning tokens are
    being detected.
    """
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "What is 15 * 37? Reply with just the number."}
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 200},
    }

    # Inclusive TTFT (includes reasoning tokens)
    endpoint_inclusive = BedrockConverseStream(
        model_id=reasoning_model_id,
        region=aws_region,
        ttft_visible_tokens_only=False,
    )
    response_inclusive = endpoint_inclusive.invoke(payload)

    assert response_inclusive.error is None, (
        f"Inclusive response should not contain errors: {response_inclusive.error}"
    )

    # Verify response text
    assert response_inclusive.response_text is not None, (
        "Response text should not be None"
    )
    assert "555" in response_inclusive.response_text, (
        f"Expected '555' in response, got: {response_inclusive.response_text}"
    )

    # Verify timing
    assert response_inclusive.time_to_first_token is not None, (
        "Inclusive TTFT should not be None"
    )
    assert response_inclusive.time_to_first_token > 0, (
        "Inclusive TTFT should be positive"
    )
    assert response_inclusive.time_to_last_token is not None, (
        "TTLT should not be None"
    )
    assert response_inclusive.time_to_last_token >= response_inclusive.time_to_first_token, (
        "TTLT should be >= TTFT"
    )

    # Visible-only TTFT
    endpoint_visible = BedrockConverseStream(
        model_id=reasoning_model_id,
        region=aws_region,
        ttft_visible_tokens_only=True,
    )
    response_visible = endpoint_visible.invoke(payload)

    assert response_visible.error is None, (
        f"Visible response should not contain errors: {response_visible.error}"
    )
    assert response_visible.time_to_first_token is not None, (
        "Visible TTFT should not be None"
    )

    # The inclusive TTFT should be <= visible-only TTFT because reasoning
    # tokens arrive before visible text tokens. We allow a small tolerance
    # for network jitter.
    assert response_inclusive.time_to_first_token <= response_visible.time_to_first_token + 0.5, (
        f"Inclusive TTFT ({response_inclusive.time_to_first_token:.3f}s) should be "
        f"<= visible TTFT ({response_visible.time_to_first_token:.3f}s) + tolerance"
    )
