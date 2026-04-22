# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for reasoning token parsing via OpenAI Chat Completions API.

This module tests that OpenAICompletionEndpoint and OpenAICompletionStreamEndpoint
correctly extract reasoning token counts when using a reasoning-capable model
(openai.gpt-oss-120b) through Bedrock's OpenAI-compatible endpoint.

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest -m integ -k reasoning

Required AWS Permissions:
    - bedrock:InvokeModel
    - bedrock:InvokeModelWithResponseStream

Estimated Cost:
    - ~$0.001 per test run (reasoning models use more tokens)

Environment Variables:
    - AWS_REGION: AWS region for testing (default: us-east-1)
    - BEDROCK_OPENAI_TEST_MODEL: Model ID for OpenAI SDK tests
      (default: openai.gpt-oss-20b-1:0)
"""

import os

import pytest

try:
    from aws_bedrock_token_generator import provide_token

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from llmeter.endpoints.openai import (
    OpenAICompletionEndpoint,
    OpenAICompletionStreamEndpoint,
)


def _mantle_base_url(region: str) -> str:
    return f"https://bedrock-mantle.{region}.api.aws/v1"


def _strip_model_version(model_id: str) -> str:
    """Strip version suffix for Mantle API compatibility.

    Mantle requires model ID without version suffix
    (e.g., openai.gpt-oss-120b instead of openai.gpt-oss-120b-1:0).
    """
    if "-" in model_id and ":" in model_id:
        return model_id.rsplit("-", 1)[0]
    return model_id


@pytest.fixture(scope="module")
def reasoning_model_id():
    """Get the reasoning-capable model ID for Chat Completions API tests.

    Defaults to openai.gpt-oss-120b which supports reasoning via the
    Bedrock Mantle OpenAI-compatible Chat Completions API.

    The version suffix is stripped automatically since Mantle does not
    accept versioned model IDs.
    """
    raw = os.environ.get(
        "BEDROCK_REASONING_OPENAI_TEST_MODEL", "openai.gpt-oss-120b-1:0"
    )
    return _strip_model_version(raw)


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_completion_non_streaming_reasoning_tokens(
    aws_credentials, aws_region, reasoning_model_id
):
    """Test non-streaming Chat Completions with a reasoning model.

    Validates that OpenAICompletionEndpoint correctly extracts:
    - Response text
    - num_tokens_output_reasoning from completion_tokens_details
    - Standard token counts (input, output)
    - Timing information
    """
    token = provide_token(region=aws_region)
    base_url = _mantle_base_url(aws_region)

    endpoint = OpenAICompletionEndpoint(
        model_id=reasoning_model_id,
        api_key=token,
        base_url=base_url,
    )

    payload = OpenAICompletionEndpoint.create_payload(
        user_message="What is 15 * 37? Reply with just the number.",
        max_tokens=200,
    )

    response = endpoint.invoke(payload)

    # Verify no errors
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert "555" in response.response_text, (
        f"Expected '555' in response, got: {response.response_text}"
    )

    # Verify token counts
    assert response.num_tokens_input is not None, (
        "Input token count should not be None"
    )
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert response.num_tokens_output is not None, (
        "Output token count should not be None"
    )
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify reasoning token count is populated for a reasoning model
    if response.num_tokens_output_reasoning is not None:
        assert response.num_tokens_output_reasoning >= 0, (
            "Reasoning token count should be non-negative"
        )
        # Reasoning tokens should be included in total output tokens
        assert response.num_tokens_output_reasoning <= response.num_tokens_output, (
            f"Reasoning tokens ({response.num_tokens_output_reasoning}) should be "
            f"<= total output tokens ({response.num_tokens_output})"
        )

    # Verify timing
    assert response.time_to_last_token is not None, (
        "Time to last token should not be None"
    )
    assert response.time_to_last_token > 0, "Response time should be positive"

    # Verify response ID
    assert response.id is not None, "Response should have an ID"


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_completion_streaming_reasoning_tokens(
    aws_credentials, aws_region, reasoning_model_id
):
    """Test streaming Chat Completions with a reasoning model.

    Validates that OpenAICompletionStreamEndpoint correctly extracts:
    - Response text assembled from stream chunks
    - num_tokens_output_reasoning from the final usage chunk
    - Standard token counts (input, output)
    - TTFT and TTLT timing
    """
    token = provide_token(region=aws_region)
    base_url = _mantle_base_url(aws_region)

    endpoint = OpenAICompletionStreamEndpoint(
        model_id=reasoning_model_id,
        api_key=token,
        base_url=base_url,
    )

    payload = OpenAICompletionStreamEndpoint.create_payload(
        user_message="What is 15 * 37? Reply with just the number.",
        max_tokens=200,
    )

    response = endpoint.invoke(payload)

    # Verify no errors
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert "555" in response.response_text, (
        f"Expected '555' in response, got: {response.response_text}"
    )

    # Verify token counts
    assert response.num_tokens_input is not None, (
        "Input token count should not be None"
    )
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert response.num_tokens_output is not None, (
        "Output token count should not be None"
    )
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify reasoning token count is populated for a reasoning model
    if response.num_tokens_output_reasoning is not None:
        assert response.num_tokens_output_reasoning >= 0, (
            "Reasoning token count should be non-negative"
        )
        assert response.num_tokens_output_reasoning <= response.num_tokens_output, (
            f"Reasoning tokens ({response.num_tokens_output_reasoning}) should be "
            f"<= total output tokens ({response.num_tokens_output})"
        )

    # Verify TTFT
    assert response.time_to_first_token is not None, (
        "Time to first token should not be None"
    )
    assert response.time_to_first_token > 0, "TTFT should be positive"

    # Verify TTLT
    assert response.time_to_last_token is not None, (
        "Time to last token should not be None"
    )
    assert response.time_to_last_token > 0, "TTLT should be positive"

    # Verify TTLT >= TTFT
    assert response.time_to_last_token >= response.time_to_first_token, (
        "TTLT should be >= TTFT"
    )

    # Verify response ID
    assert response.id is not None, "Response should have an ID"
