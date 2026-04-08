# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for ResponseEndpoint and ResponseStreamEndpoint.

This module contains integration tests that verify the llmeter ResponseEndpoint
and ResponseStreamEndpoint wrappers work correctly with Bedrock's Response API
via the Mantle endpoint.

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

try:
    from aws_bedrock_token_generator import provide_token

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from llmeter.endpoints.openai_response import OpenAIResponseEndpoint, OpenAIResponseStreamEndpoint


def _mantle_base_url(region: str) -> str:
    return f"https://bedrock-mantle.{region}.api.aws/v1"


def _strip_model_version(model_id: str) -> str:
    """Strip version suffix for Response API compatibility.

    Response API requires model ID without version suffix
    (e.g., openai.gpt-oss-120b instead of openai.gpt-oss-120b-1:0).
    """
    if "-" in model_id and ":" in model_id:
        return model_id.rsplit("-", 1)[0]
    return model_id


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_response_endpoint_non_streaming(
    aws_credentials, aws_region, bedrock_openai_test_model
):
    """
    Test ResponseEndpoint wrapper with Bedrock's Response API.

    Validates that the llmeter ResponseEndpoint can:
    - Initialize with Bedrock Mantle endpoint credentials
    - Invoke the Response API via create_payload helper
    - Return an InvocationResponse with text, timing, and optionally token counts
    - Complete without errors
    """
    token = provide_token(region=aws_region)
    model_id = _strip_model_version(bedrock_openai_test_model)

    endpoint = OpenAIResponseEndpoint(
        model_id=model_id,
        api_key=token,
        base_url=_mantle_base_url(aws_region),
    )

    payload = OpenAIResponseEndpoint.create_payload(
        user_message="Hello, this is a test message. Please respond with a brief greeting.",
        max_output_tokens=100,
    )

    response = endpoint.invoke(payload)

    # Verify no errors
    assert response.error is None, f"Response should not contain errors: {response.error}"

    # Verify response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str), "Response text should be a string"

    # Verify token counts if available (Bedrock Mantle may not return usage)
    if response.num_tokens_input is not None:
        assert response.num_tokens_input > 0, "Input token count should be positive"
    if response.num_tokens_output is not None:
        assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify timing
    assert response.time_to_last_token is not None, "Response time should not be None"
    assert response.time_to_last_token > 0, "Response time should be positive"

    # Verify response ID
    assert response.id is not None, "Response should have an ID"


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_response_endpoint_streaming(
    aws_credentials, aws_region, bedrock_openai_test_model
):
    """
    Test ResponseStreamEndpoint wrapper with Bedrock's Response API.

    Validates that the llmeter ResponseStreamEndpoint can:
    - Initialize with Bedrock Mantle endpoint credentials
    - Invoke the streaming Response API
    - Return an InvocationResponse with text, TTFT, and TTLT
    - Verify TTLT > TTFT
    - Complete without errors
    """
    token = provide_token(region=aws_region)
    model_id = _strip_model_version(bedrock_openai_test_model)

    endpoint = OpenAIResponseStreamEndpoint(
        model_id=model_id,
        api_key=token,
        base_url=_mantle_base_url(aws_region),
    )

    payload = OpenAIResponseEndpoint.create_payload(
        user_message="Hello, this is a test message. Please respond with a brief greeting.",
        max_output_tokens=100,
    )

    response = endpoint.invoke(payload)

    # Verify no errors
    assert response.error is None, f"Response should not contain errors: {response.error}"

    # Verify response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str), "Response text should be a string"

    # Verify token counts if available (Bedrock Mantle may not return usage)
    if response.num_tokens_input is not None:
        assert response.num_tokens_input > 0, "Input token count should be positive"
    if response.num_tokens_output is not None:
        assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify TTFT
    assert response.time_to_first_token is not None, "Time to first token should not be None"
    assert response.time_to_first_token > 0, "Time to first token should be positive"

    # Verify TTLT
    assert response.time_to_last_token is not None, "Time to last token should not be None"
    assert response.time_to_last_token > 0, "Time to last token should be positive"

    # Verify TTLT > TTFT
    assert (
        response.time_to_last_token > response.time_to_first_token
    ), "Time to last token should be greater than time to first token"

    # Verify response ID
    assert response.id is not None, "Response should have an ID"
