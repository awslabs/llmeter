# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for reasoning token parsing via OpenAI Response API.

This module tests that OpenAIResponseEndpoint and OpenAIResponseStreamEndpoint
correctly handle reasoning tokens when using a reasoning-capable model
(openai.gpt-oss-120b) through Bedrock's Mantle endpoint.

Tests cover:
- Non-streaming: num_tokens_output_reasoning extraction
- Streaming: `time_to_first_token` on the first reasoning delta, and
  `time_to_first_content_token` on the first visible text delta
- Comparison of inclusive vs visible-only TTFT

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest -m integ -k reasoning

Required AWS Permissions:
    - bedrock:InvokeModel
    - bedrock:InvokeModelWithResponseStream

Estimated Cost:
    - ~$0.003 per full run (reasoning models use more tokens)

Environment Variables:
    - AWS_REGION: AWS region for testing (default: us-east-1)
    - BEDROCK_REASONING_OPENAI_TEST_MODEL: Model ID for reasoning tests
      (default: openai.gpt-oss-120b-1:0, version suffix stripped automatically)
"""

import os

import pytest

try:
    from aws_bedrock_token_generator import provide_token

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ._prompts import SIMPLE_ANSWER, SIMPLE_PROMPT
from llmeter.endpoints.openai_response import (
    OpenAIResponseEndpoint,
    OpenAIResponseStreamEndpoint,
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
    """Get the reasoning-capable model ID for Response API tests.

    Defaults to openai.gpt-oss-120b (version suffix stripped automatically).
    """
    raw = os.environ.get(
        "BEDROCK_REASONING_OPENAI_TEST_MODEL", "openai.gpt-oss-120b-1:0"
    )
    return _strip_model_version(raw)


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_response_non_streaming_reasoning_tokens(
    aws_credentials, aws_region, reasoning_model_id
):
    """Test non-streaming Response API with a reasoning model.

    Validates that OpenAIResponseEndpoint correctly extracts:
    - Response text
    - num_tokens_output_reasoning from output_tokens_details
    - Standard token counts (input, output)
    - Timing information
    """
    token = provide_token(region=aws_region)

    endpoint = OpenAIResponseEndpoint(
        model_id=reasoning_model_id,
        api_key=token,
        base_url=_mantle_base_url(aws_region),
    )

    payload = OpenAIResponseEndpoint.create_payload(
        user_message=SIMPLE_PROMPT,
        max_output_tokens=200,
    )

    response = endpoint.invoke(payload)

    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert SIMPLE_ANSWER in response.response_text, (
        f"Expected {SIMPLE_ANSWER!r} in response, got: {response.response_text}"
    )

    # Verify token counts
    if response.num_tokens_input is not None:
        assert response.num_tokens_input > 0, "Input token count should be positive"
    if response.num_tokens_output is not None:
        assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify reasoning token count if available
    if response.num_tokens_output_reasoning is not None:
        assert response.num_tokens_output_reasoning >= 0, (
            "Reasoning token count should be non-negative"
        )
        if response.num_tokens_output is not None:
            assert response.num_tokens_output_reasoning <= response.num_tokens_output, (
                f"Reasoning tokens ({response.num_tokens_output_reasoning}) should be "
                f"<= total output tokens ({response.num_tokens_output})"
            )

    # Verify timing
    assert response.time_to_last_token is not None
    assert response.time_to_last_token > 0

    # Verify response ID
    assert response.id is not None, "Response should have an ID"


# ---------------------------------------------------------------------------
# Streaming — first-token metrics
# ---------------------------------------------------------------------------


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_response_streaming_reasoning_first_token_metrics(
    aws_credentials, aws_region, reasoning_model_id
):
    """Both first-token metrics are recorded from a single reasoning-model invocation.

    Validates that:
    - `time_to_first_token` (any delta, i.e. reasoning) is populated
    - `time_to_first_content_token` (first visible text delta) is populated
    - TTFT <= content TTFT <= TTLT, exact within a single request
    - Response text contains only visible content
    - num_tokens_output_reasoning is extracted from the completed event
    """
    token = provide_token(region=aws_region)

    endpoint = OpenAIResponseStreamEndpoint(
        model_id=reasoning_model_id,
        api_key=token,
        base_url=_mantle_base_url(aws_region),
    )

    payload = OpenAIResponseEndpoint.create_payload(
        user_message=SIMPLE_PROMPT,
        max_output_tokens=200,
    )

    response = endpoint.invoke(payload)

    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response text
    assert response.response_text is not None, "Response text should not be None"
    assert SIMPLE_ANSWER in response.response_text, (
        f"Expected {SIMPLE_ANSWER!r} in response, got: {response.response_text}"
    )

    # Verify timing. Both metrics come from the same stream, so the ordering is exact -
    # no jitter tolerance is needed.
    assert response.time_to_first_token is not None, "TTFT should not be None"
    assert response.time_to_first_token > 0, "TTFT should be positive"
    assert response.time_to_first_content_token is not None, (
        "Content TTFT should not be None"
    )
    assert response.time_to_first_token <= response.time_to_first_content_token, (
        f"TTFT ({response.time_to_first_token:.3f}s) must not exceed content TTFT "
        f"({response.time_to_first_content_token:.3f}s)"
    )
    assert response.time_to_last_token is not None, "TTLT should not be None"
    assert response.time_to_last_token >= response.time_to_first_content_token, (
        "TTLT should be >= content TTFT"
    )

    # Verify reasoning token count if available
    if response.num_tokens_output_reasoning is not None:
        assert response.num_tokens_output_reasoning >= 0
        if response.num_tokens_output is not None:
            assert response.num_tokens_output_reasoning <= response.num_tokens_output

    assert response.id is not None, "Response should have an ID"


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_response_streaming_reasoning_precedes_visible_text(
    aws_credentials, aws_region, reasoning_model_id
):
    """Reasoning deltas are detected, so TTFT lands strictly before the first visible token.

    This is the assertion that would fail if the reasoning delta event types stopped being
    recognized: both metrics would collapse onto the same text delta.

    Note this requires the model to actually stream reasoning deltas for the prompt. It is
    separated from the metric-plumbing test above so that a model or prompt that happens not
    to trigger reasoning fails here only, and diagnosably.
    """
    token = provide_token(region=aws_region)

    endpoint = OpenAIResponseStreamEndpoint(
        model_id=reasoning_model_id,
        api_key=token,
        base_url=_mantle_base_url(aws_region),
    )
    payload = OpenAIResponseEndpoint.create_payload(
        user_message=SIMPLE_PROMPT,
        max_output_tokens=200,
    )

    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"
    assert response.time_to_first_token is not None
    assert response.time_to_first_content_token is not None
    assert response.time_to_first_token < response.time_to_first_content_token, (
        f"Expected reasoning to precede visible text, but TTFT "
        f"({response.time_to_first_token:.3f}s) was not less than content TTFT "
        f"({response.time_to_first_content_token:.3f}s). Either the model streamed no "
        f"reasoning deltas, or they are no longer being detected."
    )


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_response_streaming_reasoning_type_is_detected(
    aws_credentials, aws_region, reasoning_model_id
):
    """`reasoning_type` is derived purely from event type names, so the API can invalidate it.

    This endpoint needs no declared default: `response.reasoning_text.delta` means the reasoning
    itself, `response.reasoning_summary_text.delta` means a summary of it. If either event were
    renamed or replaced, `reasoning_type` would silently fall back to `None` -- which the Runner
    reads as "this model did no reasoning" and would use to select the whole-output TPOT pairing.

    Deliberately asserts membership rather than one exact value: which of the two a given model
    emits is a model/deployment property, and pinning it would make this test brittle for no gain.

    Estimated Cost: ~$0.001 per run
    """
    token = provide_token(region=aws_region)

    endpoint = OpenAIResponseStreamEndpoint(
        model_id=reasoning_model_id,
        api_key=token,
        base_url=_mantle_base_url(aws_region),
    )
    payload = OpenAIResponseEndpoint.create_payload(
        user_message=SIMPLE_PROMPT,
        max_output_tokens=200,
    )

    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"
    assert response.reasoning_type in ("verbatim", "summary"), (
        f"A reasoning model should yield a detected disclosure level, got "
        f"{response.reasoning_type!r}. `None` here means neither reasoning event type was "
        f"recognized -- check whether the Responses API event names have changed."
    )

    # Whichever it is, reasoning arrived before the answer
    assert response.time_to_first_token < response.time_to_first_content_token

    # And it is consistent with the reasoning token count the API reported
    if response.num_tokens_output_reasoning is not None:
        assert response.num_tokens_output_reasoning > 0, (
            "A response with detected reasoning deltas should report reasoning tokens"
        )
