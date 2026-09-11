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

from ._prompts import REASONING_ANSWER, REASONING_PROMPT
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


@pytest.mark.integ
def test_anthropic_messages_thinking_token_accounting(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
):
    """`usage.output_tokens_details.thinking_tokens` is read into num_tokens_output_reasoning.

    This is the assertion that catches the field being renamed, dropped, or surfaced in a shape
    our extraction doesn't handle -- note the anthropic SDK returns it as a plain dict on
    versions that don't model it, and as an object on versions that do.

    Also verifies the downstream consequence: with a thinking-token count available, TPOT stays
    derivable in `display: "omitted"` mode, where the answer-only pairing must be used.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = AnthropicMessagesStream(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )

    payload = AnthropicMessagesEndpoint.create_payload(
        REASONING_PROMPT,
        max_tokens=4096,
        thinking={"type": "adaptive", "display": "omitted"},
    )

    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"

    assert response.num_tokens_output is not None, (
        "Output token count should not be None"
    )
    assert response.num_tokens_output_reasoning is not None, (
        "Anthropic reports usage.output_tokens_details.thinking_tokens, so "
        "num_tokens_output_reasoning should be populated. If this fails, check whether the "
        "field has been renamed or is arriving in an unexpected shape."
    )
    assert response.num_tokens_output_reasoning > 0, (
        "Model declined to think for this prompt (adaptive thinking is the model's choice), so "
        "the reasoning path could not be exercised on this run"
    )
    assert response.num_tokens_output_reasoning <= response.num_tokens_output, (
        f"Thinking tokens ({response.num_tokens_output_reasoning}) should be <= total output "
        f"tokens ({response.num_tokens_output}) -- the total is inclusive"
    )

    # The disclosure level is recorded, but not pinned to a specific value: `display: "omitted"` is
    # requested here, yet Bedrock Mantle was observed streaming summarized thinking anyway (see
    # test_anthropic_messages_reasoning_is_detected_for_both_display_modes). What matters for this
    # test is that reasoning was detected at all, so the token accounting below is meaningful.
    assert response.reasoning_type is not None, (
        f"Reasoning happened (thinking_tokens={response.num_tokens_output_reasoning}) but was "
        f"not detected"
    )
    assert response.time_to_first_token is not None, (
        "TTFT should record the first model output received"
    )
    assert response.time_to_first_content_token is not None, (
        "Content TTFT should also be measured"
    )
    assert response.time_to_first_token <= response.time_to_first_content_token, (
        f"TTFT ({response.time_to_first_token:.3f}s) must not exceed content TTFT "
        f"({response.time_to_first_content_token:.3f}s)"
    )

    # ...and the ingredients for the answer-only TPOT pairing are all present. The pairing
    # arithmetic itself is covered by the unit tests; what only a live call can confirm is that
    # the provider supplies the inputs it needs.
    assert response.time_to_last_token is not None
    assert response.num_tokens_output - response.num_tokens_output_reasoning >= 0, (
        "Visible token count should not be negative"
    )


@pytest.mark.integ
@pytest.mark.parametrize("display", ["summarized", "omitted"])
def test_anthropic_messages_reasoning_is_detected_for_both_display_modes(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
    display,
):
    """Reasoning must be detected, and the two first-token metrics ordered, in either display mode.

    Deliberately does *not* pin `reasoning_type` per mode. The API documentation describes
    `thinking.display: "omitted"` as suppressing thinking deltas (which LLMeter would detect
    structurally as `"redacted"`), but Bedrock Mantle was observed returning *summarized* thinking
    for both settings even though `display` is sent correctly in the payload. Asserting a
    per-mode value would therefore pin provider behaviour we do not control, rather than LLMeter's.
    The `"redacted"` detection path is covered by unit tests instead.

    Estimated Cost: ~$0.004 per run (two invocations, thinking tokens)
    """
    endpoint = AnthropicMessagesStream(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )

    payload = AnthropicMessagesEndpoint.create_payload(
        REASONING_PROMPT,
        max_tokens=4096,
        thinking={"type": "adaptive", "display": display},
    )

    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"
    assert response.num_tokens_output_reasoning > 0, (
        "Model declined to think for this prompt (adaptive thinking is the model's choice), so "
        "the reasoning path could not be exercised on this run"
    )
    assert response.reasoning_type is not None, (
        f"Reasoning happened (thinking_tokens="
        f"{response.num_tokens_output_reasoning}) but was not detected. `None` means no "
        f"thinking delta, signature or redacted block was recognized in the stream."
    )
    assert response.reasoning_type in ("summary", "verbatim", "redacted"), (
        f"Unexpected reasoning_type {response.reasoning_type!r}"
    )

    # Both metrics recorded and ordered, whatever the disclosure level
    assert response.time_to_first_token is not None
    assert response.time_to_first_content_token is not None
    assert response.time_to_first_token <= response.time_to_first_content_token

    # Reasoning content must never leak into the answer
    assert response.response_text is not None
    assert REASONING_ANSWER in response.response_text, (
        f"Expected {REASONING_ANSWER!r} in response, got: {response.response_text}"
    )


@pytest.mark.integ
def test_anthropic_messages_streamed_thinking_precedes_text(
    aws_credentials,
    bedrock_anthropic_mantle_region,
    bedrock_anthropic_mantle_test_model,
):
    """When thinking is streamed, TTFT lands strictly before the first visible token.

    Separated from the test above so that a run where the model declines to think, or where
    thinking is not streamed at all, fails here only and diagnosably.

    Estimated Cost: ~$0.002 per run
    """
    endpoint = AnthropicMessagesStream(
        model_id=bedrock_anthropic_mantle_test_model,
        provider="bedrock-mantle",
        aws_region=bedrock_anthropic_mantle_region,
    )
    payload = AnthropicMessagesEndpoint.create_payload(
        REASONING_PROMPT,
        max_tokens=4096,
        thinking={"type": "adaptive", "display": "summarized"},
    )

    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"
    assert response.reasoning_type == "summary"
    assert response.time_to_first_token < response.time_to_first_content_token, (
        f"Expected streamed thinking to precede the text, but TTFT "
        f"({response.time_to_first_token:.3f}s) was not less than content TTFT "
        f"({response.time_to_first_content_token:.3f}s)."
    )
