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

from ._prompts import (
    REASONING_ANSWER,
    REASONING_PROMPT,
    SIMPLE_ANSWER,
    SIMPLE_PROMPT,
)


@pytest.fixture(scope="module")
def reasoning_model_id():
    """Get the reasoning-capable model ID for Converse API tests.

    Defaults to openai.gpt-oss-120b-1:0 which supports extended thinking
    via the Bedrock Converse API.
    """
    return os.environ.get("BEDROCK_REASONING_TEST_MODEL", "openai.gpt-oss-120b-1:0")


def _payload() -> dict:
    return {
        "messages": [{"role": "user", "content": [{"text": SIMPLE_PROMPT}]}],
        "inferenceConfig": {"maxTokens": 200},
    }


@pytest.mark.integ
def test_converse_stream_reasoning_first_token_metrics(
    aws_credentials, aws_region, reasoning_model_id
):
    """Both first-token metrics are recorded from a single reasoning-model invocation.

    Validates that:
    - The endpoint successfully invokes a reasoning-capable model
    - `time_to_first_token` (any token, i.e. reasoning) is populated
    - `time_to_first_content_token` (first visible token) is populated
    - TTFT <= content TTFT <= TTLT, which holds within a single request by construction
    - Response text contains only visible content
    - Token counts are populated
    """
    endpoint = BedrockConverseStream(model_id=reasoning_model_id, region=aws_region)

    response = endpoint.invoke(_payload())

    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
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

    # Verify token counts
    if response.num_tokens_input is not None:
        assert response.num_tokens_input > 0, "Input token count should be positive"
    if response.num_tokens_output is not None:
        assert response.num_tokens_output > 0, "Output token count should be positive"


@pytest.mark.integ
def test_converse_stream_reasoning_precedes_visible_text(
    aws_credentials, aws_region, reasoning_model_id
):
    """Reasoning deltas are detected, so TTFT lands strictly before the first visible token.

    This is the assertion that would fail if `reasoningContent` deltas stopped being
    recognized: both metrics would collapse onto the same text delta.

    Note this requires the model to actually emit reasoning content for the prompt. It is
    separated from the metric-plumbing test above so that a model or prompt that happens not
    to trigger reasoning fails here only, and diagnosably.
    """
    endpoint = BedrockConverseStream(model_id=reasoning_model_id, region=aws_region)

    response = endpoint.invoke(_payload())

    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )
    assert response.time_to_first_token is not None
    assert response.time_to_first_content_token is not None
    assert response.time_to_first_token < response.time_to_first_content_token, (
        f"Expected reasoning to precede visible text, but TTFT "
        f"({response.time_to_first_token:.3f}s) was not less than content TTFT "
        f"({response.time_to_first_content_token:.3f}s). Either the model emitted no "
        f"reasoning content, or reasoning deltas are no longer being detected."
    )


@pytest.mark.integ
def test_converse_stream_reasoning_type_inferred_for_non_anthropic(
    aws_credentials, aws_region, reasoning_model_id
):
    """A non-Anthropic Converse model should be detected as reasoning, and inferred `"verbatim"`.

    Converse gives no structural signal for verbatim-vs-summary, so the value itself comes from the
    model ID. What a live call verifies is that `reasoningContent` deltas arrive at all -- if they
    stopped, `reasoning_type` would be `None` and TTFT would quietly become the first visible token.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = BedrockConverseStream(model_id=reasoning_model_id, region=aws_region)
    assert "anthropic" not in reasoning_model_id.split("."), (
        f"This test assumes a non-Anthropic model; got {reasoning_model_id!r}"
    )

    response = endpoint.invoke(_payload())

    assert response.error is None, f"Response error: {response.error}"
    assert response.reasoning_type == "verbatim", (
        f"Expected reasoning detected and inferred 'verbatim', got "
        f"{response.reasoning_type!r}. `None` means no `reasoningContent` delta was seen."
    )
    assert response.time_to_first_token < response.time_to_first_content_token


@pytest.mark.integ
def test_converse_stream_reasoning_type_inferred_for_anthropic(
    aws_credentials, aws_region, bedrock_anthropic_converse_test_model
):
    """An Anthropic model via Converse should infer `"summary"`, matching Claude 4 behaviour.

    Also confirms the Converse path streams *readable* `reasoningContent.text` for Claude rather
    than only `redactedContent` -- if it were redacted, detection would report `"redacted"` and the
    model-ID inference would never be consulted.

    Estimated Cost: ~$0.002 per run (Claude pricing)
    """
    endpoint = BedrockConverseStream(
        model_id=bedrock_anthropic_converse_test_model, region=aws_region
    )

    payload = {
        "messages": [{"role": "user", "content": [{"text": REASONING_PROMPT}]}],
        "inferenceConfig": {"maxTokens": 4096},
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"}
        },
    }

    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"
    if response.reasoning_type is None:
        pytest.skip(
            "No reasoningContent delta arrived: the model declined to think for this prompt "
            "(adaptive thinking is the model's choice). Converse reports no thinking-token "
            "count, so 'declined' cannot be distinguished from 'detection broke' here."
        )
    assert response.reasoning_type == "summary", (
        f"Anthropic model via Converse should infer 'summary', got "
        f"{response.reasoning_type!r}. 'redacted' would mean only encrypted reasoning arrived; "
        f"None would mean no reasoning delta was seen at all."
    )
    assert response.time_to_first_token <= response.time_to_first_content_token

    # The model must still have answered correctly -- reasoning content must never displace or
    # contaminate the visible answer.
    assert response.response_text is not None
    assert REASONING_ANSWER in response.response_text, (
        f"Expected {REASONING_ANSWER!r} in response, got: {response.response_text}"
    )
