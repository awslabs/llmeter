# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the LiteLLM endpoints, against Amazon Bedrock.

LiteLLM has no Bedrock-Mantle-specific provider, but Bedrock exposes an OpenAI-compatible
Chat Completions API - so these tests point LiteLLM's `openai` provider at that base URL with a
short-lived Bedrock bearer token. That exercises the real LiteLLM code path (its own chunk types,
its normalized `reasoning_content`/`thinking_blocks` fields) without needing a separate provider
account.

Credentials are supplied through **environment variables** rather than the request payload.
LiteLLM accepts `api_key`/`base_url` as `completion()` kwargs, and LLMeter's
`prepare_payload` would pass them straight through - but the payload is also recorded on
`InvocationResponse.input_payload` and persisted to `responses.jsonl`, so a token passed that
way would be written to disk.

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest tests/integ/test_litellm_bedrock.py -m integ

Required AWS Permissions:
    - bedrock:InvokeModelWithResponseStream (via the OpenAI-compatible endpoint)

Environment Variables:
    - AWS_REGION: AWS region for testing (default: us-east-1)
    - BEDROCK_REASONING_TEST_MODEL: Reasoning-capable model ID
      (default: openai.gpt-oss-120b-1:0, version suffix stripped)

Estimated Cost:
    - ~$0.003 total for all tests in this module
"""

import os

import pytest

try:
    from aws_bedrock_token_generator import provide_token

    from llmeter.endpoints.litellm import LiteLLM, LiteLLMStreaming

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from ._prompts import SIMPLE_ANSWER, SIMPLE_PROMPT

pytestmark = pytest.mark.skipif(
    not LITELLM_AVAILABLE, reason="litellm or aws_bedrock_token_generator not installed"
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
def litellm_reasoning_model():
    """LiteLLM model string for a reasoning-capable model on Bedrock's OpenAI-compatible API.

    The ``openai/`` prefix tells LiteLLM to speak the OpenAI Chat Completions protocol; the base URL
    (set via environment in :func:`bedrock_openai_env`) points it at Bedrock.
    """
    raw = os.environ.get("BEDROCK_REASONING_TEST_MODEL", "openai.gpt-oss-120b-1:0")
    # Note the doubled provider name is expected: "openai/" selects LiteLLM's OpenAI protocol
    # handler, while "openai.gpt-oss-120b" is the model name Bedrock itself expects.
    return f"openai/{_strip_model_version(raw)}"


@pytest.fixture
def bedrock_openai_env(monkeypatch, aws_credentials, aws_region):
    """Point LiteLLM's OpenAI provider at Bedrock, via environment variables.

    Deliberately not passed in the request payload: LLMeter persists ``input_payload`` alongside
    every response, so a bearer token there would end up in saved result files.
    """
    token = provide_token(region=aws_region)
    monkeypatch.setenv("OPENAI_API_KEY", token)
    monkeypatch.setenv("OPENAI_BASE_URL", _mantle_base_url(aws_region))
    monkeypatch.setenv("OPENAI_API_BASE", _mantle_base_url(aws_region))
    return aws_region


@pytest.mark.integ
def test_litellm_non_streaming(bedrock_openai_env, litellm_reasoning_model):
    """LiteLLM non-streaming: text, token counts, TTLT, and no first-token metrics.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = LiteLLM(litellm_model=litellm_reasoning_model)
    payload = LiteLLM.create_payload(SIMPLE_PROMPT, max_tokens=512)

    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"
    assert response.response_text is not None
    assert SIMPLE_ANSWER in response.response_text, (
        f"Expected {SIMPLE_ANSWER!r} in response, got: {response.response_text}"
    )
    assert response.num_tokens_input is not None and response.num_tokens_input > 0
    assert response.num_tokens_output is not None and response.num_tokens_output > 0

    assert response.time_to_last_token is not None and response.time_to_last_token > 0
    # Non-streaming: neither first-token metric is measurable
    assert response.time_to_first_token is None
    assert response.time_to_first_content_token is None


@pytest.mark.integ
def test_litellm_streaming_first_token_metrics(
    bedrock_openai_env, litellm_reasoning_model
):
    """LiteLLM streaming: both first-token metrics, ordered, plus reasoning detection.

    LiteLLM normalizes provider reasoning onto ``delta.reasoning_content`` /
    ``delta.thinking_blocks``. If it renamed those, ``reasoning_type`` would come back `None` and
    TTFT would silently revert to the first *visible* token.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = LiteLLMStreaming(litellm_model=litellm_reasoning_model)
    payload = LiteLLMStreaming.create_payload(SIMPLE_PROMPT, max_tokens=512)

    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"
    assert response.response_text is not None
    assert SIMPLE_ANSWER in response.response_text, (
        f"Expected {SIMPLE_ANSWER!r} in response, got: {response.response_text}"
    )

    assert response.time_to_first_token is not None, "TTFT should be measured"
    assert response.time_to_first_content_token is not None, (
        "Content TTFT should be measured"
    )
    assert response.time_to_first_token <= response.time_to_first_content_token, (
        f"TTFT ({response.time_to_first_token:.3f}s) must not exceed content TTFT "
        f"({response.time_to_first_content_token:.3f}s)"
    )
    assert response.time_to_last_token >= response.time_to_first_content_token

    # Reasoning content must never leak into the answer
    assert "thinking" not in response.response_text.lower()


@pytest.mark.integ
def test_litellm_streaming_reasoning_type_is_inferred(
    bedrock_openai_env, litellm_reasoning_model
):
    """`reasoning_type` is inferred from the LiteLLM model string, since the stream cannot show it.

    An ``openai/`` prefixed model is not Anthropic, so the inference should say `"verbatim"`. What a
    live call verifies is the *detection* half: that reasoning is present under a field name
    ``delta_has_reasoning_content`` recognizes.

    Separated from the timing test above so that a model or prompt which happens not to trigger
    reasoning fails here only, and diagnosably.

    Estimated Cost: ~$0.001 per run
    """
    endpoint = LiteLLMStreaming(litellm_model=litellm_reasoning_model)
    assert endpoint.default_reasoning_visibility == "verbatim", (
        f"An 'openai/'-prefixed model should infer 'verbatim', got "
        f"{endpoint.default_reasoning_visibility!r}"
    )

    payload = LiteLLMStreaming.create_payload(SIMPLE_PROMPT, max_tokens=512)
    response = endpoint.invoke(payload)

    assert response.error is None, f"Response error: {response.error}"
    assert response.reasoning_type == "verbatim", (
        f"Expected reasoning detected and inferred 'verbatim', got "
        f"{response.reasoning_type!r}. `None` means LiteLLM exposed no recognized reasoning "
        f"field -- check whether it renamed `reasoning_content`/`thinking_blocks`."
    )
    assert response.time_to_first_token < response.time_to_first_content_token, (
        "Reasoning should have arrived before the first visible token"
    )
