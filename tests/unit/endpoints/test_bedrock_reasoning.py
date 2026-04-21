# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reasoning token parsing in BedrockConverseStream.

Covers:
- TTFT measurement with ``ttft_visible_tokens_only=True`` (default): reasoning
  content deltas are ignored, TTFT set on first visible text delta.
- TTFT measurement with ``ttft_visible_tokens_only=False``: TTFT set on first
  ``reasoningContent`` delta.
- Reasoning deltas do not contribute to ``response_text``.
"""

import time
from unittest.mock import patch

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.bedrock import BedrockConverseStream


def _make_draft_response() -> InvocationResponse:
    return InvocationResponse(response_text=None, id=None)


def _stream_response(stream_chunks: list[dict]) -> dict:
    """Wrap stream chunks in the Bedrock ConverseStream response envelope."""
    return {
        "stream": stream_chunks,
        "ResponseMetadata": {"RequestId": "req-123", "RetryAttempts": 0},
    }


# ---------------------------------------------------------------------------
# Tests: TTFT with ttft_visible_tokens_only=True (default)
# ---------------------------------------------------------------------------


class TestTTFTVisibleTokensOnly:
    """When ttft_visible_tokens_only=True, reasoningContent deltas must not set TTFT."""

    def test_reasoning_delta_ignored_for_ttft(self):
        """reasoningContent delta should NOT set TTFT when visible_only=True."""
        endpoint = BedrockConverseStream(model_id="test-model")

        raw = _stream_response([
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking..."}}}},
            {"contentBlockDelta": {"delta": {"text": "Hello"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
        ])

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(raw, start_t, response)

        assert response.time_to_first_token is not None
        assert response.response_text == "Hello"

    def test_multiple_reasoning_deltas_ignored(self):
        """Multiple reasoningContent deltas should all be ignored for TTFT."""
        endpoint = BedrockConverseStream(model_id="test-model")

        raw = _stream_response([
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "step 1"}}}},
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "step 2"}}}},
            {"contentBlockDelta": {"delta": {"text": "Answer"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
        ])

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(raw, start_t, response)

        assert response.time_to_first_token is not None
        assert response.response_text == "Answer"

    def test_reasoning_deltas_do_not_contribute_to_response_text(self):
        """reasoningContent deltas must never appear in response_text."""
        endpoint = BedrockConverseStream(model_id="test-model")

        raw = _stream_response([
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "internal thought"}}}},
            {"contentBlockDelta": {"delta": {"text": "Visible"}}},
            {"contentBlockDelta": {"delta": {"text": " answer"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
        ])

        response = _make_draft_response()
        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.response_text == "Visible answer"

    @patch("time.perf_counter")
    def test_ttft_set_on_text_not_reasoning(self, mock_perf_counter):
        """Verify TTFT value corresponds to the text delta, not the reasoning delta."""
        start_t = 100.0
        # Calls: reasoning_delta, text_delta, contentBlockStop, metadata
        mock_perf_counter.side_effect = [100.3, 100.5, 100.6, 100.7]

        endpoint = BedrockConverseStream(model_id="test-model")

        raw = _stream_response([
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking"}}}},
            {"contentBlockDelta": {"delta": {"text": "Result"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
        ])

        response = _make_draft_response()
        endpoint.process_raw_response(raw, start_t, response)

        # TTFT should be ~0.5 (text delta), not ~0.3 (reasoning delta)
        assert abs(response.time_to_first_token - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Tests: TTFT with ttft_visible_tokens_only=False
# ---------------------------------------------------------------------------


class TestTTFTIncludesReasoning:
    """When ttft_visible_tokens_only=False, first reasoningContent delta sets TTFT."""

    def test_reasoning_delta_sets_ttft(self):
        """reasoningContent delta should set TTFT when visible_only=False."""
        endpoint = BedrockConverseStream(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        raw = _stream_response([
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking..."}}}},
            {"contentBlockDelta": {"delta": {"text": "Result"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
        ])

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(raw, start_t, response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token <= response.time_to_last_token
        assert response.response_text == "Result"

    def test_ttft_not_overwritten_by_later_text_delta(self):
        """Once TTFT is set by reasoning delta, text delta must not overwrite it."""
        endpoint = BedrockConverseStream(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        raw = _stream_response([
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "step 1"}}}},
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "step 2"}}}},
            {"contentBlockDelta": {"delta": {"text": "Final"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
        ])

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(raw, start_t, response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token <= response.time_to_last_token

    @patch("time.perf_counter")
    def test_ttft_timing_set_on_reasoning_not_text(self, mock_perf_counter):
        """Verify TTFT value corresponds to the reasoning delta, not the text delta."""
        start_t = 100.0
        # Calls: reasoning_delta, text_delta, contentBlockStop, metadata
        mock_perf_counter.side_effect = [100.2, 100.5, 100.6, 100.7]

        endpoint = BedrockConverseStream(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        raw = _stream_response([
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking"}}}},
            {"contentBlockDelta": {"delta": {"text": "Answer"}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
        ])

        response = _make_draft_response()
        endpoint.process_raw_response(raw, start_t, response)

        # TTFT should be ~0.2 (reasoning delta), not ~0.5 (text delta)
        assert abs(response.time_to_first_token - 0.2) < 1e-5
        assert abs(response.time_to_last_token - 0.6) < 1e-5

    def test_only_reasoning_no_text(self):
        """Stream with only reasoning deltas and no text — TTFT set, no response_text."""
        endpoint = BedrockConverseStream(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        raw = _stream_response([
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking"}}}},
            {"contentBlockStop": {}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 0}}},
        ])

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(raw, start_t, response)

        assert response.time_to_first_token is not None
        assert response.response_text is None
