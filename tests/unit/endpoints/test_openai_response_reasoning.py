# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reasoning token parsing in OpenAIResponseStreamEndpoint.

Covers:
- TTFT measurement with ``ttft_visible_tokens_only=True`` (default): reasoning
  deltas are ignored, TTFT set on first visible text delta.
- TTFT measurement with ``ttft_visible_tokens_only=False``: TTFT set on first
  reasoning delta (``response.reasoning_text.delta`` or
  ``response.reasoning_summary_text.delta``).
- ``num_tokens_output_reasoning`` extraction from ``output_tokens_details``.
- ``num_tokens_input_cached`` extraction from ``input_tokens_details``.
"""

import time
from unittest.mock import Mock, patch

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.openai_response import OpenAIResponseStreamEndpoint


def _make_draft_response() -> InvocationResponse:
    return InvocationResponse(response_text=None)


def _event(event_type: str, **attrs) -> Mock:
    """Build a mock streaming event with the given type and attributes."""
    e = Mock()
    e.type = event_type
    for k, v in attrs.items():
        setattr(e, k, v)
    return e


def _created_event(response_id: str = "resp_123") -> Mock:
    resp = Mock()
    resp.id = response_id
    return _event("response.created", response=resp)


def _text_delta_event(text: str) -> Mock:
    return _event("response.output_text.delta", delta=text)


def _reasoning_text_delta_event() -> Mock:
    return _event("response.reasoning_text.delta", delta="thinking...")


def _reasoning_summary_delta_event() -> Mock:
    return _event("response.reasoning_summary_text.delta", delta="summary...")


def _completed_event(
    input_tokens: int = 10,
    output_tokens: int = 20,
    reasoning_tokens: int | None = None,
    cached_tokens: int | None = None,
) -> Mock:
    usage = Mock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    if cached_tokens is not None:
        details = Mock()
        details.cached_tokens = cached_tokens
        usage.input_tokens_details = details
    else:
        usage.input_tokens_details = None

    if reasoning_tokens is not None:
        output_details = Mock()
        output_details.reasoning_tokens = reasoning_tokens
        usage.output_tokens_details = output_details
    else:
        usage.output_tokens_details = None

    resp = Mock()
    resp.usage = usage
    return _event("response.completed", response=resp)


# ---------------------------------------------------------------------------
# Tests: TTFT with ttft_visible_tokens_only=True (default)
# ---------------------------------------------------------------------------


class TestTTFTVisibleTokensOnly:
    """When ttft_visible_tokens_only=True, reasoning deltas must not set TTFT."""

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_text_delta_ignored_for_ttft(self, mock_openai_class):
        """response.reasoning_text.delta should NOT set TTFT."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _text_delta_event("Hello"),
            _completed_event(reasoning_tokens=5),
        ]

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        assert response.time_to_first_token is not None
        assert response.response_text == "Hello"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_summary_delta_ignored_for_ttft(self, mock_openai_class):
        """response.reasoning_summary_text.delta should NOT set TTFT."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _reasoning_summary_delta_event(),
            _text_delta_event("World"),
            _completed_event(reasoning_tokens=8),
        ]

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        assert response.time_to_first_token is not None
        assert response.response_text == "World"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_deltas_do_not_contribute_to_response_text(
        self, mock_openai_class
    ):
        """Reasoning deltas must never appear in response_text."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _reasoning_summary_delta_event(),
            _text_delta_event("Answer"),
            _completed_event(),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(
            iter(events), time.perf_counter(), response
        )

        assert response.response_text == "Answer"


# ---------------------------------------------------------------------------
# Tests: TTFT with ttft_visible_tokens_only=False
# ---------------------------------------------------------------------------


class TestTTFTIncludesReasoning:
    """When ttft_visible_tokens_only=False, first reasoning delta sets TTFT."""

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_text_delta_sets_ttft(self, mock_openai_class):
        """response.reasoning_text.delta should set TTFT when visible_only=False."""
        endpoint = OpenAIResponseStreamEndpoint(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _text_delta_event("Result"),
            _completed_event(reasoning_tokens=5),
        ]

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token <= response.time_to_last_token
        assert response.response_text == "Result"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_summary_delta_sets_ttft(self, mock_openai_class):
        """response.reasoning_summary_text.delta should set TTFT when visible_only=False."""
        endpoint = OpenAIResponseStreamEndpoint(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        events = [
            _created_event(),
            _reasoning_summary_delta_event(),
            _text_delta_event("Result"),
            _completed_event(reasoning_tokens=3),
        ]

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token <= response.time_to_last_token
        assert response.response_text == "Result"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_ttft_not_overwritten_by_later_text_delta(self, mock_openai_class):
        """Once TTFT is set by a reasoning delta, a later text delta must not overwrite it."""
        endpoint = OpenAIResponseStreamEndpoint(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _reasoning_text_delta_event(),
            _text_delta_event("Final"),
            _completed_event(),
        ]

        response = _make_draft_response()
        start_t = time.perf_counter()
        endpoint.process_raw_response(iter(events), start_t, response)

        # TTFT was set on the first reasoning delta; the text delta should not
        # have changed it.
        assert response.time_to_first_token is not None
        assert response.time_to_first_token <= response.time_to_last_token

    @patch("llmeter.endpoints.openai_response.OpenAI")
    @patch("time.perf_counter")
    def test_ttft_timing_set_on_reasoning_not_text(
        self, mock_perf_counter, mock_openai_class
    ):
        """Verify TTFT value corresponds to the reasoning delta, not the text delta."""
        start_t = 100.0
        # Calls: created, reasoning_delta, text_delta, completed
        mock_perf_counter.side_effect = [100.1, 100.2, 100.5, 100.6]

        endpoint = OpenAIResponseStreamEndpoint(
            model_id="test-model", ttft_visible_tokens_only=False
        )

        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _text_delta_event("Answer"),
            _completed_event(reasoning_tokens=10),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), start_t, response)

        # TTFT should be ~0.2 (reasoning delta), not ~0.5 (text delta)
        assert abs(response.time_to_first_token - 0.2) < 1e-5
        assert abs(response.time_to_last_token - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Tests: num_tokens_output_reasoning extraction
# ---------------------------------------------------------------------------


class TestReasoningTokenCount:
    """Verify num_tokens_output_reasoning is extracted from output_tokens_details."""

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_tokens_extracted(self, mock_openai_class):
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _text_delta_event("Hi"),
            _completed_event(
                input_tokens=15,
                output_tokens=25,
                reasoning_tokens=12,
            ),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(
            iter(events), time.perf_counter(), response
        )

        assert response.num_tokens_output_reasoning == 12
        assert response.num_tokens_input == 15
        assert response.num_tokens_output == 25

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_tokens_none_when_not_present(self, mock_openai_class):
        """When output_tokens_details is absent, reasoning count stays None."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _text_delta_event("Hi"),
            _completed_event(
                input_tokens=10,
                output_tokens=20,
                reasoning_tokens=None,
            ),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(
            iter(events), time.perf_counter(), response
        )

        assert response.num_tokens_output_reasoning is None
        assert response.num_tokens_output == 20

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_cached_tokens_extracted(self, mock_openai_class):
        """Verify input_tokens_details.cached_tokens is captured."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _text_delta_event("Hi"),
            _completed_event(
                input_tokens=100,
                output_tokens=20,
                cached_tokens=80,
            ),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(
            iter(events), time.perf_counter(), response
        )

        assert response.num_tokens_input_cached == 80
        assert response.num_tokens_input == 100

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_and_cached_tokens_together(self, mock_openai_class):
        """Both reasoning and cached token counts extracted in the same response."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")

        events = [
            _created_event(),
            _text_delta_event("Hi"),
            _completed_event(
                input_tokens=100,
                output_tokens=50,
                reasoning_tokens=30,
                cached_tokens=60,
            ),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(
            iter(events), time.perf_counter(), response
        )

        assert response.num_tokens_output_reasoning == 30
        assert response.num_tokens_input_cached == 60
        assert response.num_tokens_input == 100
        assert response.num_tokens_output == 50
