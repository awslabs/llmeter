# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reasoning token parsing in OpenAIResponseStreamEndpoint.

Covers:
- ``time_to_first_token`` is set by the first delta of any kind, including reasoning
  deltas (``response.reasoning_text.delta``, ``response.reasoning_summary_text.delta``).
- ``time_to_first_content_token`` is set only by the first
  ``response.output_text.delta``.
- Reasoning deltas never contribute to ``response_text``.
- ``num_tokens_output_reasoning`` extraction from ``output_tokens_details``.
- ``num_tokens_input_cached`` extraction from ``input_tokens_details``.
- The retired ``ttft_visible_tokens_only`` argument warns and is ignored.
"""

import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.openai_response import (
    OpenAIResponseEndpoint,
    OpenAIResponseStreamEndpoint,
)


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
# Tests: the two first-token metrics
# ---------------------------------------------------------------------------


class TestFirstTokenMetrics:
    """`time_to_first_token` covers any delta; `time_to_first_content_token` only text."""

    @patch("llmeter.endpoints.openai_response.OpenAI")
    @patch("time.perf_counter")
    def test_reasoning_text_delta_sets_ttft_only(
        self, mock_perf_counter, mock_openai_class
    ):
        """`response.reasoning_text.delta` sets TTFT but not the content TTFT."""
        # Calls: created, reasoning_delta, text_delta, completed
        mock_perf_counter.side_effect = [100.1, 100.2, 100.5, 100.6]

        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _text_delta_event("Answer"),
            _completed_event(reasoning_tokens=10),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.5)
        assert response.time_to_last_token == pytest.approx(0.5)
        assert response.response_text == "Answer"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    @patch("time.perf_counter")
    def test_reasoning_summary_delta_sets_ttft_only(
        self, mock_perf_counter, mock_openai_class
    ):
        """`response.reasoning_summary_text.delta` sets TTFT but not the content TTFT."""
        mock_perf_counter.side_effect = [100.1, 100.3, 100.7, 100.8]

        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _reasoning_summary_delta_event(),
            _text_delta_event("Result"),
            _completed_event(reasoning_tokens=3),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.3)
        assert response.time_to_first_content_token == pytest.approx(0.7)
        assert response.response_text == "Result"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_ttft_not_overwritten_by_later_deltas(self, mock_openai_class):
        """Only the first reasoning delta sets TTFT; only the first text delta sets content TTFT."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _reasoning_text_delta_event(),
            _text_delta_event("Fin"),
            _text_delta_event("al"),
            _completed_event(),
        ]

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.1, 100.2, 100.3, 100.5, 100.6, 100.7]
            endpoint.process_raw_response(iter(events), 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.5)
        assert response.response_text == "Final"

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_metrics_equal_without_reasoning(self, mock_openai_class):
        """With no reasoning deltas the two metrics describe the same token."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _text_delta_event("Hello"),
            _completed_event(),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token == response.time_to_first_content_token

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_reasoning_only_leaves_content_ttft_unset(self, mock_openai_class):
        """A reasoning-only stream has no content token to time."""
        endpoint = OpenAIResponseStreamEndpoint(model_id="test-model")
        events = [
            _created_event(),
            _reasoning_text_delta_event(),
            _completed_event(reasoning_tokens=5),
        ]

        response = _make_draft_response()
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_content_token is None
        assert response.response_text is None

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
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.response_text == "Answer"


# ---------------------------------------------------------------------------
# Tests: deprecated constructor argument
# ---------------------------------------------------------------------------


class TestDeprecatedTtftFlag:
    @patch("llmeter.endpoints.openai_response.OpenAI")
    @pytest.mark.parametrize("value", [True, False])
    def test_passing_flag_warns(self, mock_openai_class, value):
        with pytest.warns(DeprecationWarning, match="ttft_visible_tokens_only"):
            OpenAIResponseStreamEndpoint(
                model_id="test-model", ttft_visible_tokens_only=value
            )

    @patch("llmeter.endpoints.openai_response.OpenAI")
    def test_omitting_flag_does_not_warn(self, mock_openai_class, recwarn):
        OpenAIResponseStreamEndpoint(model_id="test-model")
        assert [w for w in recwarn if w.category is DeprecationWarning] == []


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
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

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
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

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
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

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
        endpoint.process_raw_response(iter(events), time.perf_counter(), response)

        assert response.num_tokens_output_reasoning == 30
        assert response.num_tokens_input_cached == 60
        assert response.num_tokens_input == 100
        assert response.num_tokens_output == 50


# ---------------------------------------------------------------------------
# Tests: reasoning_type resolution, across both Responses endpoints
# ---------------------------------------------------------------------------


def _reasoning_item(content=None, summary=None):
    return SimpleNamespace(type="reasoning", content=content, summary=summary)


def _sync_response(output_items):
    return SimpleNamespace(
        id="resp_1", output_text="Answer", output=output_items, usage=None
    )


def _stream_for_shape(shape: str):
    events = {
        "verbatim": [_reasoning_text_delta_event()],
        "summary": [_reasoning_summary_delta_event()],
        # Both present: the summary arrived first and set TTFT, so it is not safe to claim verbatim
        "mixed": [_reasoning_summary_delta_event(), _reasoning_text_delta_event()],
        "none": [],
    }[shape]
    return iter(
        [_created_event(), *events, _text_delta_event("Answer"), _completed_event()]
    )


def _sync_for_shape(shape: str):
    items = {
        "verbatim": [_reasoning_item(content=[SimpleNamespace(text="t")])],
        "summary": [_reasoning_item(summary=[SimpleNamespace(text="s")])],
        "mixed": [
            _reasoning_item(summary=[SimpleNamespace(text="s")]),
            _reasoning_item(content=[SimpleNamespace(text="t")]),
        ],
        "none": [SimpleNamespace(type="message")],
    }[shape]
    return _sync_response(items)


_MODES = {
    "streaming": (OpenAIResponseStreamEndpoint, _stream_for_shape),
    "non-streaming": (OpenAIResponseEndpoint, _sync_for_shape),
}
"""
Transports to resolve each reasoning shape through

The Responses API states the disclosure level in the response itself - as event types when streaming,
and as reasoning-item fields when not - so this endpoint needs no declared default. Parametrising
over the transport keeps the two readings in agreement for *every* shape, mixed included.
"""


def _resolve(mode: str, shape: str):
    endpoint_cls, build = _MODES[mode]
    endpoint = endpoint_cls(model_id="test-model")
    response = _make_draft_response()
    endpoint.process_raw_response(build(shape), time.perf_counter(), response)
    return response


@patch("llmeter.endpoints.openai_response.OpenAI")
class TestResponsesReasoningTypeResolution:
    """Disclosure level is fully observable here, so no `default_reasoning_visibility` exists."""

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "shape,expected",
        [
            ("verbatim", "verbatim"),
            ("summary", "summary"),
            # Mixed disclosure degrades to the more conservative level, on both transports
            ("mixed", "summary"),
            # No reasoning at all must stay unset
            ("none", None),
        ],
    )
    def test_resolution_by_content_shape(self, mock_openai, mode, shape, expected):
        assert _resolve(mode, shape).reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    def test_mixed_disclosure_resolves_to_summary(self, mock_openai, mode):
        """Both transports downgrade, so the same disclosure never reads differently.

        Claiming `"verbatim"` would let the Runner pair the whole measured window against the full
        output token count, when part of the reasoning only ever reached us summarized. Non-streaming
        has no timings to protect, but reporting a *different* level for identical content would make
        the two transports incomparable - and `BedrockConverse` sets the precedent, preferring
        `"redacted"` over readable reasoning despite also having no timings.
        """
        assert _resolve(mode, "mixed").reasoning_type == "summary"

    @pytest.mark.parametrize(
        "items,why",
        [
            (
                [
                    _reasoning_item(summary=[SimpleNamespace(text="s")]),
                    _reasoning_item(content=[SimpleNamespace(text="t")]),
                ],
                "summary first",
            ),
            (
                [
                    _reasoning_item(content=[SimpleNamespace(text="t")]),
                    _reasoning_item(summary=[SimpleNamespace(text="s")]),
                ],
                "raw content first",
            ),
        ],
    )
    def test_non_streaming_result_is_independent_of_item_order(
        self, mock_openai, items, why
    ):
        """Regression: the old code broke out of the loop on the first `content` item.

        That made the outcome depend on which item happened to come first, so a response listing raw
        reasoning before its summary never saw the summary at all.
        """
        endpoint = OpenAIResponseEndpoint(model_id="test-model")
        response = _make_draft_response()
        endpoint.process_raw_response(
            _sync_response(items), time.perf_counter(), response
        )

        assert response.reasoning_type == "summary", why

    @pytest.mark.parametrize(
        "items,why",
        [
            (
                [
                    _reasoning_item(),
                    _reasoning_item(summary=[SimpleNamespace(text="s")]),
                ],
                "withheld item before a summarized one",
            ),
            (
                [
                    _reasoning_item(summary=[SimpleNamespace(text="s")]),
                    _reasoning_item(),
                ],
                "summarized item before a withheld one",
            ),
            (
                [
                    _reasoning_item(),
                    _reasoning_item(content=[SimpleNamespace(text="t")]),
                ],
                "withheld item alongside raw content",
            ),
        ],
    )
    def test_redaction_outranks_other_levels(self, mock_openai, items, why):
        """Matches the partial-redaction rule in the Bedrock and Anthropic connectors.

        Previously a `summary` item overwrote `"redacted"` unconditionally, so whether redaction was
        reported depended on item order.
        """
        endpoint = OpenAIResponseEndpoint(model_id="test-model")
        response = _make_draft_response()
        endpoint.process_raw_response(
            _sync_response(items), time.perf_counter(), response
        )

        assert response.reasoning_type == "redacted", why

    @pytest.mark.parametrize("mode", list(_MODES))
    def test_reasoning_never_leaks_into_response_text(self, mock_openai, mode):
        for shape in ("verbatim", "summary", "mixed", "none"):
            assert _resolve(mode, shape).response_text == "Answer", shape

    def test_non_streaming_reasoning_item_disclosing_nothing_is_redacted(
        self, mock_openai
    ):
        """A reasoning item with neither raw content nor a summary withheld both."""
        endpoint = OpenAIResponseEndpoint(model_id="test-model")
        response = _make_draft_response()
        endpoint.process_raw_response(
            _sync_response([_reasoning_item()]), time.perf_counter(), response
        )
        assert response.reasoning_type == "redacted"

    def test_non_streaming_records_no_first_token_metrics(self, mock_openai):
        response = _resolve("non-streaming", "verbatim")
        assert response.time_to_first_token is None
        assert response.time_to_first_content_token is None

    def test_non_list_output_is_tolerated(self, mock_openai):
        """A placeholder/mock `output` must not be iterated as if it were a list."""
        endpoint = OpenAIResponseEndpoint(model_id="test-model")
        response = _make_draft_response()
        endpoint.process_raw_response(
            _sync_response(Mock()), time.perf_counter(), response
        )
        assert response.error is None
        assert response.reasoning_type is None


# ---------------------------------------------------------------------------
# Tests: reasoning that is billed but never disclosed
# ---------------------------------------------------------------------------


def _sync_response_with_usage(output_items, reasoning_tokens=None):
    usage = Mock()
    usage.input_tokens = 10
    usage.output_tokens = 20
    usage.input_tokens_details = None
    if reasoning_tokens is not None:
        output_details = Mock()
        output_details.reasoning_tokens = reasoning_tokens
        usage.output_tokens_details = output_details
    else:
        usage.output_tokens_details = None
    return SimpleNamespace(
        id="resp_1", output_text="Answer", output=output_items, usage=usage
    )


@patch("llmeter.endpoints.openai_response.OpenAI")
class TestResponsesHiddenReasoning:
    """Reasoning events only arrive if the request asked for a summary.

    `reasoning={"effort": ...}` without `"summary"` produces no reasoning events and no reasoning
    items -- just a `reasoning_tokens` figure in usage. That must not be left as `None` (which
    asserts the model did not reason) or TPOT would divide a post-reasoning window by a
    reasoning-inclusive token count.

    This connector resolves it to `"redacted"` rather than `"unknown"`: the Responses schema states
    disclosure structurally, so the absence of every reasoning signal *is* withholding rather than a
    parsing gap. It also keeps the endpoint self-consistent with a reasoning item that discloses
    neither content nor summary, which already yields `"redacted"`.
    """

    def _invoke_stream(self, events):
        endpoint = OpenAIResponseStreamEndpoint(model_id="o3")
        with patch.object(endpoint._client.responses, "create") as create:
            create.return_value = iter(events)
            return endpoint.invoke({"input": "Hi"})

    def _invoke_sync(self, raw):
        endpoint = OpenAIResponseEndpoint(model_id="o3")
        with patch.object(endpoint._client.responses, "create") as create:
            create.return_value = raw
            return endpoint.invoke({"input": "Hi"})

    def test_streaming_resolves_redacted_from_token_count(self, mock_openai):
        response = self._invoke_stream(
            [
                _created_event(),
                _text_delta_event("Answer"),
                _completed_event(reasoning_tokens=9),
            ]
        )

        assert response.num_tokens_output_reasoning == 9
        assert response.reasoning_type == "redacted"

    def test_streaming_stays_unset_without_reasoning_tokens(self, mock_openai):
        response = self._invoke_stream(
            [
                _created_event(),
                _text_delta_event("Answer"),
                _completed_event(reasoning_tokens=0),
            ]
        )

        assert response.reasoning_type is None

    @pytest.mark.parametrize(
        "reasoning_event,expected",
        [
            (_reasoning_text_delta_event, "verbatim"),
            (_reasoning_summary_delta_event, "summary"),
        ],
    )
    def test_streamed_disclosure_still_wins(
        self, mock_openai, reasoning_event, expected
    ):
        """An observed event type is precise; the token-count fallback must not overwrite it."""
        response = self._invoke_stream(
            [
                _created_event(),
                reasoning_event(),
                _text_delta_event("Answer"),
                _completed_event(reasoning_tokens=9),
            ]
        )

        assert response.reasoning_type == expected

    def test_non_streaming_resolves_redacted_from_token_count(self, mock_openai):
        response = self._invoke_sync(
            _sync_response_with_usage(
                [SimpleNamespace(type="message")], reasoning_tokens=9
            )
        )

        assert response.reasoning_type == "redacted"

    def test_no_reasoning_item_matches_an_empty_reasoning_item(self, mock_openai):
        """The two describe the same physical situation, so they must not disagree."""
        no_item = self._invoke_sync(
            _sync_response_with_usage(
                [SimpleNamespace(type="message")], reasoning_tokens=9
            )
        )
        empty_item = self._invoke_sync(
            _sync_response_with_usage([_reasoning_item()], reasoning_tokens=9)
        )

        assert no_item.reasoning_type == empty_item.reasoning_type == "redacted"

    def test_non_streaming_reasoning_item_still_wins(self, mock_openai):
        """A reasoning item present but disclosing nothing is *observed* redaction."""
        response = self._invoke_sync(
            _sync_response_with_usage([_reasoning_item()], reasoning_tokens=9)
        )

        assert response.reasoning_type == "redacted"
