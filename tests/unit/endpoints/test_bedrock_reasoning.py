# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reasoning token parsing in BedrockConverseStream.

Covers:
- ``time_to_first_token`` is set by the first delta of any kind, including
  ``reasoningContent`` deltas.
- ``time_to_first_content_token`` is set only by the first visible ``text`` delta.
- Reasoning deltas never contribute to ``response_text``.
- The retired ``ttft_visible_tokens_only`` argument warns and is ignored.
"""

import time
from unittest.mock import patch

import pytest

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.bedrock import BedrockConverse, BedrockConverseStream


def _make_draft_response() -> InvocationResponse:
    return InvocationResponse(response_text=None, id=None)


def _stream_response(stream_chunks: list[dict]) -> dict:
    """Wrap stream chunks in the Bedrock ConverseStream response envelope."""
    return {
        "stream": stream_chunks,
        "ResponseMetadata": {"RequestId": "req-123", "RetryAttempts": 0},
    }


def _reasoning_delta(text: str = "thinking...") -> dict:
    return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": text}}}}


def _text_delta(text: str) -> dict:
    return {"contentBlockDelta": {"delta": {"text": text}}}


_REDACTED_DELTA = {
    "contentBlockDelta": {"delta": {"reasoningContent": {"redactedContent": b"enc"}}}
}

_USAGE_CHUNK = {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}}


# ---------------------------------------------------------------------------
# Tests: the two first-token metrics
# ---------------------------------------------------------------------------


class TestFirstTokenMetrics:
    """`time_to_first_token` covers any token; `time_to_first_content_token` only visible ones."""

    @patch("time.perf_counter")
    def test_reasoning_sets_ttft_and_text_sets_content_ttft(self, mock_perf_counter):
        """The two metrics must resolve to the reasoning and text deltas respectively."""
        # Calls: reasoning_delta, text_delta, contentBlockStop, metadata
        mock_perf_counter.side_effect = [100.2, 100.5, 100.6, 100.7]

        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _reasoning_delta(),
                _text_delta("Answer"),
                {"contentBlockStop": {}},
                _USAGE_CHUNK,
            ]
        )

        response = _make_draft_response()
        endpoint.process_raw_response(raw, 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.5)
        assert response.time_to_last_token == pytest.approx(0.6)

    def test_multiple_reasoning_deltas_do_not_overwrite_ttft(self):
        """Only the *first* reasoning delta sets TTFT."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _reasoning_delta("step 1"),
                _reasoning_delta("step 2"),
                _text_delta("Answer"),
                {"contentBlockStop": {}},
                _USAGE_CHUNK,
            ]
        )

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.2, 100.3, 100.5, 100.6, 100.7]
            endpoint.process_raw_response(raw, 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.2)
        assert response.time_to_first_content_token == pytest.approx(0.5)

    def test_multiple_text_deltas_do_not_overwrite_content_ttft(self):
        """Only the *first* text delta sets the content TTFT."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _text_delta("Hello"),
                _text_delta(" world"),
                {"contentBlockStop": {}},
                _USAGE_CHUNK,
            ]
        )

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.4, 100.5, 100.6, 100.7]
            endpoint.process_raw_response(raw, 100.0, response)

        assert response.time_to_first_content_token == pytest.approx(0.4)
        assert response.response_text == "Hello world"

    def test_metrics_equal_without_reasoning(self):
        """With no reasoning content the two metrics describe the same token."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [_text_delta("Hello"), {"contentBlockStop": {}}, _USAGE_CHUNK]
        )

        response = _make_draft_response()
        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token == response.time_to_first_content_token

    def test_empty_text_delta_does_not_set_timings(self):
        """An empty text delta is not a token."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [_text_delta(""), _text_delta("Hi"), {"contentBlockStop": {}}, _USAGE_CHUNK]
        )

        response = _make_draft_response()
        with patch("time.perf_counter") as clock:
            clock.side_effect = [100.2, 100.5, 100.6, 100.7]
            endpoint.process_raw_response(raw, 100.0, response)

        assert response.time_to_first_token == pytest.approx(0.5)
        assert response.time_to_first_content_token == pytest.approx(0.5)

    def test_only_reasoning_no_text(self):
        """Reasoning-only stream: TTFT is set, but there is no content token to time."""
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _reasoning_delta(),
                {"contentBlockStop": {}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 0}}},
            ]
        )

        response = _make_draft_response()
        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_content_token is None
        assert response.response_text is None


# ---------------------------------------------------------------------------
# Tests: response_text must never include reasoning
# ---------------------------------------------------------------------------


class TestResponseText:
    def test_reasoning_deltas_excluded_from_response_text(self):
        endpoint = BedrockConverseStream(model_id="test-model")
        raw = _stream_response(
            [
                _reasoning_delta("internal thought"),
                _text_delta("Visible"),
                _text_delta(" answer"),
                {"contentBlockStop": {}},
                _USAGE_CHUNK,
            ]
        )

        response = _make_draft_response()
        endpoint.process_raw_response(raw, time.perf_counter(), response)

        assert response.response_text == "Visible answer"
        assert "internal thought" not in (response.response_text or "")


# ---------------------------------------------------------------------------
# Tests: deprecated constructor argument
# ---------------------------------------------------------------------------


class TestDeprecatedTtftFlag:
    @pytest.mark.parametrize("value", [True, False])
    def test_passing_flag_warns(self, value):
        with pytest.warns(DeprecationWarning, match="ttft_visible_tokens_only"):
            BedrockConverseStream(model_id="test-model", ttft_visible_tokens_only=value)

    def test_omitting_flag_does_not_warn(self, recwarn):
        BedrockConverseStream(model_id="test-model")
        assert [w for w in recwarn if w.category is DeprecationWarning] == []

    def test_flag_does_not_change_behaviour(self):
        """Both metrics are recorded regardless of the retired flag's value."""
        raw_chunks = [
            _reasoning_delta(),
            _text_delta("Answer"),
            {"contentBlockStop": {}},
            _USAGE_CHUNK,
        ]

        results = []
        for value in (True, False):
            with pytest.warns(DeprecationWarning):
                endpoint = BedrockConverseStream(
                    model_id="test-model", ttft_visible_tokens_only=value
                )
            response = _make_draft_response()
            with patch("time.perf_counter") as clock:
                clock.side_effect = [100.2, 100.5, 100.6, 100.7]
                endpoint.process_raw_response(
                    _stream_response(raw_chunks), 100.0, response
                )
            results.append(
                (response.time_to_first_token, response.time_to_first_content_token)
            )

        assert results[0] == results[1]
        assert results[0] == (pytest.approx(0.2), pytest.approx(0.5))


# ---------------------------------------------------------------------------
# Tests: reasoning_type resolution, across both Converse endpoints
# ---------------------------------------------------------------------------

#: The reasoning shapes a Converse response can carry, expressed once per transport. Parametrising
#: over the mode is the point: streaming and non-streaming must resolve identical content to the
#: same `reasoning_type`, and writing them as separate suites let that parity drift unchecked.
_REASONING_SHAPES = ("readable", "redacted", "partial", "none")


def _stream_for_shape(shape: str) -> dict:
    deltas = {
        "readable": [_reasoning_delta("thinking")],
        "redacted": [_REDACTED_DELTA],
        "partial": [_reasoning_delta("thinking"), _REDACTED_DELTA],
        "none": [],
    }[shape]
    return _stream_response([*deltas, _text_delta("Answer"), _USAGE_CHUNK])


def _sync_for_shape(shape: str) -> dict:
    parts = {
        "readable": [{"reasoningContent": {"reasoningText": {"text": "thinking"}}}],
        "redacted": [{"reasoningContent": {"redactedContent": b"enc"}}],
        "partial": [
            {"reasoningContent": {"reasoningText": {"text": "thinking"}}},
            {"reasoningContent": {"redactedContent": b"enc"}},
        ],
        "none": [],
    }[shape]
    return {
        "output": {"message": {"content": [*parts, {"text": "Answer"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 20},
        "ResponseMetadata": {"RequestId": "req-1", "RetryAttempts": 0},
    }


_MODES = {
    "streaming": (BedrockConverseStream, _stream_for_shape),
    "non-streaming": (BedrockConverse, _sync_for_shape),
}


def _resolve(mode: str, shape: str, model_id: str, declared=None):
    """Run the given endpoint over a response carrying `shape`, and return the parsed response."""
    endpoint_cls, build = _MODES[mode]
    endpoint = endpoint_cls(model_id=model_id, default_reasoning_visibility=declared)
    response = _make_draft_response()
    endpoint.process_raw_response(build(shape), time.perf_counter(), response)
    return response


class TestReasoningTypeResolution:
    """How `reasoning_type` is resolved, for both Converse endpoints.

    Converse cannot distinguish verbatim from summarized reasoning structurally, so readable
    reasoning falls back to the declared or model-inferred visibility. Redaction *is* observable and
    takes precedence - including alongside readable reasoning, since partial redaction still means
    some reasoning was not delivered as plain text.
    """

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "shape,expected",
        [
            # Readable reasoning: not observable which kind, so use the inferred default
            ("readable", "summary"),
            # Observable, and overrides the default
            ("redacted", "redacted"),
            # Partial redaction: conservative, because some reasoning was not plain text
            ("partial", "redacted"),
            # No reasoning at all must stay unset, *not* take the endpoint's default
            ("none", None),
        ],
    )
    def test_resolution_for_anthropic_model(self, mode, shape, expected):
        response = _resolve(mode, shape, "anthropic.claude-opus-4-7")
        assert response.reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("anthropic.claude-opus-4-7", "summary"),
            ("openai.gpt-oss-120b-1:0", "verbatim"),
        ],
    )
    def test_readable_reasoning_uses_model_inference(self, mode, model_id, expected):
        response = _resolve(mode, "readable", model_id)
        assert response.reasoning_type == expected

    @pytest.mark.parametrize("mode", list(_MODES))
    def test_declared_visibility_overrides_inference(self, mode):
        """Needed for Claude models before v4, which stream their full thinking output."""
        response = _resolve(
            mode, "readable", "anthropic.claude-3-7-sonnet", declared="verbatim"
        )
        assert response.reasoning_type == "verbatim"

    @pytest.mark.parametrize("mode", list(_MODES))
    def test_observable_redaction_beats_declared_visibility(self, mode):
        response = _resolve(
            mode, "redacted", "openai.gpt-oss-120b-1:0", declared="verbatim"
        )
        assert response.reasoning_type == "redacted"

    @pytest.mark.parametrize("mode", list(_MODES))
    def test_reasoning_never_leaks_into_response_text(self, mode):
        for shape in _REASONING_SHAPES:
            response = _resolve(mode, shape, "anthropic.claude-opus-4-7")
            assert response.response_text == "Answer", f"leaked for shape={shape!r}"

    def test_non_streaming_records_no_first_token_metrics(self):
        """Resolution must not accidentally start populating timings on a sync response."""
        response = _resolve("non-streaming", "readable", "anthropic.claude-opus-4-7")
        assert response.time_to_first_token is None
        assert response.time_to_first_content_token is None
