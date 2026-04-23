# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for InvocationResponse.to_dict() with its downstream consumers.

These tests exercise the real pipeline:
    InvocationResponse.to_dict() → RunningStats.update() → to_stats()

They ensure that any changes to to_dict() serialization remain compatible
with RunningStats (which compares and subtracts request_time values) and
with json.dumps() (for Lambda marshaling and other plain-JSON code paths).

See: https://github.com/awslabs/llmeter/issues/67
"""

import json
from datetime import datetime, timezone

import pytest

from llmeter.endpoints.base import InvocationResponse
from llmeter.utils import RunningStats


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_response(request_time: datetime, **kwargs) -> InvocationResponse:
    """Create a minimal InvocationResponse with the given request_time."""
    defaults = dict(
        id="test",
        response_text="hello",
        time_to_first_token=0.1,
        time_to_last_token=0.5,
        num_tokens_input=10,
        num_tokens_output=5,
    )
    defaults.update(kwargs)
    return InvocationResponse(request_time=request_time, **defaults)


# ── to_dict() → RunningStats.update() integration ───────────────────────────


class TestToDictRunningStatsIntegration:
    """Verify that InvocationResponse.to_dict() output is compatible with
    RunningStats.update() and the full stats pipeline."""

    def test_single_response_through_pipeline(self):
        """A single response.to_dict() fed into RunningStats should not raise."""
        t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        response = _make_response(request_time=t)

        rs = RunningStats(metrics=["time_to_first_token", "time_to_last_token"])
        rs.update(response.to_dict())

        stats = rs.to_stats()
        assert stats["failed_requests"] == 0

    def test_multiple_responses_through_pipeline(self):
        """Multiple response.to_dict() calls fed into RunningStats must allow
        request_time comparisons without TypeError."""
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)

        rs = RunningStats(
            metrics=["time_to_first_token", "time_to_last_token", "num_tokens_output"]
        )
        for t in (t1, t2, t3):
            rs.update(_make_response(request_time=t).to_dict())

        stats = rs.to_stats()
        assert stats["failed_requests"] == 0
        # 3 requests over 10s = 18 rpm
        assert stats["requests_per_minute"] == pytest.approx(18.0)

    def test_to_stats_with_end_time_after_to_dict(self):
        """to_stats(end_time=datetime) must work when _first_send_time was
        populated via to_dict() — no mixed-type comparison."""
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)

        rs = RunningStats(metrics=["num_tokens_output"])
        rs.update(_make_response(request_time=t1, num_tokens_output=500).to_dict())
        rs.update(_make_response(request_time=t2, num_tokens_output=300).to_dict())

        stats = rs.to_stats(end_time=end)
        # 800 tokens / 10 seconds = 80 tok/s
        assert stats["output_tps"] == pytest.approx(80.0)

    def test_send_window_with_to_dict(self):
        """_send_window() must return correct seconds when fed via to_dict()."""
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)

        rs = RunningStats(metrics=[])
        rs.update(_make_response(request_time=t1).to_dict())
        rs.update(_make_response(request_time=t2).to_dict())

        assert rs._send_window() == pytest.approx(10.0)

    def test_first_and_last_send_time_are_datetime(self):
        """After feeding to_dict() output, _first_send_time and _last_send_time
        must remain datetime objects (not strings) for downstream arithmetic."""
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

        rs = RunningStats(metrics=[])
        rs.update(_make_response(request_time=t1).to_dict())
        rs.update(_make_response(request_time=t2).to_dict())

        assert isinstance(rs._first_send_time, datetime)
        assert isinstance(rs._last_send_time, datetime)
        assert rs._first_send_time == t1
        assert rs._last_send_time == t2


# ── to_dict() → json.dumps() integration ────────────────────────────────────


class TestToDictJsonIntegration:
    """Verify that InvocationResponse.to_dict() output can be serialized to
    JSON, either via to_json() or with llmeter_default_serializer."""

    def test_to_json_with_request_time(self):
        """to_json() must handle datetime request_time without error."""
        response = _make_response(
            request_time=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        serialized = response.to_json()
        parsed = json.loads(serialized)
        assert isinstance(parsed["request_time"], str)

    def test_to_json_round_trips_request_time(self):
        """The serialized request_time should parse back to the same instant."""
        dt = datetime(2026, 6, 15, 8, 30, 45, tzinfo=timezone.utc)
        response = _make_response(request_time=dt)
        parsed = json.loads(response.to_json())
        restored = datetime.fromisoformat(
            parsed["request_time"].replace("Z", "+00:00")
        )
        assert restored == dt

    def test_error_response_to_json(self):
        """Error responses with request_time must also serialize cleanly."""
        response = InvocationResponse.error_output(
            input_payload={"messages": []},
            error="Connection timeout",
            request_time=datetime.now(timezone.utc),
        )
        serialized = response.to_json()
        assert isinstance(json.loads(serialized), dict)

    def test_full_response_to_json(self):
        """A fully-populated InvocationResponse must serialize via to_json()."""
        response = InvocationResponse(
            id="resp-001",
            response_text="The answer is 42.",
            input_payload={"messages": [{"role": "user", "content": "What is 6*7?"}]},
            input_prompt="What is 6*7?",
            time_to_first_token=0.15,
            time_to_last_token=0.85,
            num_tokens_input=12,
            num_tokens_output=5,
            num_tokens_input_cached=4,
            num_tokens_output_reasoning=2,
            time_per_output_token=0.14,
            error=None,
            retries=0,
            request_time=datetime.now(timezone.utc),
        )
        serialized = response.to_json()
        assert isinstance(json.loads(serialized), dict)
