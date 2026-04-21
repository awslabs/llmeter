# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest

from llmeter.utils import RunningStats


@pytest.fixture
def rs():
    return RunningStats(
        metrics=[
            "time_to_first_token",
            "time_to_last_token",
            "time_per_output_token",
            "num_tokens_input",
            "num_tokens_output",
        ]
    )


@pytest.fixture
def populated_rs(rs):
    """A RunningStats with 3 responses recorded."""
    responses = [
        {
            "time_to_first_token": 0.3,
            "time_to_last_token": 0.8,
            "time_per_output_token": 0.02,
            "num_tokens_input": 100,
            "num_tokens_output": 25,
            "error": None,
            "request_time": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        },
        {
            "time_to_first_token": 0.5,
            "time_to_last_token": 1.2,
            "time_per_output_token": 0.03,
            "num_tokens_input": 120,
            "num_tokens_output": 30,
            "error": None,
            "request_time": datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        },
        {
            "time_to_first_token": 0.4,
            "time_to_last_token": 1.0,
            "time_per_output_token": 0.025,
            "num_tokens_input": 110,
            "num_tokens_output": 28,
            "error": "timeout",
            "request_time": datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc),
        },
    ]
    for r in responses:
        rs.update(r)
    return rs


# ── request_time tracking ────────────────────────────────────────────────────


class TestRequestTimeTracking:
    def test_first_update_sets_first_send_time(self, rs):
        assert rs._first_send_time is None
        t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rs.update({"time_to_first_token": 0.3, "error": None, "request_time": t})
        assert rs._first_send_time == t
        assert rs._last_send_time == t

    def test_subsequent_updates_track_min_max(self, rs):
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
        rs.update({"error": None, "request_time": t1})
        rs.update({"error": None, "request_time": t2})
        assert rs._first_send_time == t1
        assert rs._last_send_time == t2

    def test_out_of_order_timestamps(self, rs):
        t1 = datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rs.update({"error": None, "request_time": t1})
        rs.update({"error": None, "request_time": t2})
        assert rs._first_send_time == t2  # min
        assert rs._last_send_time == t1  # max


# ── update ───────────────────────────────────────────────────────────────────


class TestUpdate:
    def test_count_increments(self, rs):
        rs.update({"time_to_first_token": 0.3, "error": None})
        assert rs._count == 1
        rs.update({"time_to_first_token": 0.5, "error": None})
        assert rs._count == 2

    def test_failed_count(self, rs):
        rs.update({"error": "timeout"})
        rs.update({"error": None})
        rs.update({"error": "connection refused"})
        assert rs._failed == 2

    def test_none_values_skipped(self, rs):
        rs.update({"time_to_first_token": None, "error": None})
        assert len(rs._values["time_to_first_token"]) == 0

    def test_nan_values_skipped(self, rs):
        rs.update({"time_to_first_token": float("nan"), "error": None})
        assert len(rs._values["time_to_first_token"]) == 0

    def test_sums_accumulated(self, rs):
        rs.update({"num_tokens_output": 10, "error": None})
        rs.update({"num_tokens_output": 20, "error": None})
        assert rs._sums["num_tokens_output"] == 30

    def test_values_sorted(self, rs):
        rs.update({"time_to_first_token": 0.5, "error": None})
        rs.update({"time_to_first_token": 0.1, "error": None})
        rs.update({"time_to_first_token": 0.3, "error": None})
        assert rs._values["time_to_first_token"] == [0.1, 0.3, 0.5]


# ── to_stats ─────────────────────────────────────────────────────────────────


class TestToStats:
    def test_basic_stats(self, populated_rs):
        stats = populated_rs.to_stats()
        assert stats["failed_requests"] == 1
        assert "time_to_first_token-p50" in stats
        assert "time_to_last_token-average" in stats
        assert "num_tokens_output-p90" in stats

    def test_with_run_context(self, populated_rs):
        end_time = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        stats = populated_rs.to_stats(
            end_time=end_time,
            result_dict={"model_id": "test"},
        )
        assert stats["model_id"] == "test"
        # 3 responses over 10 second send window = 18 rpm
        assert stats["requests_per_minute"] == pytest.approx(18.0)
        assert stats["failed_requests_rate"] == pytest.approx(1 / 3)
        assert stats["total_output_tokens"] == 83

    def test_without_run_context(self, populated_rs):
        stats = populated_rs.to_stats()
        assert stats["failed_requests"] == 1
        assert stats["total_input_tokens"] == 330
        assert stats["total_output_tokens"] == 83

    def test_empty_stats(self, rs):
        stats = rs.to_stats()
        assert stats["failed_requests"] == 0
        assert stats["total_input_tokens"] == 0


# ── send-window throughput in to_stats ────────────────────────────────────────


class TestSendWindowStats:
    def test_rpm_uses_send_window(self, rs):
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)  # 10 second window
        rs.update({"error": None, "request_time": t1})
        rs.update(
            {
                "error": None,
                "request_time": datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
            }
        )
        rs.update({"error": None, "request_time": t2})
        stats = rs.to_stats()
        # 3 responses / 10 seconds * 60 = 18.0 rpm
        assert stats["requests_per_minute"] == pytest.approx(18.0)

    def test_output_tps_uses_end_time(self, rs):
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)  # 10 second window
        end = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        rs.update({"num_tokens_output": 500, "error": None, "request_time": t1})
        rs.update({"num_tokens_output": 300, "error": None, "request_time": t2})
        stats = rs.to_stats(end_time=end)
        # 800 tokens / 10 seconds = 80.0 tok/s
        assert stats["output_tps"] == pytest.approx(80.0)

    def test_no_send_window_when_single_request(self, rs):
        """With only one request, first == last, no window to compute RPM."""
        t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rs.update({"error": None, "request_time": t})
        stats = rs.to_stats()
        assert "requests_per_minute" not in stats
        assert "output_tps" not in stats

    def test_no_send_window_when_no_requests(self, rs):
        stats = rs.to_stats()
        assert "requests_per_minute" not in stats
        assert "output_tps" not in stats

    def test_send_window_helper(self, rs):
        assert rs._send_window() is None
        t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rs._first_send_time = t1
        rs._last_send_time = t1
        assert rs._send_window() is None
        rs._last_send_time = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        assert rs._send_window() == pytest.approx(10.0)
