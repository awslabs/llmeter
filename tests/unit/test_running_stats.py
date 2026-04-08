# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time

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
        },
        {
            "time_to_first_token": 0.5,
            "time_to_last_token": 1.2,
            "time_per_output_token": 0.03,
            "num_tokens_input": 120,
            "num_tokens_output": 30,
            "error": None,
        },
        {
            "time_to_first_token": 0.4,
            "time_to_last_token": 1.0,
            "time_per_output_token": 0.025,
            "num_tokens_input": 110,
            "num_tokens_output": 28,
            "error": "timeout",
        },
    ]
    for r in responses:
        rs.record_send()
        rs.update(r)
    return rs


# ── record_send ──────────────────────────────────────────────────────────────


class TestRecordSend:
    def test_first_send_sets_first_time(self, rs):
        assert rs._first_send_time is None
        rs.record_send()
        assert rs._first_send_time is not None
        assert rs._last_send_time is not None
        assert rs._sends == 1

    def test_subsequent_sends_update_last_time(self, rs):
        rs.record_send()
        first = rs._first_send_time
        time.sleep(0.01)
        rs.record_send()
        assert rs._first_send_time == first
        assert rs._last_send_time > first
        assert rs._sends == 2

    def test_send_count_increments(self, rs):
        for _ in range(5):
            rs.record_send()
        assert rs._sends == 5


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
        stats = populated_rs.to_stats(
            total_requests=3,
            total_test_time=10.0,
            result_dict={"model_id": "test"},
        )
        assert stats["model_id"] == "test"
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
        rs._first_send_time = 100.0
        rs._last_send_time = 110.0  # 10 second window
        rs.update({"error": None})
        rs.update({"error": None})
        rs.update({"error": None})
        stats = rs.to_stats()
        # 3 responses / 10 seconds * 60 = 18.0 rpm
        assert stats["rpm"] == pytest.approx(18.0)

    def test_output_tps_uses_send_window(self, rs):
        rs._first_send_time = 100.0
        rs._last_send_time = 110.0  # 10 second window
        rs.update({"num_tokens_output": 500, "error": None})
        rs.update({"num_tokens_output": 300, "error": None})
        stats = rs.to_stats()
        # 800 tokens / 10 seconds = 80.0 tok/s
        assert stats["output_tps"] == pytest.approx(80.0)

    def test_no_send_window_when_single_send(self, rs):
        """With only one send, first == last, no window to compute RPM."""
        rs._first_send_time = 100.0
        rs._last_send_time = 100.0
        rs.update({"error": None})
        stats = rs.to_stats()
        assert "rpm" not in stats
        assert "output_tps" not in stats

    def test_no_send_window_when_no_sends(self, rs):
        stats = rs.to_stats()
        assert "rpm" not in stats
        assert "output_tps" not in stats

    def test_send_window_helper(self, rs):
        assert rs._send_window() is None
        rs._first_send_time = 10.0
        rs._last_send_time = 10.0
        assert rs._send_window() is None
        rs._last_send_time = 20.0
        assert rs._send_window() == pytest.approx(10.0)
