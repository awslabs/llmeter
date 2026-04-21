# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from unittest.mock import patch

from llmeter.live_display import (
    DEFAULT_DISPLAY_STATS,
    LiveStatsDisplay,
    _classify,
    _format_stat,
    _group_stats,
    _in_notebook,
)

# ── _classify ────────────────────────────────────────────────────────────────


class TestClassify:
    def test_rpm_goes_to_throughput(self):
        assert _classify("rpm") == "Throughput"

    def test_tps_goes_to_throughput(self):
        assert _classify("p50_tps") == "Throughput"
        assert _classify("output_tps") == "Throughput"

    def test_ttft_goes_to_ttft(self):
        assert _classify("p50_ttft") == "TTFT"
        assert _classify("p90_ttft") == "TTFT"

    def test_ttlt_goes_to_ttlt(self):
        assert _classify("p50_ttlt") == "TTLT"
        assert _classify("p90_ttlt") == "TTLT"

    def test_token_goes_to_tokens(self):
        assert _classify("input_tokens") == "Tokens"
        assert _classify("output_tokens") == "Tokens"

    def test_fail_goes_to_errors(self):
        assert _classify("fail") == "Errors"

    def test_unknown_goes_to_other(self):
        assert _classify("custom_metric") == "Other"

    def test_case_insensitive(self):
        assert _classify("RPM") == "Throughput"
        assert _classify("P50_TTFT") == "TTFT"


# ── _group_stats ─────────────────────────────────────────────────────────────


class TestGroupStats:
    def test_groups_by_category(self):
        stats = {
            "rpm": "185.9",
            "p50_ttft": "0.312s",
            "p90_ttlt": "1.203s",
            "input_tokens": "12540",
            "fail": "0",
        }
        groups = _group_stats(stats)
        assert "Throughput" in groups
        assert "TTFT" in groups
        assert "TTLT" in groups
        assert "Tokens" in groups
        assert "Errors" in groups

    def test_preserves_order(self):
        stats = OrderedDict(
            [
                ("rpm", "185.9"),
                ("p50_ttft", "0.312s"),
                ("p50_ttlt", "0.847s"),
                ("fail", "0"),
            ]
        )
        groups = _group_stats(stats)
        group_names = list(groups.keys())
        assert group_names == ["Throughput", "TTFT", "TTLT", "Errors"]

    def test_unknown_keys_go_to_other(self):
        stats = {"custom": "42"}
        groups = _group_stats(stats)
        assert "Other" in groups
        assert groups["Other"] == [("custom", "42")]

    def test_empty_stats(self):
        groups = _group_stats({})
        assert len(groups) == 0


# ── _in_notebook ─────────────────────────────────────────────────────────────


class TestInNotebook:
    def test_returns_false_outside_notebook(self):
        assert _in_notebook() is False

    def test_returns_false_for_terminal_ipython(self):
        mock_shell = type("TerminalInteractiveShell", (), {})()
        with patch("IPython.get_ipython", return_value=mock_shell):
            assert _in_notebook() is False

    def test_returns_true_for_zmq_shell(self):
        mock_shell = type("ZMQInteractiveShell", (), {})()
        with patch("IPython.get_ipython", return_value=mock_shell):
            assert _in_notebook() is True

    def test_returns_false_for_none(self):
        with patch("IPython.get_ipython", return_value=None):
            assert _in_notebook() is False


# ── _format_stat ─────────────────────────────────────────────────────────────


class TestFormatStat:
    def test_time_metric(self):
        assert _format_stat("time_to_first_token-p50", 0.312) == "0.312s"

    def test_rpm_metric(self):
        assert _format_stat("rpm", 185.9) == "185.9"

    def test_tps_metric(self):
        assert _format_stat("output_tps", 80.0) == "80.0 tok/s"

    def test_inverse(self):
        result = _format_stat("time_per_output_token-p50", 0.04, invert=True)
        assert "tok/s" in result
        assert "25.0" in result

    def test_integer_value(self):
        assert _format_stat("failed_requests", 3) == "3"

    def test_float_whole_number(self):
        assert _format_stat("failed_requests", 0.0) == "0.0"


# ── LiveStatsDisplay ─────────────────────────────────────────────────────────


class TestLiveStatsDisplay:
    def test_disabled_does_nothing(self):
        display = LiveStatsDisplay(disabled=True)
        display.update({"rpm": 100})
        display.close()

    def test_format_stats_with_empty_raw(self):
        display = LiveStatsDisplay()
        result = display.format_stats({})
        assert all(v == "—" for v in result.values())
        assert "rpm" in result
        assert "fail" in result

    def test_format_stats_with_data(self):
        display = LiveStatsDisplay(
            display_stats={
                "rpm": "rpm",
                "fail": "failed_requests",
                "p50_ttft": "time_to_first_token-p50",
            }
        )
        raw = {
            "rpm": 185.9,
            "failed_requests": 0,
            "time_to_first_token-p50": 0.312,
        }
        result = display.format_stats(raw)
        assert result["rpm"] == "185.9"
        assert result["fail"] == "0.0"
        assert result["p50_ttft"] == "0.312s"

    def test_format_stats_inverse(self):
        display = LiveStatsDisplay(
            display_stats={"tps": ("time_per_output_token-p50", "inv")}
        )
        raw = {"time_per_output_token-p50": 0.04}
        result = display.format_stats(raw)
        assert "tok/s" in result["tps"]

    def test_format_stats_missing_key_shows_placeholder(self):
        display = LiveStatsDisplay(
            display_stats={"rpm": "requests_per_minute", "missing": "nonexistent_key"}
        )
        result = display.format_stats({"requests_per_minute": 123.4})
        assert result["rpm"] == "123.4"
        assert result["missing"] == "—"

    def test_format_stats_round_number_float(self):
        display = LiveStatsDisplay(display_stats={"rpm": "requests_per_minute"})
        result = display.format_stats({"requests_per_minute": 100.0})
        assert result["rpm"] == "100.0"

    def test_custom_display_stats(self):
        custom = {"latency": "time_to_last_token-p99", "errors": "failed_requests"}
        display = LiveStatsDisplay(display_stats=custom)
        assert display._display_stats == custom

    def test_default_display_stats_used(self):
        display = LiveStatsDisplay()
        assert display._display_stats is DEFAULT_DISPLAY_STATS

    def test_terminal_output(self):
        display = LiveStatsDisplay(
            disabled=False,
            display_stats={"rpm": "rpm", "fail": "failed_requests"},
        )
        display._is_notebook = False
        display.update({"rpm": 100.0, "failed_requests": 0})
        assert display._last_line_count > 0
        display.close()
        assert display._last_line_count == 0

    def test_terminal_with_prefix(self):
        display = LiveStatsDisplay(disabled=False, display_stats={"rpm": "rpm"})
        display._is_notebook = False
        display.update({"rpm": 100.0}, extra_prefix="reqs=42")
        assert display._last_line_count >= 2  # prefix line + stats line
        display.close()

    def test_terminal_overwrites_previous(self):
        display = LiveStatsDisplay(disabled=False, display_stats={"rpm": "rpm"})
        display._is_notebook = False
        display.update({"rpm": 100.0})
        first_count = display._last_line_count
        display.update({"rpm": 200.0})
        assert display._last_line_count == first_count
        display.close()
