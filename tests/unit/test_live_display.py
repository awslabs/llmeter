# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from unittest.mock import patch

from llmeter.live_display import (
    LiveStatsDisplay,
    _classify,
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


# ── LiveStatsDisplay ─────────────────────────────────────────────────────────


class TestLiveStatsDisplay:
    def test_disabled_does_nothing(self):
        display = LiveStatsDisplay(disabled=True)
        # Should not raise
        display.update({"rpm": "100"})
        display.close()

    def test_update_empty_stats_does_nothing(self):
        display = LiveStatsDisplay(disabled=False)
        display.update({})
        assert display._handle is None
        assert display._last_line_count == 0

    def test_terminal_output(self, capsys):
        display = LiveStatsDisplay(disabled=False)
        display._is_notebook = False
        display.update({"rpm": "100", "fail": "0"})
        # Should have written to stderr
        assert display._last_line_count > 0
        display.close()
        assert display._last_line_count == 0

    def test_terminal_with_prefix(self, capsys):
        display = LiveStatsDisplay(disabled=False)
        display._is_notebook = False
        display.update({"rpm": "100"}, extra_prefix="reqs=42")
        assert display._last_line_count >= 2  # prefix line + stats line
        display.close()

    def test_terminal_overwrites_previous(self):
        display = LiveStatsDisplay(disabled=False)
        display._is_notebook = False
        display.update({"rpm": "100"})
        first_count = display._last_line_count
        display.update({"rpm": "200"})
        # Should still be same number of lines (overwritten)
        assert display._last_line_count == first_count
        display.close()
