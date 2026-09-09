# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CI-gated quantile behavior in summary_stats_from_list."""

import random


from llmeter.utils import summary_stats_from_list


class TestSummaryStatsDefaultBehavior:
    """Test that the default (confidence=None) preserves legacy behavior."""

    def test_default_always_reports_all_quantiles(self):
        """With default settings, all percentiles are always reported."""
        random.seed(42)
        data = [random.gauss(100, 10) for _ in range(5)]
        stats = summary_stats_from_list(data, percentiles=(50, 90, 99))
        assert "p50" in stats
        assert "p90" in stats
        assert "p99" in stats

    def test_default_no_ci_keys(self):
        """Default behavior does not include CI bounds."""
        random.seed(42)
        data = [random.gauss(100, 10) for _ in range(500)]
        stats = summary_stats_from_list(data, percentiles=(50, 90, 99))
        assert "p90_ci_lower" not in stats
        assert "p90_ci_upper" not in stats

    def test_empty_data_returns_empty(self):
        stats = summary_stats_from_list([])
        assert stats == {}

    def test_single_value(self):
        """Single value always reports the value for all percentiles."""
        stats = summary_stats_from_list([42.0], percentiles=(50, 90, 99))
        assert stats["p50"] == 42.0
        assert stats["p90"] == 42.0
        assert stats["p99"] == 42.0


class TestSummaryStatsWithConfidence:
    """Test opt-in confidence interval and gating behavior."""

    def test_small_sample_omits_p99(self):
        """With 20 samples and confidence=0.95, p99 should be omitted (needs ~473)."""
        random.seed(42)
        data = [random.gauss(100, 10) for _ in range(20)]
        stats = summary_stats_from_list(
            data, percentiles=(50, 90, 99), confidence=0.95
        )
        assert "average" in stats
        assert "p99" not in stats
        # p50 needs n >= 8, so should be present for n=20
        assert "p50" in stats

    def test_small_sample_omits_p90(self):
        """With 20 samples, p90 also omitted (needs ~46)."""
        random.seed(42)
        data = [random.gauss(100, 10) for _ in range(20)]
        stats = summary_stats_from_list(
            data, percentiles=(50, 90, 99), confidence=0.95
        )
        assert "p90" not in stats

    def test_large_sample_includes_all(self):
        """With 500 samples, all quantiles should be present."""
        random.seed(42)
        data = [random.gauss(100, 10) for _ in range(500)]
        stats = summary_stats_from_list(
            data, percentiles=(50, 90, 99), confidence=0.95
        )
        assert "p50" in stats
        assert "p90" in stats
        assert "p99" in stats

    def test_ci_bounds_included(self):
        """CI bounds should be attached when quantile is reported."""
        random.seed(42)
        data = [random.gauss(100, 10) for _ in range(500)]
        stats = summary_stats_from_list(
            data, percentiles=(50, 90, 99), confidence=0.95
        )
        assert "p90_ci_lower" in stats
        assert "p90_ci_upper" in stats
        assert stats["p90_ci_lower"] < stats["p90"]
        assert stats["p90_ci_upper"] > stats["p90"]

    def test_ci_lower_confidence_is_less_strict(self):
        """Lower confidence level requires fewer samples."""
        random.seed(42)
        data = [random.gauss(100, 10) for _ in range(30)]
        # At 95%, p90 is gated (needs ~46)
        stats_95 = summary_stats_from_list(
            data, percentiles=(90,), confidence=0.95
        )
        assert "p90" not in stats_95

        # At 80%, fewer samples needed
        stats_80 = summary_stats_from_list(
            data, percentiles=(90,), confidence=0.80
        )
        assert "p90" in stats_80

    def test_single_value_gated_out(self):
        """Single value: n=1 is too small for any quantile CI at 95%."""
        stats = summary_stats_from_list(
            [42.0], percentiles=(50, 90, 99), confidence=0.95
        )
        # n=1 can't form any CI
        assert "p50" not in stats
        assert "p90" not in stats
        assert "p99" not in stats
        # average is always reported
        assert "average" in stats
