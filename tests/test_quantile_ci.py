# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantile confidence interval estimation."""


from llmeter.quantile_ci import (
    _binom_cdf,
    _binom_ppf,
    _ci_coverage,
    can_estimate_quantile,
    quantile_ci,
)


class TestBinomCdf:
    """Test the pure-Python binomial CDF implementation."""

    def test_known_value_n10_p05(self):
        """Binom(10, 0.5) CDF at k=5 is 0.623046875."""
        assert abs(_binom_cdf(5, 10, 0.5) - 0.623046875) < 1e-9

    def test_boundary_k_negative(self):
        assert _binom_cdf(-1, 10, 0.5) == 0.0

    def test_boundary_k_equals_n(self):
        assert _binom_cdf(10, 10, 0.5) == 1.0

    def test_p_zero(self):
        assert _binom_cdf(0, 5, 0.0) == 1.0
        assert _binom_cdf(-1, 5, 0.0) == 0.0

    def test_p_one(self):
        assert _binom_cdf(4, 5, 1.0) == 0.0
        assert _binom_cdf(5, 5, 1.0) == 1.0

    def test_large_n_no_overflow(self):
        """Ensure no overflow for n=5000."""
        result = _binom_cdf(4950, 5000, 0.99)
        assert 0.0 <= result <= 1.0


class TestBinomPpf:
    """Test the inverse CDF."""

    def test_median_n10_p05(self):
        """Smallest k with CDF >= 0.5 for Binom(10, 0.5) is 5."""
        assert _binom_ppf(0.5, 10, 0.5) == 5

    def test_known_value_n100_p09(self):
        """ppf(0.025, 100, 0.9) should be around 84."""
        result = _binom_ppf(0.025, 100, 0.9)
        assert 82 <= result <= 86

    def test_boundary_q_zero(self):
        assert _binom_ppf(0, 100, 0.5) == 0

    def test_boundary_q_one(self):
        assert _binom_ppf(1, 100, 0.5) == 100


class TestCiCoverage:
    """Test the coverage probability calculation."""

    def test_full_range_n3_p05(self):
        """[X_(1), X_(3)] for n=3, p=0.5 should cover 50% (not 95%)."""
        # Coverage = P(0 <= X <= 1) = P(X=0) + P(X=1) = 0.125 + 0.375 = 0.5
        coverage = _ci_coverage(0, 2, 3, 0.5)
        assert abs(coverage - 0.5) < 0.01

    def test_n8_p05_covers_95(self):
        """n=8, p50: indices [1, 7] should achieve >= 95% coverage."""
        coverage = _ci_coverage(1, 7, 8, 0.5)
        assert coverage >= 0.95


class TestCanEstimateQuantile:
    """Test the reliability gate function."""

    def test_p50_needs_at_least_8(self):
        """p50 at 95% confidence requires n >= 8."""
        assert not can_estimate_quantile(7, 0.5)
        assert can_estimate_quantile(8, 0.5)

    def test_p90_needs_many_samples(self):
        """p90 at 95% confidence requires n >= 46."""
        assert not can_estimate_quantile(10, 0.9)
        assert not can_estimate_quantile(30, 0.9)
        assert can_estimate_quantile(50, 0.9)

    def test_p99_needs_hundreds(self):
        """p99 at 95% confidence requires n >= 482."""
        assert not can_estimate_quantile(100, 0.99)
        assert can_estimate_quantile(500, 0.99)

    def test_n_less_than_2(self):
        assert not can_estimate_quantile(0, 0.5)
        assert not can_estimate_quantile(1, 0.5)

    def test_lower_confidence_needs_fewer_samples(self):
        """At 80% confidence, fewer samples are needed."""
        assert can_estimate_quantile(5, 0.5, confidence=0.80)


class TestQuantileCi:
    """Test the main CI computation function."""

    def test_returns_none_for_small_sample(self):
        """Should return None when sample is too small."""
        assert quantile_ci([1, 2, 3], 0.99) is None

    def test_returns_tuple_for_large_sample(self):
        """Should return a valid (lower, upper) tuple."""
        import random

        random.seed(42)
        data = [random.gauss(100, 10) for _ in range(500)]
        ci = quantile_ci(data, 0.9)
        assert ci is not None
        lower, upper = ci
        assert lower < upper

    def test_ci_brackets_point_estimate(self):
        """The point estimate should typically fall within the CI."""
        import random

        random.seed(42)
        data = [random.gauss(100, 10) for _ in range(500)]
        ci = quantile_ci(data, 0.5)
        assert ci is not None
        point_est = sorted(data)[250]
        assert ci[0] <= point_est <= ci[1]

    def test_empty_data(self):
        assert quantile_ci([], 0.5) is None

    def test_single_value(self):
        assert quantile_ci([42.0], 0.5) is None
