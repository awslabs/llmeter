# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantile confidence interval estimation — zero external dependencies.

Provides functions to compute confidence intervals for quantiles using the
exact binomial method, and to gate unreliable quantile estimates based on
sample size. Only uses the Python standard library (math.lgamma).

Reference: The confidence interval for a quantile is based on order statistics
of the binomial distribution. If X_(1), ..., X_(n) are sorted i.i.d. samples,
then [X_(l+1), X_(u+1)] covers the true p-quantile with probability:

    P(l <= Binomial(n, p) <= u - 1)

See: https://en.wikipedia.org/wiki/Quantile#Confidence_intervals
"""

from math import exp, lgamma, log
from typing import Optional, Sequence, Tuple


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    """Log of the binomial PMF: log(C(n,k) * p^k * (1-p)^(n-k)).

    Uses lgamma to avoid integer overflow for large n.
    """
    if p == 0:
        return 0.0 if k == 0 else float("-inf")
    if p == 1:
        return 0.0 if k == n else float("-inf")
    return (
        lgamma(n + 1)
        - lgamma(k + 1)
        - lgamma(n - k + 1)
        + k * log(p)
        + (n - k) * log(1 - p)
    )


def _binom_cdf(k: int, n: int, p: float) -> float:
    """Cumulative distribution function P(X <= k) for Binomial(n, p).

    Uses log-space arithmetic per term to avoid overflow.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    cumulative = 0.0
    for i in range(k + 1):
        cumulative += exp(_log_binom_pmf(i, n, p))
    return cumulative


def _binom_ppf(q: float, n: int, p: float) -> int:
    """Percent point function (inverse CDF) for Binomial(n, p).

    Returns the smallest k such that P(X <= k) >= q.
    Complexity: O(n) — negligible for n < 10,000.
    """
    if q <= 0:
        return 0
    if q >= 1:
        return n

    cumulative = 0.0
    for k in range(n + 1):
        cumulative += exp(_log_binom_pmf(k, n, p))
        if cumulative >= q:
            return k
    return n


def _ci_coverage(lower_idx: int, upper_idx: int, n: int, p: float) -> float:
    """Compute actual coverage probability of a quantile CI.

    The interval [X_(lower+1), X_(upper+1)] covers the true p-quantile
    when the number of samples below it (Binomial(n, p)) falls in
    [lower_idx, upper_idx - 1].

    Coverage = P(lower_idx <= X <= upper_idx - 1)
             = CDF(upper_idx - 1) - CDF(lower_idx - 1)
    """
    cdf_upper = _binom_cdf(upper_idx - 1, n, p)
    cdf_lower = _binom_cdf(lower_idx - 1, n, p) if lower_idx > 0 else 0.0
    return cdf_upper - cdf_lower


def _find_ci_indices(
    n: int, quantile: float, confidence: float
) -> Optional[Tuple[int, int]]:
    """Find order-statistic indices that achieve the requested coverage.

    Uses the binomial ppf as a starting point, then verifies actual coverage.
    Returns (lower_idx, upper_idx) where the CI is
    [sorted_data[lower_idx], sorted_data[upper_idx]], or None if no valid
    interval exists within the data range.
    """
    alpha = 1 - confidence

    # Starting point from binomial inverse CDF:
    # lower_idx: one below the alpha/2 quantile of Binom(n, p)
    # upper_idx: the (1 - alpha/2) quantile of Binom(n, p)
    lower_idx = max(0, _binom_ppf(alpha / 2, n, quantile) - 1)
    upper_idx = min(n - 1, _binom_ppf(1 - alpha / 2, n, quantile))

    if lower_idx >= upper_idx:
        return None

    # Verify actual coverage meets the confidence threshold
    coverage = _ci_coverage(lower_idx, upper_idx, n, quantile)
    if coverage >= confidence:
        return (lower_idx, upper_idx)

    # If coverage is insufficient, try widening by one on each side
    # (accounts for discreteness of the binomial distribution)
    candidates = []
    if lower_idx > 0:
        c = _ci_coverage(lower_idx - 1, upper_idx, n, quantile)
        if c >= confidence:
            candidates.append((lower_idx - 1, upper_idx, c))
    if upper_idx < n - 1:
        c = _ci_coverage(lower_idx, upper_idx + 1, n, quantile)
        if c >= confidence:
            candidates.append((lower_idx, upper_idx + 1, c))

    if not candidates:
        return None

    # Pick the tightest interval (smallest coverage above threshold)
    best = min(candidates, key=lambda x: x[2])
    return (best[0], best[1])


def quantile_ci(
    data: Sequence[float],
    quantile: float,
    confidence: float = 0.95,
) -> Optional[Tuple[float, float]]:
    """Compute a confidence interval for the given quantile of a dataset.

    Uses the exact binomial method to find order statistic indices that
    bracket the true quantile. Verifies that the actual coverage probability
    meets the requested confidence level (accounting for the discreteness
    of the binomial distribution).

    Args:
        data: Sequence of observed values (e.g., latencies in seconds).
        quantile: The quantile of interest (e.g., 0.9 for p90, 0.99 for p99).
        confidence: Desired confidence level (default 0.95 for 95% CI).

    Returns:
        A tuple (lower_bound, upper_bound) representing the confidence interval,
        or None if the sample size is too small to achieve the requested coverage.
    """
    sorted_data = sorted(data)
    n = len(sorted_data)

    if n < 2:
        return None

    result = _find_ci_indices(n, quantile, confidence)
    if result is None:
        return None

    lower_idx, upper_idx = result
    return (sorted_data[lower_idx], sorted_data[upper_idx])


def can_estimate_quantile(
    n: int,
    quantile: float,
    confidence: float = 0.95,
) -> bool:
    """Check whether a CI can be formed for the given quantile and sample size.

    Verifies that the actual achievable coverage meets the requested confidence
    level. Useful as a validity gate: if this returns False, the point estimate
    of the quantile should be considered unreliable and may be omitted/flagged.

    Args:
        n: Sample size.
        quantile: The quantile of interest (e.g. 0.99 for p99).
        confidence: Desired confidence level.

    Returns:
        True if a confidence interval with adequate coverage can be formed.
    """
    if n < 2:
        return False
    return _find_ci_indices(n, quantile, confidence) is not None
