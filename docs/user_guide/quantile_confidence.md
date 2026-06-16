# Quantile Confidence Intervals

## Motivation

When you report a percentile like p99 from a test run, the **reliability** of that
estimate depends on how many data points you collected. With only 50 samples, your
p99 value is based on a single observation — any outlier (network glitch, cold start,
garbage collection pause) dominates the estimate.

LLMeter can optionally compute confidence intervals for quantile estimates using the
exact binomial method. This feature also acts as a **reliability gate**: when the sample
size is too small to form a meaningful confidence interval, the quantile is omitted from
the output rather than reported with false precision.

## Enabling confidence intervals

Pass `confidence` to `summary_stats_from_list`, or set it on the `Runner`:

```python
from llmeter.utils import summary_stats_from_list

data = [resp.time_to_first_token for resp in result.responses if resp.time_to_first_token]

# With 95% confidence interval estimation and gating
stats = summary_stats_from_list(data, percentiles=(50, 90, 99), confidence=0.95)
```

When `confidence` is set:

1. **Gating** — quantiles that cannot achieve the requested confidence level are
   silently omitted from the output. For example, with `confidence=0.95`, p99 requires
   at least ~473 samples. If you only have 100, `p99` won't appear in the dict.

2. **CI bounds** — for quantiles that pass the gate, lower and upper bounds are
   included:
   ```python
   stats["p90"]            # 0.612  (point estimate)
   stats["p90_ci_lower"]   # 0.581  (lower bound of 95% CI)
   stats["p90_ci_upper"]   # 0.647  (upper bound of 95% CI)
   ```

Without `confidence` (the default), all requested percentiles are always reported
with no CI bounds — the legacy behavior is unchanged.

## Minimum sample sizes

The minimum number of samples required to form a confidence interval at 95% confidence:

| Quantile | Min samples | Practical guidance |
|----------|-------------|-------------------|
| p50      | 8           | Almost always available |
| p90      | 46          | Small test runs may miss this |
| p95      | 90          | Need ~100 successful requests |
| p99      | 482         | Need ~500 successful requests |

These are derived from the exact binomial distribution — they represent the smallest
sample sizes where the CI achieves ≥95% actual coverage probability.

!!! tip "Sizing your test runs"
    If p99 latency matters for your SLOs and you want confidence intervals,
    aim for at least 500 successful requests per run. For p90, 50 requests is sufficient.

## How it works

The method is based on the relationship between order statistics and the binomial
distribution:

1. Sort the n observations: X₍₁₎ ≤ X₍₂₎ ≤ ... ≤ X₍ₙ₎
2. The number of observations below the true p-quantile follows Binomial(n, p)
3. Find indices `l` and `u` such that P(l ≤ Binomial(n,p) ≤ u-1) ≥ confidence
4. The CI is [X₍ₗ₊₁₎, X₍ᵤ₊₁₎]

The implementation uses log-space arithmetic (`math.lgamma`) to avoid integer overflow
for large n, with zero external dependencies beyond the Python standard library.

!!! note "Coverage vs. nominal confidence"
    Because the binomial distribution is discrete, the actual coverage of the CI
    may slightly exceed the nominal confidence level. For example, the first n where
    a 95% CI is achievable for p50 is n=8, which gives 96.1% actual coverage. The
    implementation verifies that actual coverage meets the threshold rather than relying
    on a naive index check.

## Example: interpreting gated output

```python
result = await runner.run(payload=payload, n_requests=30, clients=5)

# Without confidence (default) — all percentiles reported
stats = summary_stats_from_list(
    [r.time_to_first_token for r in result.responses if r.time_to_first_token],
    percentiles=(50, 90, 99),
)
# → {"average": 0.42, "p50": 0.38, "p90": 0.61, "p99": 1.23}

# With confidence=0.95 — unreliable percentiles gated out
stats = summary_stats_from_list(
    [r.time_to_first_token for r in result.responses if r.time_to_first_token],
    percentiles=(50, 90, 99),
    confidence=0.95,
)
# With only 30 samples:
# → {"average": 0.42, "p50": 0.38, "p50_ci_lower": 0.31, "p50_ci_upper": 0.44}
# p90 and p99 are omitted because n=30 is insufficient
```

## API reference

::: llmeter.quantile_ci.quantile_ci
::: llmeter.quantile_ci.can_estimate_quantile
