# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from llmeter.callbacks.cost.results import CalculatedCostWithDimensions


def test_cost_recalculates_total():
    """CalculatedCostWithDimensions.total should be re-calculated automatically"""
    cost1 = CalculatedCostWithDimensions(foo=4, bar=2)
    assert cost1.total == 6
    cost1["baz"] = 6  # Add new dimension
    assert cost1.total == 12
    cost1["bar"] = 4  # Update existing dimension
    assert cost1.total == 14


def test_calculated_cost_merge():
    """.merge() should combine dimensions with the same `name`, and modify in-place"""
    cost1 = CalculatedCostWithDimensions(foo=4, bar=2)
    cost2 = CalculatedCostWithDimensions(bar=4, baz=9)
    cost2_copy = CalculatedCostWithDimensions(bar=4, baz=9)
    cost1.merge(cost2)
    # Original cost2 untouched:
    assert cost2.total == 13
    assert cost2 == cost2_copy
    assert cost2["bar"] == 4
    # Merge is correct:
    assert cost1.total == 19
    assert len(cost1) == 3
    assert cost1["bar"] == 6


def test_calculated_cost_sum():
    """CalculatedCostWithDimensions should support + and sum() operations"""
    cost1 = CalculatedCostWithDimensions(foo=4, bar=2)
    cost1_copy = CalculatedCostWithDimensions(foo=4, bar=2)
    cost2 = CalculatedCostWithDimensions(bar=4, baz=9)
    cost2_copy = CalculatedCostWithDimensions(bar=4, baz=9)
    # Sum:
    sumres = cost1 + cost2
    # Original objects untouched:
    assert cost1.total == 6
    assert cost1 == cost1_copy
    assert cost2.total == 13
    assert cost2 == cost2_copy
    # Sum is correct:
    assert sumres.total == 19
    assert len(sumres) == 3
    assert sumres["bar"] == 6

    # In addition to `+` operator, `sum()` should also work:
    assert sum([cost1, cost2]) == sumres


def test_calculated_cost_summary_stats():
    """CalculatedCostWithDimensions.summary_statistics outputs expected format"""
    costs = [
        CalculatedCostWithDimensions(input_tokens=1, output_tokens=9),
        CalculatedCostWithDimensions(input_tokens=2, output_tokens=8),
        CalculatedCostWithDimensions(input_tokens=3, output_tokens=7),
        CalculatedCostWithDimensions(input_tokens=4, output_tokens=6),
        CalculatedCostWithDimensions(input_tokens=5, output_tokens=5),
        CalculatedCostWithDimensions(input_tokens=6, other_dim=4),
        CalculatedCostWithDimensions(input_tokens=7, other_dim=3),
        CalculatedCostWithDimensions(input_tokens=8, other_dim=2),
        CalculatedCostWithDimensions(input_tokens=9, other_dim=1),
        CalculatedCostWithDimensions(input_tokens=10, other_dim=0),
    ]
    # Default key name parameters:
    summary = CalculatedCostWithDimensions.summary_statistics(costs)
    assert set(summary.keys()) == set(
        (
            "input_tokens-average",
            "input_tokens-p50",
            "input_tokens-p90",
            "input_tokens-p99",
            "other_dim-average",
            "other_dim-p50",
            "other_dim-p90",
            "other_dim-p99",
            "output_tokens-average",
            "output_tokens-p50",
            "output_tokens-p90",
            "output_tokens-p99",
            "total-average",
            "total-p50",
            "total-p90",
            "total-p99",
        )
    )
    assert summary["total-average"] == 10
    assert summary["input_tokens-p50"] == 5.5

    # Custom key name parameters:
    summary = CalculatedCostWithDimensions.summary_statistics(
        costs,
        key_prefix="foo_",
        key_dim_name_suffix="_bar",
        key_stat_name_prefix="~",
        key_total_name_and_suffix="baz",
    )
    assert set(summary.keys()) == set(
        (
            "foo_input_tokens_bar~average",
            "foo_input_tokens_bar~p50",
            "foo_input_tokens_bar~p90",
            "foo_input_tokens_bar~p99",
            "foo_other_dim_bar~average",
            "foo_other_dim_bar~p50",
            "foo_other_dim_bar~p90",
            "foo_other_dim_bar~p99",
            "foo_output_tokens_bar~average",
            "foo_output_tokens_bar~p50",
            "foo_output_tokens_bar~p90",
            "foo_output_tokens_bar~p99",
            "foo_baz~average",
            "foo_baz~p50",
            "foo_baz~p90",
            "foo_baz~p99",
        )
    )
    assert summary["foo_baz~p99"] == 10
