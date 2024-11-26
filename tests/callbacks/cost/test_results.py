from llmeter.callbacks.cost.results import (
    CalculatedCostDimension,
    CalculatedCostWithDimensions,
)


def test_cost_recalculates_total():
    """CalculatedCostWithDimensions.total should be re-calculated automatically"""
    cost1 = CalculatedCostWithDimensions(
        dims=[
            CalculatedCostDimension(name="foo", cost=4),
            CalculatedCostDimension(name="bar", cost=2),
        ],
    )
    assert cost1.total == 6
    cost1.dims.append(CalculatedCostDimension(name="baz", cost=6))
    assert cost1.total == 12


def test_calculated_cost_merge():
    """.merge() should combine dimensions with the same `name`, and modify in-place"""
    cost1 = CalculatedCostWithDimensions(
        dims=[
            CalculatedCostDimension(name="foo", cost=4),
            CalculatedCostDimension(name="bar", cost=2),
        ],
    )
    cost2 = CalculatedCostWithDimensions(
        dims=[
            CalculatedCostDimension(name="bar", cost=4),
            CalculatedCostDimension(name="baz", cost=9),
            CalculatedCostDimension(name="baz", cost=100),
        ],
    )
    cost2_copy = CalculatedCostWithDimensions(
        dims=[
            CalculatedCostDimension(name="bar", cost=4),
            CalculatedCostDimension(name="baz", cost=9),
        ],
    )
    cost1.merge(cost2)
    # Original cost2 untouched:
    assert cost2.total == 13
    assert cost2 == cost2_copy
    assert next(d for d in cost2.dims if d.name == "bar").cost == 4
    # Merge is correct:
    assert cost1.total == 19
    assert len(cost1.dims) == 3
    assert next(d for d in cost1.dims if d.name == "bar").cost == 6


def test_calculated_cost_sum():
    """CalculatedCostWithDimensions should support + and sum() operations"""
    cost1 = CalculatedCostWithDimensions(
        dims=[
            CalculatedCostDimension(name="foo", cost=4),
            CalculatedCostDimension(name="bar", cost=2),
        ],
    )
    cost1_copy = CalculatedCostWithDimensions(
        dims=[
            CalculatedCostDimension(name="foo", cost=4),
            CalculatedCostDimension(name="bar", cost=2),
        ],
    )
    cost2 = CalculatedCostWithDimensions(
        dims=[
            CalculatedCostDimension(name="bar", cost=4),
            CalculatedCostDimension(name="baz", cost=9),
        ],
    )
    cost2_copy = CalculatedCostWithDimensions(
        dims=[
            CalculatedCostDimension(name="bar", cost=4),
            CalculatedCostDimension(name="baz", cost=9),
        ],
    )
    # Sum:
    sumres = cost1 + cost2
    # Original objects untouched:
    assert cost1.total == 6
    assert cost1 == cost1_copy
    assert cost2.total == 13
    assert cost2 == cost2_copy
    # Sum is correct:
    assert sumres.total == 19
    assert len(sumres.dims) == 3
    assert next(d for d in sumres.dims if d.name == "bar").cost == 6

    # In addition to `+` operator, `sum()` should also work:
    assert sum([cost1, cost2]) == sumres
