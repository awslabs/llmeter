from unittest.mock import AsyncMock, Mock, NonCallableMock

import pytest

from llmeter.callbacks.cost.model import CostModel
from llmeter.callbacks.cost.results import CalculatedCostWithDimensions


def test_cost_model_serialization():
    """Cost models can be serialized & de-serialized"""
    spec = {
        "_type": "CostModel",
        "request_dims": {
            "TokensIn": {"_type": "InputTokens", "price_per_million": 30},
        },
        "run_dims": {
            "ComputeSeconds": {"_type": "EndpointTime", "price_per_hour": 50},
        },
    }
    model = CostModel.from_dict(spec)
    assert model.request_dims["TokensIn"].price_per_million == 30
    assert model.run_dims["ComputeSeconds"].price_per_hour == 50
    assert model.to_dict() == {
        "_type": "CostModel",
        "request_dims": {
            "TokensIn": {
                "_type": "InputTokens",
                "price_per_million": 30,
                "granularity": 1,
            },
        },
        "run_dims": {
            "ComputeSeconds": {
                "_type": "EndpointTime",
                "price_per_hour": 50,
                "granularity_secs": 1,
            },
        },
    }


@pytest.mark.asyncio
async def test_cost_model_callback_saves_request_costs():
    """By default, CostModel callbacks save request cost calculations to InvocationResponse"""
    dummy_req_dim = Mock()
    dummy_req_dim.calculate = AsyncMock(return_value=42)

    model = CostModel(
        request_dims=[dummy_req_dim],
        run_dims=[],
    )

    response_mock = NonCallableMock()
    assert await model.after_invoke(response_mock) is None
    assert response_mock.cost_total == 42
    assert response_mock.cost_Mock == 42  # Class name is the default dimension name

    # Check calculate_* fn produces same result as callback:
    assert await model.calculate_request_cost(
        response_mock
    ) == CalculatedCostWithDimensions.load_from_namespace(
        response_mock, key_prefix="cost_"
    )


@pytest.mark.asyncio
async def test_cost_model_callback_saves_run_costs():
    """By default, CostModel callbacks save run cost calculations to Result"""
    dummy_run_dim = Mock()
    dummy_run_dim.before_run_start = AsyncMock()
    dummy_run_dim.calculate = AsyncMock(return_value=5000)

    model = CostModel(
        request_dims=[],
        run_dims=[dummy_run_dim],
    )

    runner_mock = NonCallableMock()
    assert await model.before_run(runner_mock) is None
    results_mock = NonCallableMock()
    results_mock.additional_metrics_for_aggregation = None
    results_mock.responses = []
    assert await model.after_run(results_mock) is None
    assert results_mock.cost_total == 5000
    assert results_mock.cost_Mock == 5000  # Class name is the default dimension name

    # Check calculate_* fn produces same result as callback:
    await model.before_run(runner_mock) is None
    assert await model.calculate_run_cost(
        results_mock
    ) == CalculatedCostWithDimensions.load_from_namespace(
        results_mock, key_prefix="cost_"
    )


@pytest.mark.asyncio
async def test_cost_model_combines_req_and_run_dims():
    req_dim_1 = Mock()
    req_dim_1.calculate = AsyncMock(return_value=1)
    req_dim_2 = Mock()
    req_dim_2.calculate = AsyncMock(return_value=10)
    run_dim_1 = Mock()
    run_dim_1.calculate = AsyncMock(return_value=5000)
    run_dim_2 = Mock()
    run_dim_2.calculate = AsyncMock(return_value=100)
    run_dim_1.before_run_start = AsyncMock()
    run_dim_2.before_run_start = AsyncMock()

    model = CostModel(
        request_dims={"Req1": req_dim_1, "Req2": req_dim_2},
        run_dims={"Run1": run_dim_1, "Run2": run_dim_2},
    )

    # Run the dummy test:
    runner_mock = NonCallableMock()
    await model.before_run(runner_mock)
    response_mocks = [NonCallableMock(), NonCallableMock(), NonCallableMock()]
    for r in response_mocks:
        await model.after_invoke(r)
    results_mock = NonCallableMock()
    update_contrib_stats_mock = Mock()
    results_mock._update_contributed_stats = update_contrib_stats_mock
    results_mock.responses = response_mocks
    results_mock.additional_metrics_for_aggregation = None
    await model.after_run(results_mock)

    # Check the cost results:
    assert results_mock.cost_Run1 == 5000
    assert results_mock.cost_Run2 == 100
    assert results_mock.cost_Req1 == 3
    assert results_mock.cost_Req2 == 30
    assert results_mock.cost_total == 5133

    # And the summaries:
    update_contrib_stats_mock.assert_called_once()
    actual_stats = update_contrib_stats_mock.call_args[0][0]
    assert isinstance(actual_stats, dict)
    expected_stats_subset = {
        # Overall costs:
        "cost_total": 5133,
        "cost_Run1": 5000,
        "cost_Run2": 100,
        "cost_Req1": 3,
        "cost_Req2": 30,
        # Dimension-level per-request summary stats:
        "cost_Req1_per_request-average": 1,
        "cost_Req2_per_request-p90": 10,
        # Total-cost-level per-request summary stats:
        "cost_per_request-average": 11,
        # ...and etc
    }
    actual_stats_subset = {
        k: v for k, v in actual_stats.items() if k in expected_stats_subset
    }
    assert actual_stats_subset == expected_stats_subset
    # Stats include overall, plus 4 stats (avg, p50, p90, p99) for each request-level dimension and
    # request totals:
    assert len(actual_stats) == 5 + 4 * 3

    # Check recalculating with an adjusted model works correctly:
    req_dim_1.calculate = AsyncMock(return_value=2)
    run_dim_1.calculate = AsyncMock(return_value=6000)
    new_costs = await model.calculate_run_cost(results_mock)
    assert new_costs["Run1"] == 6000
    assert new_costs["Run2"] == 100
    assert new_costs["Req1"] == 6
    assert new_costs["Req2"] == 30
    assert new_costs.total == 6136
