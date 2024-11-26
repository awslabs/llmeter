from unittest.mock import AsyncMock, Mock, NonCallableMock

import pytest

from llmeter.callbacks.cost.model import CostModel
from llmeter.callbacks.cost.results import CalculatedCostWithDimensions


def test_cost_model_serialization():
    """Cost models can be serialized & de-serialized"""
    spec = {
        "_type": "CostModel",
        "request_dims": [
            {"_type": "CostPerInputToken", "name": "TokensIn", "rate_per_million": 30}
        ],
        "run_dims": [{"_type": "CostPerHour", "name": "ComputeSeconds", "rate": 50}],
    }
    model = CostModel.from_dict(spec)
    assert model.request_dims[0].rate_per_million == 30
    assert model.run_dims[0].rate == 50
    assert model.to_dict() == {
        "_type": "CostModel",
        "request_dims": [
            {
                "_type": "CostPerInputToken",
                "name": "TokensIn",
                "rate_per_million": 30,
                "granularity_tokens": 1,
            }
        ],
        "run_dims": [
            {
                "_type": "CostPerHour",
                "name": "ComputeSeconds",
                "granularity_secs": 1,
                "rate": 50,
            }
        ],
    }


@pytest.mark.asyncio
async def test_cost_model_callback_saves_request_costs():
    """By default, CostModel callbacks save request cost calculations to InvocationResponse"""
    dummy_req_dim = Mock()
    dummy_req_dim.name = "MockDimension"
    dummy_req_dim.calculate = AsyncMock(return_value=42)

    model = CostModel(
        request_dims=[dummy_req_dim],
        run_dims=[],
    )

    response_mock = NonCallableMock()
    assert await model.after_invoke(response_mock) is None
    assert isinstance(response_mock.cost, CalculatedCostWithDimensions)
    assert response_mock.cost.total == 42
    assert len(response_mock.cost.dims) == 1
    assert response_mock.cost.dims[0].name == "MockDimension"

    # Check calculate_* fn produces same result as callback:
    assert await model.calculate_request_cost(response_mock) == response_mock.cost


@pytest.mark.asyncio
async def test_cost_model_callback_saves_run_costs():
    """By default, CostModel callbacks save run cost calculations to Result"""
    dummy_run_dim = Mock()
    dummy_run_dim.name = "MockDimension"
    dummy_run_dim.calculate = AsyncMock(return_value=5000)

    model = CostModel(
        request_dims=[],
        run_dims=[dummy_run_dim],
    )

    runner_mock = NonCallableMock()
    assert await model.before_run(runner_mock) is None
    results_mock = NonCallableMock()
    results_mock.responses = []
    assert await model.after_run(results_mock) is None
    assert isinstance(results_mock.cost, CalculatedCostWithDimensions)
    assert results_mock.cost.total == 5000
    assert len(results_mock.cost.dims) == 1
    assert results_mock.cost.dims[0].name == "MockDimension"

    # Check calculate_* fn produces same result as callback:
    await model.before_run(runner_mock) is None
    assert await model.calculate_run_cost(results_mock) == results_mock.cost


@pytest.mark.asyncio
async def test_cost_model_combines_req_and_run_dims():
    dummy_req_dim = Mock()
    dummy_req_dim.name = "MockByRequest"
    dummy_req_dim.calculate = AsyncMock(return_value=1)
    dummy_shared_dim_req = Mock()
    dummy_shared_dim_req.name = "SharedDim"
    dummy_shared_dim_req.calculate = AsyncMock(return_value=10)
    dummy_run_dim = Mock()
    dummy_run_dim.name = "MockByRun"
    dummy_run_dim.calculate = AsyncMock(return_value=5000)
    dummy_shared_dim_run = Mock()
    dummy_shared_dim_run.name = "SharedDim"
    dummy_shared_dim_run.calculate = AsyncMock(return_value=100)

    model = CostModel(
        request_dims=[dummy_req_dim, dummy_shared_dim_req],
        run_dims=[dummy_run_dim, dummy_shared_dim_run],
    )

    # Run the dummy test:
    runner_mock = NonCallableMock()
    await model.before_run(runner_mock)
    response_mocks = [NonCallableMock(), NonCallableMock(), NonCallableMock()]
    for r in response_mocks:
        await model.after_invoke(r)
    results_mock = NonCallableMock()
    results_mock.responses = response_mocks
    await model.after_run(results_mock)

    # Check the cost results:
    assert isinstance(results_mock.cost, CalculatedCostWithDimensions)
    assert len(results_mock.cost.dims) == 3
    assert next(d.cost for d in results_mock.cost.dims if d.name == "MockByRun") == 5000
    assert (
        next(d.cost for d in results_mock.cost.dims if d.name == "MockByRequest") == 3
    )
    assert next(d.cost for d in results_mock.cost.dims if d.name == "SharedDim") == 130
    assert results_mock.cost.total == 5133

    # Check recalculating with an adjusted model works correctly:
    dummy_req_dim.calculate = AsyncMock(return_value=2)
    dummy_run_dim.calculate = AsyncMock(return_value=6000)
    new_costs = await model.calculate_run_cost(
        results_mock, include_request_costs="recalculate"
    )
    assert next(d.cost for d in new_costs.dims if d.name == "MockByRun") == 6000
    assert next(d.cost for d in new_costs.dims if d.name == "MockByRequest") == 6
    assert next(d.cost for d in new_costs.dims if d.name == "SharedDim") == 130
    assert new_costs.total == 6136
