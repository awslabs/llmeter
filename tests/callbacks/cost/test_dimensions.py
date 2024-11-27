import pytest

from llmeter.callbacks.cost.dimensions import (
    CostPerInputToken,
    CostPerHour,
    CostPerOutputToken,
)
from llmeter.endpoints.base import InvocationResponse
from llmeter.results import Result


@pytest.mark.asyncio
async def test_cost_per_input_token():
    spec = {
        "_type": "CostPerInputToken",
        "rate_per_million": 30,
        "granularity_tokens": 10,
    }
    dim_valid = CostPerInputToken.from_dict(spec)

    success_response = InvocationResponse(response_text="hi", num_tokens_input=199999)
    assert await dim_valid.calculate(success_response) == 6

    unk_tokens_response = InvocationResponse(
        response_text="hi", num_tokens_output=199999
    )
    assert await dim_valid.calculate(unk_tokens_response) is None

    err_response = InvocationResponse.error_output()
    assert await dim_valid.calculate(err_response) is None

    assert dim_valid.to_dict() == {
        "_type": "CostPerInputToken",
        "rate_per_million": 30,
        "granularity_tokens": 10,
    }


@pytest.mark.asyncio
async def test_cost_per_output_token():
    spec = {
        "_type": "CostPerOutputToken",
        "rate_per_million": 40,
        "granularity_tokens": 10,
    }
    dim_valid = CostPerOutputToken.from_dict(spec)

    success_response = InvocationResponse(response_text="hi", num_tokens_output=199999)
    assert await dim_valid.calculate(success_response) == 8

    unk_tokens_response = InvocationResponse(
        response_text="hi", num_tokens_input=199999
    )
    assert await dim_valid.calculate(unk_tokens_response) is None

    err_response = InvocationResponse.error_output()
    assert await dim_valid.calculate(err_response) is None

    assert dim_valid.to_dict() == {
        "_type": "CostPerOutputToken",
        "rate_per_million": 40,
        "granularity_tokens": 10,
    }


@pytest.mark.asyncio
async def test_cost_per_hour():
    spec = {
        "_type": "CostPerHour",
        "rate": 30,
        "granularity_secs": 60,
    }
    dim_valid = CostPerHour.from_dict(spec)

    result = Result(
        [],
        total_requests=0,
        clients=0,
        n_requests=0,
        total_test_time=50,
    )
    assert await dim_valid.calculate(result) == 30 * 60 / 3600
    result.total_test_time = 65
    assert await dim_valid.calculate(result) == 30 * 120 / 3600
    result.total_test_time = 0
    assert await dim_valid.calculate(result) == 0
    result.total_test_time = None
    assert await dim_valid.calculate(result) is None

    assert dim_valid.to_dict() == {
        "_type": "CostPerHour",
        "rate": 30,
        "granularity_secs": 60,
    }
