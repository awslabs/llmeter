# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from llmeter.callbacks.cost.dimensions import (
    EndpointTime,
    InputTokens,
    OutputTokens,
    RequestCostDimensionBase,
    RunCostDimensionBase,
)
from llmeter.endpoints.base import InvocationResponse
from llmeter.results import Result


def test_base_classes_require_implementing_calculate():
    class TestReqDim(RequestCostDimensionBase):
        pass

    with pytest.raises(TypeError, match="without an implementation"):
        TestReqDim()

    class TestRunDim(RunCostDimensionBase):
        pass

    with pytest.raises(TypeError, match="without an implementation"):
        TestRunDim()


@pytest.mark.asyncio
async def test_cost_per_input_token():
    spec = {
        "_type": "InputTokens",
        "price_per_million": 30,
        "granularity": 10,
    }
    dim_valid = InputTokens.from_dict(spec)

    success_response = InvocationResponse(response_text="hi", num_tokens_input=199999)
    assert await dim_valid.calculate(success_response) == 6

    unk_tokens_response = InvocationResponse(
        response_text="hi", num_tokens_output=199999
    )
    assert await dim_valid.calculate(unk_tokens_response) is None

    err_response = InvocationResponse.error_output()
    assert await dim_valid.calculate(err_response) is None

    assert dim_valid.to_dict() == {
        "_type": "InputTokens",
        "price_per_million": 30,
        "granularity": 10,
    }


@pytest.mark.asyncio
async def test_cost_per_output_token():
    spec = {
        "_type": "OutputTokens",
        "price_per_million": 40,
        "granularity": 10,
    }
    dim_valid = OutputTokens.from_dict(spec)

    success_response = InvocationResponse(response_text="hi", num_tokens_output=199999)
    assert await dim_valid.calculate(success_response) == 8

    unk_tokens_response = InvocationResponse(
        response_text="hi", num_tokens_input=199999
    )
    assert await dim_valid.calculate(unk_tokens_response) is None

    err_response = InvocationResponse.error_output()
    assert await dim_valid.calculate(err_response) is None

    assert dim_valid.to_dict() == {
        "_type": "OutputTokens",
        "price_per_million": 40,
        "granularity": 10,
    }


@pytest.mark.asyncio
async def test_cost_per_hour():
    spec = {
        "_type": "EndpointTime",
        "price_per_hour": 30,
        "granularity_secs": 60,
    }
    dim_valid = EndpointTime.from_dict(spec)

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
        "_type": "EndpointTime",
        "price_per_hour": 30,
        "granularity_secs": 60,
    }
