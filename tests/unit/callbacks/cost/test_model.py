# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, NonCallableMock

import pytest

from llmeter.serialization import dump_object, load_object
from llmeter.callbacks.cost.model import CostModel
from llmeter.callbacks.cost.results import CalculatedCostWithDimensions
from llmeter.endpoints.base import InvocationResponse


def test_cost_model_serialization():
    """Cost models can be serialized & de-serialized via dump_object/load_object"""
    from llmeter.callbacks.cost.dimensions import EndpointTime, InputTokens

    model = CostModel(
        request_dims={"TokensIn": InputTokens(price_per_million=30)},
        run_dims={"ComputeSeconds": EndpointTime(price_per_hour=50)},
    )

    # Round-trip via dump_object / load_object
    data = dump_object(model)
    restored = load_object(data)

    assert restored.request_dims["TokensIn"].price_per_million == 30
    assert restored.run_dims["ComputeSeconds"].price_per_hour == 50

    # get state produces a plain dict representation
    d = model._get_llmeter_state()
    assert "request_dims" in d
    assert "run_dims" in d


def test_cost_model_save_load_file_roundtrip(tmp_path):
    """A CostModel can be saved to and loaded from a file, re-initialising its dimensions

    Exercises the inherited Serializable.save_to_file / load_from_file API (file I/O plus
    json_default), which the in-memory dump_object/load_object test does not cover.
    """
    from llmeter.callbacks.cost.dimensions import (
        EndpointTime,
        InputTokens,
        OutputTokens,
    )

    model = CostModel(
        request_dims={
            "TokensIn": InputTokens(price_per_million=30, granularity=10),
            "TokensOut": OutputTokens(price_per_million=60),
        },
        run_dims={"ComputeSeconds": EndpointTime(price_per_hour=50)},
    )

    path = tmp_path / "cost_model.json"
    saved_path = model.save_to_file(path)
    # save_to_file returns the (validated/normalized) path it wrote to
    assert str(saved_path) == str(path)
    assert path.is_file()

    # The file is valid JSON tagged with the CostModel class path
    with open(path) as f:
        raw = json.load(f)
    assert raw["__llmeter_class__"] == "llmeter.callbacks.cost.model.CostModel"

    restored = CostModel.load_from_file(path)

    # The correct concrete type is reconstructed from the file
    assert isinstance(restored, CostModel)

    # Each dimension is re-initialised as the correct type, under the same name, with values intact
    assert set(restored.request_dims) == {"TokensIn", "TokensOut"}
    assert set(restored.run_dims) == {"ComputeSeconds"}

    tokens_in = restored.request_dims["TokensIn"]
    assert isinstance(tokens_in, InputTokens)
    assert tokens_in.price_per_million == 30
    assert tokens_in.granularity == 10

    tokens_out = restored.request_dims["TokensOut"]
    assert isinstance(tokens_out, OutputTokens)
    assert tokens_out.price_per_million == 60

    compute = restored.run_dims["ComputeSeconds"]
    assert isinstance(compute, EndpointTime)
    assert compute.price_per_hour == 50


def test_cost_model_load_from_file_dispatches_via_base_class(tmp_path):
    """load_from_file resolves the concrete type from the file, even when called on Serializable

    The class is detected from the ``__llmeter_class__`` marker rather than the class the
    classmethod is invoked on, so loading via the base mixin still yields a CostModel.
    """
    from llmeter.callbacks.cost.dimensions import InputTokens
    from llmeter.serialization import Serializable

    model = CostModel(request_dims={"TokensIn": InputTokens(price_per_million=15)})

    path = tmp_path / "cost_model.json"
    model.save_to_file(path)

    restored = Serializable.load_from_file(path)
    assert isinstance(restored, CostModel)
    assert restored.request_dims["TokensIn"].price_per_million == 15
    assert restored.run_dims == {}


def test_cost_model_detects_duplicate_cost_dim_names():
    """CostModel detects and raises errors when created with duplicate dimension names"""

    @dataclass
    class DummyCostDimension:
        name: str  # Name property is *not* (currently?) used automatically - only class name

    with pytest.raises(ValueError, match="Duplicate cost dimension name"):
        CostModel(
            request_dims=[
                DummyCostDimension(name="foo"),
                DummyCostDimension(name="bar"),
            ],
            run_dims=[],
        )
    with pytest.raises(ValueError, match="Duplicate cost dimension name"):
        CostModel(
            request_dims=[],
            run_dims=[
                DummyCostDimension(name="foo"),
                DummyCostDimension(name="bar"),
            ],
        )
    with pytest.raises(ValueError, match="Duplicate cost dimension name"):
        CostModel(
            request_dims=[DummyCostDimension(name="foo")],
            run_dims=[DummyCostDimension(name="bar")],
        )
    with pytest.raises(ValueError, match="Duplicate cost dimension name"):
        CostModel(
            request_dims={"my_dim": DummyCostDimension(name="foo")},
            run_dims={"my_dim": DummyCostDimension(name="bar")},
        )


@pytest.mark.asyncio
async def test_cost_model_callback_saves_request_costs():
    """By default, CostModel callbacks save request cost calculations to response.annotations"""
    dummy_req_dim = Mock()
    dummy_req_dim.calculate = AsyncMock(return_value=42)

    model = CostModel(
        request_dims=[dummy_req_dim],
        run_dims=[],
    )

    response = InvocationResponse(response_text="hi")
    assert await model.after_invoke(response) is None
    # Costs are stored in the (persisted) annotations dict, not as loose attributes
    assert response.annotations["cost_total"] == 42
    assert (
        response.annotations["cost_Mock"] == 42
    )  # Class name is the default dimension name

    # Check calculate_* fn produces same result as callback:
    assert await model.calculate_request_cost(
        response
    ) == CalculatedCostWithDimensions.load_from_namespace(
        response.annotations, key_prefix="cost_"
    )


@pytest.mark.asyncio
async def test_cost_model_request_costs_survive_response_roundtrip():
    """Per-response costs saved by CostModel persist through InvocationResponse to_json/from_json.

    This is the behavior that regressed when response serialization moved to ``asdict`` (which
    drops loose attributes): costs stored in ``annotations`` must round-trip so a saved Result can
    be reloaded with its per-request costs intact.
    """
    from llmeter.callbacks.cost.dimensions import InputTokens, OutputTokens

    model = CostModel(
        request_dims=[
            InputTokens(price_per_million=3.0),
            OutputTokens(price_per_million=15.0),
        ]
    )
    response = InvocationResponse(
        response_text="hello", num_tokens_input=1000, num_tokens_output=500
    )
    await model.after_invoke(response)
    assert response.annotations["cost_total"] == pytest.approx(0.003 + 0.0075)

    # Round-trip through JSON (what gets written to responses.jsonl)
    restored = InvocationResponse.from_json(response.to_json())
    assert restored.annotations == response.annotations

    # ...and the cost model can read the costs back off the restored response
    reloaded = CalculatedCostWithDimensions.load_from_namespace(
        restored.annotations, key_prefix="cost_"
    )
    assert reloaded["InputTokens"] == pytest.approx(0.003)
    assert reloaded["OutputTokens"] == pytest.approx(0.0075)


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

    run_mock = NonCallableMock()
    assert await model.before_run(run_mock) is None
    results_mock = NonCallableMock()
    results_mock.additional_metrics_for_aggregation = None
    results_mock.responses = []
    assert await model.after_run(results_mock) is None
    assert results_mock.cost_total == 5000
    assert results_mock.cost_Mock == 5000  # Class name is the default dimension name

    # Check calculate_* fn produces same result as callback:
    await model.before_run(run_mock) is None
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
    run_mock = NonCallableMock()
    await model.before_run(run_mock)
    responses = [
        InvocationResponse(response_text="a"),
        InvocationResponse(response_text="b"),
        InvocationResponse(response_text="c"),
    ]
    for r in responses:
        await model.after_invoke(r)
    results_mock = NonCallableMock()
    update_contrib_stats_mock = Mock()
    results_mock._update_contributed_stats = update_contrib_stats_mock
    results_mock.responses = responses
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
