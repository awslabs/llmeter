# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost modelling callback for LLMeter test runs."""

from ...endpoints.base import InvocationResponse
from ...results import Result
from ...runner import _RunConfig
from ..base import Callback
from .dimensions import IRequestCostDimension, IRunCostDimension
from .results import CalculatedCostWithDimensions


class CostModel(Callback):
    """Model costs of (test runs of) Foundation Models.

    A Cost Model is composed of pricing *dimensions* applied at the individual request
    level (like ``InputTokens``) or at the overall run level (like ``EndpointTime``).
    """

    def __init__(
        self,
        request_dims: (
            dict[str, IRequestCostDimension]
            | list[IRequestCostDimension]
            | None
        ) = None,
        run_dims: (
            dict[str, IRunCostDimension]
            | list[IRunCostDimension]
            | None
        ) = None,
    ):
        all_dims: dict[str, IRequestCostDimension | IRunCostDimension] = {}
        self.request_dims: dict[str, IRequestCostDimension] = {}
        self.run_dims: dict[str, IRunCostDimension] = {}

        if request_dims is not None:
            for name, dim in (
                request_dims.items()
                if isinstance(request_dims, dict)
                else ((d.__class__.__name__, d) for d in request_dims)
            ):
                if name in all_dims:
                    raise ValueError(
                        f"Duplicate cost dimension name '{name}': "
                        f"Got both {dim} and {all_dims[name]}"
                    )
                all_dims[name] = dim
                self.request_dims[name] = dim

        if run_dims is not None:
            for name, dim in (
                run_dims.items()
                if isinstance(run_dims, dict)
                else ((d.__class__.__name__, d) for d in run_dims)
            ):
                if name in all_dims:
                    raise ValueError(
                        f"Duplicate cost dimension name '{name}': "
                        f"Got both {dim} and {all_dims[name]}"
                    )
                all_dims[name] = dim
                self.run_dims[name] = dim

    # ------------------------------------------------------------------
    # Callback lifecycle hooks
    # ------------------------------------------------------------------

    async def calculate_request_cost(
        self, response: InvocationResponse, save: bool = False
    ) -> CalculatedCostWithDimensions:
        dim_costs = CalculatedCostWithDimensions(
            **{
                name: await dim.calculate(response)
                for name, dim in self.request_dims.items()
            }
        )
        if save:
            dim_costs.save_on_namespace(response, key_prefix="cost_")
        return dim_costs

    async def calculate_run_cost(
        self,
        result: Result,
        recalculate_request_costs: bool = True,
        save: bool = False,
    ) -> CalculatedCostWithDimensions:
        run_cost = CalculatedCostWithDimensions(
            **{
                name: await dim.calculate(result)
                for name, dim in self.run_dims.items()
            }
        )
        if recalculate_request_costs:
            resp_costs = [
                await self.calculate_request_cost(r, save=save)
                for r in result.responses
            ]
        else:
            resp_costs = list(filter(
                lambda c: c,
                (
                    CalculatedCostWithDimensions.load_from_namespace(
                        r, key_prefix="cost_"
                    )
                    for r in result.responses
                ),
            ))
        if len(resp_costs):
            run_cost.merge(sum(resp_costs))  # type: ignore
        if save:
            run_cost.save_on_namespace(result, key_prefix="cost_")
            stats = CalculatedCostWithDimensions.summary_statistics(
                resp_costs,
                key_prefix="cost_",
                key_dim_name_suffix="_per_request",
                key_total_name_and_suffix="per_request",
            )
            run_cost.save_on_namespace(stats, key_prefix="cost_")
            result._update_contributed_stats(stats)
        return run_cost

    async def before_invoke(self, payload: dict) -> None:
        pass

    async def after_invoke(self, response: InvocationResponse) -> None:
        await self.calculate_request_cost(response, save=True)

    async def before_run(self, run_config: _RunConfig) -> None:
        for dim in self.run_dims.values():
            await dim.before_run_start(run_config)

    async def after_run(self, result: Result) -> None:
        await self.calculate_run_cost(
            result, recalculate_request_costs=False, save=True
        )
