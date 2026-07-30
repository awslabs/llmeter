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
    """Model costs of (test runs of) Foundation Models

    A Cost Model is composed of whatever pricing *"dimensions"* are relevant for your FM deployment
    and use-case, which may be applied at the individual request level (like
    `llmeter.callbacks.cost.dimensions.InputTokens`), or at the overall deployment / test run level
    (like `llmeter.callbacks.cost.dimensions.EndpointTime`).

    With a Cost Model defined, you can explicitly `calculate_request_cost(...)` on an
    `InvocationResponse` to estimate costs for a specific request/response;
    `calculate_run_cost(...)` on a `Result` to estimate costs for a test run; or pass the model as
    a Callback when running an LLMeter Run or Experiment, to annotate the results automatically.
    """

    def __init__(
        self,
        request_dims: (
            dict[str, IRequestCostDimension] | list[IRequestCostDimension] | None
        ) = None,
        run_dims: (
            dict[str, IRunCostDimension] | list[IRunCostDimension] | None
        ) = None,
    ):
        """Create a CostModel

        Args:
            request_dims: Dimensions of request-level cost (for example, charges by number of input
                or output tokens). If provided as a dict, the keys will be used as the name of each
                dimension. If provided as a list, each dimension's class name will be used as its
                default name. An error will be thrown if any two dimensions (including run_dims)
                have the same name.
            run_dims: Dimensions of run-level cost (for example, per-hour charges for an FM endpoint
                being available and used in a run). If provided as a dict, the keys will be used as
                the name of each dimension. If provided as a list, each dimension's class name will
                be used as its default name. An error will be thrown if any two dimensions
                (including request_dims) have the same name.
        """
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
        """Calculate the costs of a single FM invocation (excluding any session-level costs)

        Args:
            response: The InvocationResponse to estimate costs for
            save: Set `True` to also store the result in `response.annotations` (under `cost_`
                prefixed keys), in addition to returning it. Because `annotations` is a persisted
                field, saved costs survive `Result` save/load. Defaults to `False`
        """
        dim_costs = CalculatedCostWithDimensions(
            **{
                name: await dim.calculate(response)
                for name, dim in self.request_dims.items()
            }
        )
        if save:
            dim_costs.save_on_namespace(response.annotations, key_prefix="cost_")
        return dim_costs

    async def calculate_run_cost(
        self,
        result: Result,
        recalculate_request_costs: bool = True,
        save: bool = False,
    ) -> CalculatedCostWithDimensions:
        """Calculate the run-level costs of a test (including any request-level costs)

        NOTE: Depending on the types of `run_dims` in your model, this may throw an error if
        `before_run` was not called first to initialise the state.

        Args:
            result: An LLMeter Run result
            recalculate_request_costs: Set `False` if your `result.responses` have already been
                annotated with costs in line with the current cost model. By default (`True`),
                the costs for each response will be re-calculated.
            save: Set `True` to also store the result in `result.cost`, in addition to returning
                it. Defaults to `False`.
        """
        run_cost = CalculatedCostWithDimensions(
            **{name: await dim.calculate(result) for name, dim in self.run_dims.items()}
        )
        if recalculate_request_costs:
            resp_costs = [
                await self.calculate_request_cost(r, save=save)
                for r in result.responses
            ]
        else:
            resp_costs = list(
                filter(
                    lambda c: c,  # Skip responses where no cost data was found at all
                    (
                        CalculatedCostWithDimensions.load_from_namespace(
                            r.annotations, key_prefix="cost_"
                        )
                        for r in result.responses
                    ),
                )
            )
        if len(resp_costs):
            # Merge the total request-level costs into the run-level costs:
            # (Unless requests is empty, because sum([])=0 and not a CalculatedCostWithDimensions)
            run_cost.merge(sum(resp_costs))  # type: ignore
        if save:
            # Save the overall run cost and breakdown on the main result object:
            run_cost.save_on_namespace(result, key_prefix="cost_")
            # Contribute both 1/ the summary stats of request-level costs, and 2/ the overall run
            # cost+breakdown, to result.stats:
            stats = CalculatedCostWithDimensions.summary_statistics(
                resp_costs,
                key_prefix="cost_",
                key_dim_name_suffix="_per_request",
                # cost_total_per_request would be confusing, so skip 'total':
                key_total_name_and_suffix="per_request",
            )
            run_cost.save_on_namespace(stats, key_prefix="cost_")
            result._update_contributed_stats(stats)
        return run_cost

    async def after_invoke(self, response: InvocationResponse) -> None:
        """LLMeter Callback.after_invoke hook

        Calls calculate_request_cost() with `save=True` to save the per-request cost into
        `response.annotations` (under `cost_` prefixed keys), so it persists with the response.
        """
        await self.calculate_request_cost(response, save=True)

    async def before_run(self, run_config: _RunConfig) -> None:
        """Initialize state for all run-level cost dimensions in the model"""
        for dim in self.run_dims.values():
            await dim.before_run_start(run_config)

    async def after_run(self, result: Result) -> None:
        """LLMeter Callback.after_run hook

        Calls calculate_run_cost() with `save=True` to save the cost on the Result.
        """
        await self.calculate_run_cost(
            result, recalculate_request_costs=False, save=True
        )
