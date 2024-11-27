# Python Built-Ins:
from dataclasses import dataclass, field
import importlib
from itertools import chain
from typing import Literal

# Local Dependencies:
from ...endpoints.base import InvocationResponse
from ...results import Result
from ...runner import Runner
from ..base import Callback
from .dimensions import IRequestCostDimension, IRunCostDimension
from .results import CalculatedCostWithDimensions
from .serde import from_dict_with_class_map, JSONableBase


@dataclass
class CostModel(JSONableBase, Callback):
    """Model costs of (test runs of) Foundation Models

    A Cost Model is composed of whatever pricing *"dimensions"* are relevant for your FM deployment
    and use-case, which may be applied at the individual request level (like
    `llmeter.callbacks.cost.dimensions.CostPerInputToken`), or at the overall deployment / test run
    level (like `llmeter.callbacks.cost.dimensions.CostPerHour`).

    With a Cost Model defined, you can explicitly `calculate_request_cost(...)` on an
    `InvocationResponse` to estimate costs for a specific request/response;
    `calculate_run_cost(...)` on a `Result` to estimate costs for a test run; or pass the model as
    a Callback when running an LLMeter Run or Experiment, to annotate the results automatically.
    """

    request_dims: dict[str, IRequestCostDimension] = field(default_factory=dict)
    run_dims: dict[str, IRunCostDimension] = field(default_factory=dict)

    def __init__(
        self,
        request_dims: dict[str, IRequestCostDimension]
        | list[IRequestCostDimension]
        | None = None,
        run_dims: dict[str, IRunCostDimension] | list[IRunCostDimension] | None = None,
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
        if request_dims is None:
            self.request_dims = {}
        elif not isinstance(request_dims, dict):
            self.request_dims = {dim.__class__.__name__: dim for dim in request_dims}
        else:
            self.request_dims = request_dims

        if run_dims is None:
            self.run_dims = {}
        elif not isinstance(run_dims, dict):
            self.run_dims = {dim.__class__.__name__: dim for dim in run_dims}
        else:
            self.run_dims = run_dims

        # Validate no overlapping names:
        all_dims = {}
        for name, dim in chain(self.request_dims.items(), self.run_dims.items()):
            if name in all_dims:
                raise ValueError(
                    f"Duplicate cost dimension name '{name}': Got both {dim} and {all_dims[name]}"
                )
            all_dims[name] = dim

    async def before_run(self, runner: Runner) -> None:
        """Initialize state for all run-level cost dimensions in the model"""
        for dim in self.run_dims.values():
            await dim.before_run_start(runner)

    async def calculate_request_cost(
        self,
        response: InvocationResponse,
        save: bool = False,
    ) -> CalculatedCostWithDimensions:
        """Calculate the costs of a single FM invocation (excluding any session-level costs)

        Args:
            response: The InvocationResponse to estimate costs for
            save: Set `True` to also store the result in `response.cost`, in addition to returning
                it. Defaults to `False`
        """
        dim_costs = CalculatedCostWithDimensions(
            **{
                name: await dim.calculate(response)
                for name, dim in self.request_dims.items()
            }
        )
        if save:
            dim_costs.save_on_namespace(response, key_prefix="cost_")

        return dim_costs

    # @staticmethod
    # def wrap_result_stat_calculator(result: Result) -> None:
    #     """Patch the calculate_stats() method on a Result to include cost summary stats"""
    #     super_method = result.calculate_stats
    #     def calculate_stats(result: Result):
    #         stats = super_method()
    #         dim_summary_stats = CalculatedCostWithDimensions.summary_statistics([
    #             CalculatedCostWithDimensions.load_from_namespace(resp) for resp in result.responses
    #         ])
    #         for dim_name, dim_summary_stats in dim_summary_stats.items():
    #             for stat_name, val in dim_summary_stats.items():
    #                 # "total_per_request" could be confusing, so we omit 'total' here:
    #                 target_attr = (
    #                     f"cost_per_request-{stat_name}"
    #                     if dim_name == "total"
    #                     else f"cost_{dim_name}_per_request-{stat_name}"
    #                 )
    #                 stats[target_attr] = val
    #         return stats
    #     result.calculate_stats = calculate_stats

    async def calculate_run_cost(
        self,
        result: Result,
        include_request_costs: bool | Literal["recalculate"] = True,
        save: bool = False,
    ) -> CalculatedCostWithDimensions:
        """Calculate the run-level costs of a test (including any request-level costs)

        NOTE: Depending on the types of `run_dims` in your model, this may throw an error if
        `before_run` was not called first to initialise the state.

        Args:
            result: An LLMeter Run result
            include_request_costs: Set `True` to include request-level costs in the total, `False`
                to exclude them, or `'recalculate'` to re-calculate request-level costs (for
                example when running a new cost model against a previous Run Result). Defaults
                to `True`.
            save: Set `True` to also store the result in `result.cost`, in addition to returning
                it. Defaults to `False`.
        """
        run_cost = CalculatedCostWithDimensions(
            **{name: await dim.calculate(result) for name, dim in self.run_dims.items()}
        )

        resp_costs: list[CalculatedCostWithDimensions] | None = None
        if include_request_costs == True:  # noqa: E712
            resp_costs = list(
                filter(
                    lambda c: c,  # Skip responses where no cost data was found at all
                    (
                        CalculatedCostWithDimensions.load_from_namespace(
                            r, key_prefix="cost_"
                        )
                        for r in result.responses
                    ),
                ),
            )
            if len(
                resp_costs
            ):  # Need to check because sum([]) = 0, which is not what we want
                run_cost.merge(sum(resp_costs))  # type: ignore
        elif include_request_costs == "recalculate":
            resp_costs = [
                await self.calculate_request_cost(r, save=save)
                for r in result.responses
            ]
            if len(
                resp_costs
            ):  # Need to check because sum([]) = 0, which is not what we want
                run_cost.merge(sum(resp_costs))  # type: ignore
        elif include_request_costs != False:  # noqa: E712
            raise ValueError(
                "Invalid value for include_request_costs: Must be True (to include request-level "
                "costs in the total), False (to exclude them), or 'recalculate' (to re-calculate "
                "and include request-level costs)"
            )
        if save:
            # Save the overall run cost and breakdown:
            run_cost.save_on_namespace(result, key_prefix="cost_")

        return run_cost

    async def after_invoke(self, response: InvocationResponse) -> None:
        """LLMeter Callback.after_invoke hook

        Calls calculate_request_cost() with `save=True` to save the cost on the InvocationResponse.
        """
        await self.calculate_request_cost(response, save=True)

    async def after_run(self, result: Result) -> None:
        """LLMeter Callback.after_run hook

        Calls calculate_run_cost() with `save=True` to save the cost on the Result.
        """
        await self.calculate_run_cost(result, save=True)
        result._contributed_stats.update(self.calculate_stats(result))
        result._contributed_stats.update(
            {k: v for k, v in result.__dict__.items() if k.startswith("cost")}
        )

    def calculate_stats(self, result: Result):
        stats = {}
        dim_summary_stats = CalculatedCostWithDimensions.summary_statistics(
            [
                CalculatedCostWithDimensions.load_from_namespace(
                    resp, key_prefix="cost"
                )
                for resp in result.responses
            ]
        )
        for dim_name, dim_summary_stats in dim_summary_stats.items():
            for stat_name, val in dim_summary_stats.items():
                # "total_per_request" could be confusing, so we omit 'total' here:
                target_attr = (
                    f"cost_per_request-{stat_name}"
                    if dim_name == "total"
                    else f"cost{dim_name}_per_request-{stat_name}"
                    # else f"{dim_name}_per_request-{stat_name}"
                )
                stats[target_attr] = val
        return stats

    def save_to_file(self, path: str) -> None:
        """Save the cost model (including all dimensions) to a JSON file"""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, raw: dict, alt_classes: dict = {}, **kwargs) -> "CostModel":
        dim_classes = {
            **importlib.import_module("llmeter.callbacks.cost.dimensions").__dict__,
            **alt_classes,
        }
        raw_args = {**raw}
        for key in ("request_dims", "run_dims"):
            if key in raw_args:
                raw_args[key] = {
                    name: from_dict_with_class_map(d, class_map=dim_classes)
                    for name, d in raw_args[key].items()
                }
        return super().from_dict(raw_args, alt_classes=alt_classes, **kwargs)

    @classmethod
    def _load_from_file(cls, path: str):
        """Load the cost model (including all dimensions) from a JSON file"""
        with open(path, "r") as f:
            return cls.from_json(f.read())
