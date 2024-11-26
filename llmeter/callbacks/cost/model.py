# Python Built-Ins:
from dataclasses import dataclass, field
import importlib
from typing import Literal

# Local Dependencies:
from ...endpoints.base import InvocationResponse
from ...results import Result
from ...runner import Runner
from ..base import Callback
from .dimensions import IRequestCostDimension, IRunCostDimension
from .results import CalculatedCostDimension, CalculatedCostWithDimensions
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

    Args:
        request_dims: Dimensions of request-level cost (for example, charges by number of input or
            output tokens)
        run_dims: Dimensions of run-level cost (for example, per-hour charges for an FM endpoint
            being available and used in a run)
    """

    request_dims: list[IRequestCostDimension] = field(default_factory=list)
    run_dims: list[IRunCostDimension] = field(default_factory=list)

    async def before_run(self, runner: Runner) -> None:
        """Initialize state for all run-level cost dimensions in the model"""
        for dim in self.run_dims:
            dim.before_run_start(runner)

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
        total = CalculatedCostWithDimensions(
            [
                CalculatedCostDimension(
                    name=dim.name,
                    cost=dim.calculate(response),
                )
                for dim in self.request_dims
            ]
        )
        if save:
            response.cost = total
        return total

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
            [
                CalculatedCostDimension(
                    name=dim.name,
                    cost=dim.calculate(result),
                )
                for dim in self.run_dims
            ]
        )
        if include_request_costs == True:
            resp_costs = [
                r.cost
                for r in result.responses
                if hasattr(r, "cost") and r.cost is not None
            ]
            # Need to check because sum([]) = 0, which is not what we want:
            if len(resp_costs):
                run_cost.merge(sum(resp_costs))
        elif include_request_costs == "recalculate":
            run_cost.merge(
                # Surrounding [] needed because async generator is not iterable for sum():
                sum(
                    [
                        await self.calculate_request_cost(r, save=save)
                        for r in result.responses
                    ]
                )
            )
        elif include_request_costs != False:
            raise ValueError(
                "Invalid value for include_request_costs: Must be True (to include request-level "
                "costs in the total), False (to exclude them), or 'recalculate' (to re-calculate "
                "and include request-level costs)"
            )
        if save:
            result.cost = run_cost
        return run_cost

    async def after_invoke(self, response: InvocationResponse) -> None:
        """LLMeter Callback.after_invoke hook

        Calls calculate_request_cost() with `save=True` to save the cost on the InvocationRespnose.
        """
        await self.calculate_request_cost(response, save=True)

    async def after_run(self, result: Result) -> None:
        """LLMeter Callback.after_run hook

        Calls calculate_run_cost() with `save=True` to save the cost on the Result.
        """
        await self.calculate_run_cost(result, save=True)

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
                raw_args[key] = [
                    from_dict_with_class_map(d, class_map=dim_classes)
                    for d in raw_args[key]
                ]
        return super().from_dict(raw_args, alt_classes=alt_classes, **kwargs)

    @classmethod
    def _load_from_file(cls, path: str):
        """Load the cost model (including all dimensions) from a JSON file"""
        with open(path, "r") as f:
            return cls.from_json(f.read())
