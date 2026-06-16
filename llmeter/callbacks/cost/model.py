# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost modelling callback for LLMeter test runs."""

import importlib
import json

from upath.types import ReadablePathLike, WritablePathLike

from ...endpoints.base import InvocationResponse
from ...json_utils import llmeter_default_serializer
from ...results import Result
from ...runner import _RunConfig
from ...serialization import dump_object, load_object
from ...utils import ensure_path
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
    # Serialization helpers (to_dict/from_dict for legacy compat)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to dict with _type tags (legacy-compatible format)."""
        return {
            "_type": "CostModel",
            "request_dims": {
                name: dim.to_dict()
                for name, dim in self.request_dims.items()
            },
            "run_dims": {
                name: dim.to_dict()
                for name, dim in self.run_dims.items()
            },
        }

    def to_json(self, indent: int | None = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, raw: dict, **kwargs) -> "CostModel":
        """Load from dict (supports both new _class/_state and legacy _type)."""
        clean = {k: v for k, v in raw.items() if k != "_type"}

        # Check if dimensions use legacy _type format and convert
        dim_module = importlib.import_module(
            "llmeter.callbacks.cost.dimensions"
        )
        for key in ("request_dims", "run_dims"):
            dims = clean.get(key, {})
            if isinstance(dims, dict):
                converted = {}
                for name, d in dims.items():
                    if isinstance(d, dict) and "_type" in d and "_class" not in d:
                        # Legacy format → convert to object directly
                        type_name = d.pop("_type")
                        dim_cls = getattr(dim_module, type_name, None)
                        if dim_cls is None:
                            providers = importlib.import_module(
                                "llmeter.callbacks.cost.providers.sagemaker"
                            )
                            dim_cls = getattr(providers, type_name, None)
                        if dim_cls is None:
                            raise ValueError(
                                f"Unknown dimension type: {type_name!r}"
                            )
                        converted[name] = dim_cls(**d)
                    elif isinstance(d, dict) and "_class" in d:
                        converted[name] = load_object(d)
                    else:
                        converted[name] = d
                clean[key] = converted

        return cls(**clean)

    @classmethod
    def from_json(cls, json_string: str) -> "CostModel":
        return cls.from_dict(json.loads(json_string))

    @classmethod
    def from_file(cls, path: ReadablePathLike) -> "CostModel":
        path = ensure_path(path)
        with path.open("r") as f:
            return cls.from_json(f.read())

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

    def save_to_file(self, path: WritablePathLike) -> None:
        path = ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dump_object(self)
        with path.open("w") as f:
            json.dump(data, f, indent=4, default=llmeter_default_serializer)

    @classmethod
    def _load_from_file(cls, path: ReadablePathLike) -> "CostModel":
        return cls.from_file(path)
