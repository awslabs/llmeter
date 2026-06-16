# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classes defining different components of cost

A "dimension" is one aspect of the pricing for a deployed Foundation Model or application.
Here we provide implementations for some common cost dimensions, and define base classes you
can use to bring customized cost dimensions for your own cost models.
"""

# Python Built-Ins:
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from typing import Optional, Protocol

# Local Dependencies:
from ...endpoints.base import InvocationResponse
from ...results import Result
from ...runner import _RunConfig
from ...serialization import Serializable


class IRequestCostDimension(Protocol):
    """Interface for one dimension of a per-request cost model."""

    async def calculate(self, response: InvocationResponse) -> Optional[float]: ...


class IRunCostDimension(Protocol):
    """Interface for one dimension of a per-Run cost model."""

    async def before_run_start(self, run_config: _RunConfig) -> None: ...
    async def calculate(self, result: Result) -> Optional[float]: ...


class RequestCostDimensionBase(Serializable, ABC):
    """Base class for per-request cost dimensions.

    Inherits ``__getstate__``/``__setstate__`` from :class:`~llmeter.serialization.Serializable`.
    Subclasses just declare fields and implement ``calculate()``.
    """

    @abstractmethod
    async def calculate(self, response: InvocationResponse) -> Optional[float]:
        raise NotImplementedError


class RunCostDimensionBase(Serializable, ABC):
    """Base class for per-run cost dimensions.

    Inherits ``__getstate__``/``__setstate__`` from :class:`~llmeter.serialization.Serializable`.
    """

    async def before_run_start(self, run_config: _RunConfig) -> None:
        """Called before a test run starts. Default is a no-op."""
        pass

    @abstractmethod
    async def calculate(self, result: Result) -> Optional[float]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete dimension implementations
# ---------------------------------------------------------------------------


@dataclass
class InputTokens(RequestCostDimensionBase):
    """Request cost dimension: per-input-token costs with a flat charge rate.

    Args:
        price_per_million: Charge per million input (prompt) tokens.
        granularity: Minimum tokens billed per increment (Default 1).
    """

    price_per_million: float
    granularity: int = 1

    async def calculate(self, req: InvocationResponse) -> Optional[float]:
        if req.num_tokens_input is None:
            return None
        billable = ceil(req.num_tokens_input / self.granularity) * self.granularity
        return billable * self.price_per_million / 1_000_000


@dataclass
class OutputTokens(RequestCostDimensionBase):
    """Request cost dimension: per-output-token costs with a flat charge rate.

    Args:
        price_per_million: Charge per million output (completion) tokens.
        granularity: Minimum tokens billed per increment (Default 1).
    """

    price_per_million: float
    granularity: int = 1

    async def calculate(self, req: InvocationResponse) -> Optional[float]:
        if req.num_tokens_output is None:
            return None
        billable = ceil(req.num_tokens_output / self.granularity) * self.granularity
        return billable * self.price_per_million / 1_000_000


@dataclass
class EndpointTime(RunCostDimensionBase):
    """Run cost dimension: per-deployment-hour costs with a flat charge rate.

    Args:
        price_per_hour: Charge per hour a test run takes.
        granularity_secs: Minimum seconds billed per increment (Default 1).
    """

    price_per_hour: float
    granularity_secs: float = 1

    async def calculate(self, result: Result) -> Optional[float]:
        if result.total_test_time is None:
            return None
        billable = (
            ceil(result.total_test_time / self.granularity_secs)
            * self.granularity_secs
        )
        return billable * self.price_per_hour / 3600
