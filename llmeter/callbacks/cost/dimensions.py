# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classes defining different components of cost

A "dimension" is one aspect of the pricing for a deployed Foundation Model or application. In
general, multiple factors are likely to contribute to the total cost of FMs under test: For
example, an API may charge separate rates for input vs output token counts; or a self-managed
cloud deployment may carry per-hour charges for compute, as well as network bandwidth charges.

Here we provide implementations for some common cost dimensions, and define base classes you can
use to bring customized cost dimensions for your own cost models.
"""

# Python Built-Ins:
from abc import abstractmethod, ABC
from dataclasses import dataclass
from math import ceil
from typing import Optional

# Local Dependencies:
from ...endpoints.base import InvocationResponse
from ...results import Result
from ...runner import _RunConfig
from .serde import ISerializable, JSONableBase


class IRequestCostDimension(ISerializable):
    """Interface for one dimension of a per-request cost model

    Per-request cost components are calculated independently for each invocation in a test run, and
    can be used to model factors like cost-per-request, cost-per-input-tokens,
    cost-per-request-duration, etc. They're typically most relevant for serverless deployments like
    Amazon Bedrock, or estimating duration-based execution costs for AWS Lambda functions.
    """

    async def calculate(self, response: InvocationResponse) -> Optional[float]:
        """Calculate (this component of) the cost for an individual request/response"""
        ...


class RequestCostDimensionBase(ABC, JSONableBase):
    """Base class for implementing per-request cost model dimensions

    This class provides a default implementation of `ISerializable` and sets up an abstract method
    for `calculate()`. It's fine if you don't want to derive from it directly - just be sure to
    fully implement `IRequestCostDimension`!
    """

    @abstractmethod
    async def calculate(self, response: InvocationResponse) -> Optional[float]:
        """Calculate (this component of) the cost for an individual request/response"""
        raise NotImplementedError(
            "Children of RequestCostDimensionBase must implement `calculate()`! At: %s"
            % (self.__class__,)
        )


class IRunCostDimension(ISerializable):
    """Interface for one dimension of a per-Run cost model

    Per-run cost components are notified before the start of a test run via `before_run_start()`,
    and then requested to `calculate()` at the end of the run. They're most relevant for
    provisioned-infrastructure based deployments like Amazon SageMaker, where factors like a
    (request-independent) cost-per-hour are important.
    """

    async def before_run_start(self, run_config: _RunConfig) -> None:
        """Function called to notify the cost component that a test run is about to start

        This method is called before the test run starts, and can be used to perform any
        initialization or setup required for the cost component. In general, we assume a dimension
        instance may be re-used for multiple test runs, but only one run at a time: Meaning
        `before_run_start()` should not be called again before `calculate()` is called for the
        previous run.
        """
        ...

    async def calculate(self, result: Result) -> Optional[float]:
        """Calculate (this component of) the cost for a completed test run

        Dimensions that depend on `before_run_start` being called to return an accurate result
        should throw an error if this was not done. Dimensions that only need `calculate()` should
        silently ignore if `before_run_start` was not called.
        """
        ...


class RunCostDimensionBase(ABC, JSONableBase):
    """Base class for implementing per-run cost model dimensions

    This class provides a default implementation of `ISerializable`, a default empty
    `before_run_start` implementation, and abstract methods for the other requirements of the
    `IRunCostDimension` protocol. It's fine if you don't want to derive from it directly - just
    make sure you fully implement `IRunCostDimension`!
    """

    async def before_run_start(self, run_config: _RunConfig) -> None:
        """Function called to notify the cost component that a test run is about to start

        This method is called before the test run starts, and can be used to perform any
        initialization or setup required for the cost component.  In general, we assume a dimension
        instance may be re-used for multiple test runs, but only one run at a time: Meaning
        `before_run_start()` should not be called again before `calculate()` is called for the
        previous run.

        The default implementation is a pass.
        """
        pass

    @abstractmethod
    async def calculate(self, result: Result) -> Optional[float]:
        """Calculate (this component of) the cost for a completed test run

        Dimensions that depend on `before_run_start` being called to return an accurate result
        should throw an error if this was not done. Dimensions that only need `calculate()` should
        silently ignore if `before_run_start` was not called.
        """
        raise NotImplementedError(
            "Children of RunCostDimensionBase must implement `calculate()`! At: %s"
            % (self.__class__,)
        )


@dataclass
class InputTokens(RequestCostDimensionBase):
    """Request cost dimension to model per-input-token costs with a flat charge rate

    Args:
        rate_per_million: Charge applied per million input (prompt) token to the Foundation Model
        granularity_tokens: Minimum number of tokens billed per increment (Default 1)
    """

    price_per_million: float
    granularity: int = 1

    async def calculate(self, req: InvocationResponse) -> Optional[float]:
        if req.num_tokens_input is None:
            return None
        billable_tokens = (
            ceil(req.num_tokens_input / self.granularity) * self.granularity
        )
        return billable_tokens * self.price_per_million / 1000000


@dataclass
class OutputTokens(RequestCostDimensionBase):
    """Request cost dimension to model per-output-token costs with a flat charge rate

    Args:
        rate_per_million: Charge per million output (completion) token from the Foundation Model
        granularity_tokens: Minimum number of tokens billed per increment (Default 1)
    """

    price_per_million: float
    granularity: int = 1

    async def calculate(self, req: InvocationResponse) -> Optional[float]:
        if req.num_tokens_output is None:
            return None
        billable_tokens = (
            ceil(req.num_tokens_output / self.granularity) * self.granularity
        )
        return billable_tokens * self.price_per_million / 1000000


@dataclass
class EndpointTime(RunCostDimensionBase):
    """Run cost dimension to model per-deployment-hour costs with a flat charge rate

    Args:
        rate: Charge applied per hour a test run takes
        granularity_secs: Minimum number of seconds billed per increment (Default 1)
    """

    price_per_hour: float
    granularity_secs: float = 1

    async def calculate(self, result: Result) -> Optional[float]:
        if result.total_test_time is None:
            return None
        billable_secs = (
            ceil(result.total_test_time / self.granularity_secs) * self.granularity_secs
        )
        return billable_secs * self.price_per_hour / 3600
