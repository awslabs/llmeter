# Python Built-Ins:
from abc import abstractmethod
from dataclasses import dataclass
from math import ceil
from typing import Optional

# Local Dependencies:
from ...endpoints.base import InvocationResponse
from ...results import Result
from ...runner import Runner
from .serde import ISerializable, JSONableBase


class IRequestCostDimension(ISerializable):
    """Interface for one dimension of a per-request cost model

    Per-request cost components are calculated independently for each invocation in a test run, and
    can be used to model factors like cost-per-request, cost-per-input-tokens,
    cost-per-request-duration, etc. They're typically most relevant for serverless deployments like
    Amazon Bedrock, or estimating duration-based execution costs for AWS Lambda functions.
    """

    name: str

    async def calculate(self, response: InvocationResponse) -> Optional[float]:
        """Calculate (this component of) the cost for an individual request/response"""
        ...


class RequestCostDimensionBase(JSONableBase):
    """Optional base class for implementing per-request cost model dimensions

    This class provides a default implementation of `ISerializable`, defaults `.name` to the
    instantiated class name, and sets up an abstract method for `calculate()`. It's fine if you
    don't want to use it - just be sure to fully implement `IRequestCostDimension`!
    """

    name: str

    def __init__(self, name: str | None):
        self.name = name or self.__class__.__name__

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

    name: str

    def __init__(self, name: str | None):
        self.name = name or self.__class__.__name__

    async def before_run_start(self, runner: Runner) -> None:
        """Function called to notify the cost component that a test run is about to start

        This method is called before the test run starts, and can be used to perform any
        initialization or setup required for the cost component.
        """
        ...

    async def calculate(self, result: Result) -> Optional[float]:
        """Calculate (this component of) the cost for a completed test run

        Dimensions that depend on `before_run_start` being called to return an accurate result
        should throw an error if this was not done. Dimensions that only need `calculate()` should
        silently ignore if `before_run_start` was not called.
        """
        ...


class RunCostDimensionBase(JSONableBase):
    """Optional base class for implementing per-run cost model dimensions

    This class provides a default implementation of `ISerializable`, a default empty
    `before_run_path` implementation, and abstract methods for the other requirements of the
    `IRunCostDimension` protocol. It's fine if you don't want to use it - just make sure you
    fully implement `IRunCostDimension`!
    """

    async def before_run_start(self, runner: Runner) -> None:
        """Function called to notify the cost component that a test run is about to start

        This method is called before the test run starts, and can be used to perform any
        initialization or setup required for the cost component. The default implementation is a
        pass.
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
class CostPerInputToken(RequestCostDimensionBase):
    """Request cost dimension to model per-input-token costs with a flat charge rate

    Args:
        rate: Charge applied per input (prompt) token to the Foundation Model
    """

    name: str
    rate: float

    async def calculate(self, req: InvocationResponse) -> Optional[float]:
        if req.num_tokens_input is None:
            return None
        return req.num_tokens_input * self.rate


@dataclass
class CostPerOutputToken(RequestCostDimensionBase):
    """Request cost dimension to model per-output-token costs with a flat charge rate

    Args:
        rate: Charge applied per output (completion) token from the Foundation Model
    """

    name: str
    rate: float

    async def calculate(self, req: InvocationResponse) -> Optional[float]:
        if req.num_tokens_output is None:
            return None
        return req.num_tokens_output * self.rate


@dataclass
class CostPerHour(RunCostDimensionBase):
    """Run cost dimension to model per-deployment-hour costs with a flat charge rate

    Args:
        rate: Charge applied per hour a test run takes
        granularity_secs: Minimum number of seconds billed per increment (Default 1)
    """

    name: str
    rate: float
    granularity_secs: float = 1

    async def calculate(self, result: Result) -> Optional[float]:
        if result.total_test_time is None:
            return None
        billable_secs = (
            ceil(result.total_test_time / self.granularity_secs) * self.granularity_secs
        )
        return billable_secs * self.rate / 3600
