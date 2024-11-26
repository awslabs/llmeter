"""Output / result structure definitions for cost modelling

These structures represent the *outputs* of cost estimates, and are referenced by low-level LLMeter
classes like `InvocationResponse` and `Result`.
"""

# Python Built-Ins:
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

# Local Dependencies:
from .serde import JSONableBase


@dataclass
class CalculatedCostDimension(JSONableBase):
    """Result of one component of a cost estimate

    Attributes:
        name (str): The name of the dimension. For example, "CostPerInputToken"
        cost (float): The cost incurred in this dimension (which may be `None` if the source cost
            model was unable to calculate it)
    """

    name: str
    cost: float | None


@dataclass
class CalculatedCostWithDimensions(JSONableBase):
    """Result of a cost estimate (composed of multiple cost dimensions)

    Attributes:
        cost (float): The total/overall cost calculated
        dims: (list[CalculatedCostDimension]): The underlying dimensions that make up the total
            cost (for example, the costs for input tokens vs for output tokens).
    """

    dims: list[CalculatedCostDimension] = field(default_factory=list)

    @property
    def total(self) -> float:
        """The total/overall cost over all dimensions"""
        if any(d.cost is None for d in self.dims):
            return None
        else:
            return sum(d.cost for d in self.dims)

    def __add__(
        self, other: CalculatedCostDimension | CalculatedCostWithDimensions | Literal[0]
    ) -> CalculatedCostWithDimensions:
        """Add two cost estimates together, or add a dimension to a cost estimate

        Dimensions present (by `name`) in both `self` and `other` will be merged together.
        """
        result = CalculatedCostWithDimensions()  # Work from a copy
        result.merge(self)
        if other == 0:  # Important to handle this case for sum()
            return result
        result.merge(other)
        return result

    def __radd__(
        self, other: CalculatedCostDimension | CalculatedCostWithDimensions | Literal[0]
    ) -> CalculatedCostWithDimensions:
        """Our __add__ operator is commutative, so __radd__ just calls __add__

        This definition supports `sum([...things])`, which starts with `0 + things[0]` and
        therefore needs `things[0]` to support `__radd__`.
        """
        return self.__add__(other)

    def merge(
        self, other: CalculatedCostDimension | CalculatedCostWithDimensions
    ) -> None:
        """Merge a cost dimension or another cost estimate into this one

        Dimensions present (by `name`) in both `self` and `other` will be merged together.
        """
        if isinstance(other, CalculatedCostDimension):
            other = CalculatedCostWithDimensions(dims=[other])
        if not isinstance(other, CalculatedCostWithDimensions):
            raise TypeError(
                "Can only merge a CalculatedCostWithDimensions to a CalculatedCostDimension or another "
                "CalculatedCostWithDimensions (got %s)" % (other,)
            )
        # Merge dimensions from other:
        for dim in other.dims:
            try:
                target_dim = next(d for d in self.dims if d.name == dim.name)
            except StopIteration:
                target_dim = None
            if target_dim:
                if target_dim.cost is None or dim.cost is None:
                    target_dim.cost = None
                else:
                    target_dim.cost += dim.cost
            else:
                self.dims.append(CalculatedCostDimension(name=dim.name, cost=dim.cost))
