"""Output / result structure definitions for cost modelling"""

# Python Built-Ins:
from __future__ import annotations
from itertools import chain
from numbers import Number
from typing import Literal, Sequence

# Local Dependencies:
from ...utils import summary_stats_from_list


class CalculatedCostWithDimensions(dict):
    """Result of a cost estimate (composed of multiple cost dimensions)

    This class is a dictionary of costs keyed by dimension name, but provides additional
    convenience methods including a `.total` property; ability to add multiple cost estimates
    together; and functions to save to or load from a namespace including other information (such
    as an `InvocationResponse` or `Result`)
    """

    @property
    def total(self) -> Number:
        """The total/overall cost over all dimensions"""
        try:
            return sum(v for v in self.values())
        except TypeError:  # Usually in case of None + int/float
            return None

    def __add__(
        self, other: dict[str, Number] | CalculatedCostWithDimensions | Literal[0]
    ) -> CalculatedCostWithDimensions:
        """Add two cost estimates together, or add a dimension to a cost estimate

        Dimensions present by name in both `self` and `other` will be summed together.
        """
        result = CalculatedCostWithDimensions()  # Work from a copy
        result.merge(self)
        if other == 0:  # Important to handle this case for sum()
            return result
        result.merge(other)
        return result

    def __radd__(
        self, other: dict[str, Number] | CalculatedCostWithDimensions | Literal[0]
    ) -> CalculatedCostWithDimensions:
        """Our __add__ operator is commutative, so __radd__ just calls __add__

        This definition supports `sum([...things])`, which starts with `0 + things[0]` and
        therefore needs `things[0]` to support `__radd__`.
        """
        return self.__add__(other)

    def merge(self, other: dict[str, Number] | CalculatedCostWithDimensions) -> None:
        """Merge another cost estimate into this one

        Dimensions present in both `self` and `other` will be summed together.
        """
        if not isinstance(other, (CalculatedCostWithDimensions, dict)):
            raise TypeError(
                "Can only merge a CalculatedCostWithDimensions to a dictionary or another "
                "CalculatedCostWithDimensions (got %s)" % (other,)
            )
        # Merge dimensions from other:
        for k, v in other.items():
            if k in self:
                self[k] += v
            else:
                self[k] = v

    @staticmethod
    def summary_statistics(
        calculated_costs: Sequence[CalculatedCostWithDimensions],
    ) -> dict[str, dict[str, Number]]:
        """Utility function to calculate summary statistics for a sequence of cost results

        For example, you can use this to calculate average/p50/p90/etc of costs over a list of runs
        or invocations.

        Args:
            calculated_costs: Sequence of CalculatedCostWithDimensions results to summarize
        Returns:
            stats: A dictionary keyed by dimension name (plus "total"), where each entry is a
                dictionary keyed by statistic name (e.g. "average", "p90", etc).
        """
        vals_by_dimension: dict[str, list[Number]] = {}
        for c in calculated_costs:
            for dim_name, dim_cost in chain(c.items(), (("total", c.total),)):
                if dim_name not in vals_by_dimension:
                    vals_by_dimension[dim_name] = []
                vals_by_dimension[dim_name].append(dim_cost)
        return {
            name: summary_stats_from_list(vals) # type: ignore
            for name, vals in vals_by_dimension.items()
            # if name.startswith("cost")
            if isinstance(vals[0], Number)
        }

    @classmethod
    def load_from_namespace(
        cls,
        raw: object,
        key_prefix: str = "",
        ignore_total: bool = True,
    ) -> CalculatedCostWithDimensions:
        """Load a `CalculatedCostWithDimensions` from a (potentially shared) namespace/object

        Args:
            raw: The namespace (dataclass, etc) containing cost data
            key_prefix: If specified, only keys starting with this prefix will be included, and the
                prefix will be removed from the generated cost dimension names. Use this for
                loading cost results from classes like InvocationResponse or Result, which may
                contain other information.
            ignore_total: If True [default], the `{key_prefix}total` key will be ignored if
                present. This is useful when working with results saved via `save_on_namespace()`
                with `include_total=True`.
        """
        if hasattr(raw, "__dict__"):
            dict_args = raw.__dict__
        else:
            dict_args = {k: getattr(raw, k) for k in dir(raw)}
        if key_prefix:
            dict_args = {
                k[len(key_prefix) :]: v
                for k, v in dict_args.items()
                if k.startswith(key_prefix)
            }
        if ignore_total:
            dict_args.pop("total", None)
        return cls(**dict_args)

    def save_on_namespace(
        self,
        raw: object,
        key_prefix: str = "",
        include_total: bool = True,
    ) -> CalculatedCostWithDimensions:
        """Create a `CalculatedCostWithDimensions` from a (potentially shared) namespace/object

        Args:
            raw: The namespace (dataclass, etc) containing cost data
            key_prefix: If specified, only keys starting with this prefix will be included, and the
                prefix will be removed from the generated cost dimension names. Use this for
                loading cost results from classes like InvocationResponse or Result, which may
                contain other information.
        """
        for name, value in self.items():
            setattr(raw, f"{key_prefix}{name}", value)
        if include_total:
            setattr(raw, f"{key_prefix}total", self.total)
