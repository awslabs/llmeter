# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Python Built-Ins:
from __future__ import annotations
from numbers import Number
from typing import Literal, Sequence

# Local Dependencies:
from ...utils import summary_stats_from_list


class CalculatedCostWithDimensions(dict):
    """Result of a cost estimate (composed of multiple cost dimensions)

    This class is a dictionary of costs keyed by dimension name, but provides additional
    convenience methods, including:

    - A`.total` property, providing the total cost across all dimensions
    - Ability to add multiple cost estimates together (using standard `+` or `sum()`)
    - Functions to save the estimate to (or load one from) a namespace including other information
        (such as another dictionary, an `InvocationResponse` or `Result`)
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

        This definition supports `sum([...costs])`, which starts with `0 + costs[0]` and therefore
        needs `costs[0]` to support `__radd__`.
        """
        return self.__add__(other)

    def merge(self, other: dict[str, Number] | CalculatedCostWithDimensions) -> None:
        """Merge another cost estimate into this one (in-place)

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
        key_prefix: str = "",
        key_dim_name_suffix: str = "",
        key_stat_name_prefix: str = "-",
        key_total_name_and_suffix: str = "total",
    ) -> dict[str, Number]:
        """Utility function to calculate summary statistics for a dataset of cost results

        At a high level, this method produces a flat map from keys like [dimension]-[statistic], to
        the value of that statistic for that dimension.

        Args:
            calculated_costs: Sequence of CalculatedCostWithDimensions results to summarize
            key_prefix: Prefix to add to each key in the output dictionary. This is useful in case
                the output will be merged with other statistics. Set to `"cost_"` to match
                CostModel callback's default treatment, or leave default for no prefix.
            key_dim_name_suffix: Suffix to add to each dimension name in the output dictionary.
                Set to `"_per_request"` to match CostModel callback's default treatment, or leave
                default for no suffix.
            key_stat_name_prefix: Separator to use before the name of the statistic in output keys.
            key_total_name_and_suffix: Name to use for the `.total` dimension, *and* its attached
                suffix if one should be present. Set to `"per_request"` to match CostModel
                callback's default treatment, or leave default to call the dimension "total".

        Returns:
            stats: A flat dictionary of summary statistics from all dimensions of the input
                `calculated_costs`, and their totals.
        """
        vals_by_dimension: dict[str, list[Number]] = {}
        total_vals: list[Number] = []
        for c in calculated_costs:
            for dim_name, dim_cost in c.items():
                if dim_name not in vals_by_dimension:
                    vals_by_dimension[dim_name] = []
                vals_by_dimension[dim_name].append(dim_cost)
            total_vals.append(c.total)
        # Dimension-level stats:
        result = {
            f"{key_prefix}{dim_name}{key_dim_name_suffix}{key_stat_name_prefix}{stat_name}": stat_val
            for dim_name, dim_values in vals_by_dimension.items()
            if isinstance(dim_values[0], Number)
            for stat_name, stat_val in summary_stats_from_list(dim_values).items()
        }
        # Total stats:
        result.update(
            {
                f"{key_prefix}{key_total_name_and_suffix}{key_stat_name_prefix}{stat_name}": stat_val
                for stat_name, stat_val in summary_stats_from_list(total_vals).items()
            }
        )
        return result

    @classmethod
    def load_from_namespace(
        cls,
        raw: object,
        key_prefix: str = "",
        ignore_total: bool = True,
    ) -> CalculatedCostWithDimensions:
        """Load a `CalculatedCostWithDimensions` from a (potentially shared) namespace/object

        Args:
            raw: The namespace (dataclass, dictionary, etc) containing cost data
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
        elif hasattr(raw, "items") and callable(raw.items):
            dict_args = {k: v for k, v in raw.items()}
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
    ) -> None:
        """Store this cost result on a (potentially shared) namespace/object

        Args:
            raw: The namespace (dataclass, dictionary, etc) containing cost data
            key_prefix: If specified, only keys starting with this prefix will be included, and the
                prefix will be removed from the generated cost dimension names. Use this for
                loading cost results from classes like InvocationResponse or Result, which may
                contain other information.
            include_total: If True [default], the `.total` property will also be written to the
                `{key_prefix}total` key on the output. Set False to omit.
        """

        def setter(target, key, value):
            if hasattr(target, "__setitem__"):
                target[key] = value
            else:
                setattr(target, key, value)

        for name, value in self.items():
            setter(raw, f"{key_prefix}{name}", value)
        if include_total:
            setter(raw, f"{key_prefix}total", self.total)
