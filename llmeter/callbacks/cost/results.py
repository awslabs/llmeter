"""Output / result structure definitions for cost modelling"""

# Python Built-Ins:
from __future__ import annotations
from numbers import Number
from typing import Literal


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
