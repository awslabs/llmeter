from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from math import ceil, floor
from typing import Literal

import jmespath

from llmeter.endpoints.base import InvocationResponse
from llmeter.results import Result, _get_stats_from_results


@dataclass(eq=True, frozen=True, order=True)
class Interval:
    """A class representing a mathematical interval.

    Represents an interval [left, right] with configurable endpoint inclusion.

    Attributes:
        left (float | int): Left endpoint of the interval
        right (float | int): Right endpoint of the interval
        closed (Literal["right", "left", "both", "neither"]): Which endpoints are included
            - "right": Include right endpoint only (default)
            - "left": Include left endpoint only
            - "both": Include both endpoints
            - "neither": Exclude both endpoints
    """

    left: float | int
    right: float | int
    closed: Literal["right", "left", "both", "neither"] = "right"

    def __contains__(self, value: float) -> bool:
        """Check if a value is contained in the interval.

        Args:
            value (float): Value to check

        Returns:
            bool: True if value is in interval, False otherwise
        """
        return self.left <= value <= self.right

    def __str__(self) -> str:
        """Return string representation of the interval.

        Returns:
            str: String in interval notation, e.g. "[1,2]" or "(1,2)"
        """
        return f"{
            "[" if self.closed
            in ["left", "both"] else "("
            }{self.left}, {self.right}{
                "]" if self.closed in ["right", "both"] else ")"
                }"

    @property
    def mid(self):
        """Return the midpoint of the interval.

        Returns:
            float: Midpoint of the interval
        """
        return (self.left + self.right) / 2


class Heatmap:
    def __init__(
        self,
        responses: list[InvocationResponse] | None = None,
        result: Result | None = None,
        n_bins_output_tokens: int = 10,
        n_bins_input_tokens: int = 10,
    ):
        """Initialize the Heatmap class.

        Args:
            responses (list[InvocationResponse] | None): List of InvocationResponse objects
            result (Result | None): Result object containing InvocationResponse objects
            n_bins_output_tokens (int): Number of bins for output tokens
            n_bins_input_tokens (int): Number of bins for input tokens

        Raises:
            ValueError: If neither responses nor result is provided
        """

        if responses is None and result is None:
            raise ValueError("Either responses or result must be provided")

        self.responses = responses or result.responses  # type: ignore
        self.bins_output_tokens = n_bins_output_tokens
        self.bins_input_tokens = n_bins_input_tokens

        self._map, self.bin_boundaries_input, self.bin_boundaries_output = (
            _bin_responses_by_tokens(
                self.responses,
                n_bins_output_tokens,
                n_bins_input_tokens,
            )
        )

    def get_map(self, metric: str, aggregation: str | None = None):
        h_maps = _calculate_maps(self._map, [metric])

        search_path = f"{metric}"
        if aggregation:
            search_path = f"{metric}.{aggregation}"

        h = _get_heatmap_stats(h_maps, search_path)
        return h


def _cut(arr, bins):
    """Cut an array into bins and create intervals.

    This function takes a numeric array and divides it into a specified number of bins,
    creating intervals for both individual values and bin boundaries.

    Args:
        arr (list): Array of numeric values to bin
        bins (int): Number of bins to create

    Returns:
        tuple: A tuple containing:
            - list[Interval]: Intervals for each value in the input array
            - list[Interval]: Intervals representing the bin boundaries

    Example:
        >>> _cut([1,2,3,4], 2)
        ([Interval(1,2.5), Interval(1,2.5), Interval(2.5,4), Interval(2.5,4)],
         [Interval(1,2.5), Interval(2.5,4)])
    """
    t_max, t_min = max(arr), min(arr)
    bin_width = ceil((t_max - t_min) / bins)
    bin_boundaries = [i * bin_width + t_min for i in range(bins + 1)]
    bin_indexes = [floor((k - t_min) / bin_width) for k in arr]
    # binned = [Interval(bin_boundaries[k], bin_boundaries[k + 1]) for k in bin_indexes]
    return [Interval(bin_boundaries[k], bin_boundaries[k + 1]) for k in bin_indexes], [
        Interval(left, right)
        for left, right in zip(bin_boundaries[:-1], bin_boundaries[1:])
    ]


def initialize_map(n_bin_input, n_bin_output):
    return [[None for k in range(n_bin_input)] for j in range(n_bin_output)]


def _bin_responses_by_tokens(
    responses,
    bins_output_tokens: int,
    bins_input_tokens: int,
):
    """Bin responses based on their input and output token counts.

    This function takes a list of responses and bins them based on their input and output
    token counts into a 2D grid structure.

    Args:
        responses (list): List of response objects with num_tokens_input and num_tokens_output attributes
        bins_output_tokens (int): Number of bins to create for output tokens
        bins_input_tokens (int): Number of bins to create for input tokens

    Returns:
        tuple: A tuple containing:
            - defaultdict: 2D grid of binned responses, indexed by output bin then input bin
            - list[Interval]: List of intervals representing input token bin boundaries
            - list[Interval]: List of intervals representing output token bin boundaries

    Example:
        >>> responses = [Response(num_tokens_input=10, num_tokens_output=20), ...]
        >>> binned, input_bins, output_bins = _bin_responses_by_tokens(responses, 5, 5)
    """
    n_input = [r.num_tokens_input for r in responses]
    n_output = [r.num_tokens_output for r in responses]
    bin_idx_input, bin_input = _cut(n_input, bins_input_tokens)
    bin_idx_output, bin_output = _cut(n_output, bins_output_tokens)

    binned = defaultdict(lambda: defaultdict(list))
    for bi, bo, r in zip(bin_idx_input, bin_idx_output, responses):
        binned[bo][bi].append(r)
    return binned, bin_input, bin_output


def _counts_and_errors(results):
    """
    Compute counts, errors and error rates from a list of results.

    Args:
        results (list): List of result objects that have an 'error' attribute

    Returns:
        dict: Dictionary containing:
            - counts (int): Total number of results
            - errors (int): Number of results with errors
            - error_rate (float): Ratio of errors to total results, 0 if no results

    Handles empty results list by returning 0 for all metrics.
    """
    if not results:
        return {"counts": 0, "errors": 0, "error_rate": 0}

    counts = len(results)
    errors = sum(1 for r in results if r.error)

    return {
        "counts": counts,
        "errors": errors,
        "error_rate": errors / counts if counts else 0,
    }


def _calculate_maps(binned_data, metrics):
    """Calculate heatmap statistics from binned data.

    This function processes binned data to generate heatmaps containing statistics and counts.
    It applies metrics calculations and error counting to each bin, then merges and sorts the results.

    Args:
        binned_data (dict): Nested dictionary containing binned response data
        metrics (list): List of metrics to calculate for each bin

    Returns:
        dict: Sorted heatmap data containing:
            - Statistics calculated from the specified metrics
            - Counts and error rates for each bin
            - Outer keys sorted in descending order
            - Inner keys sorted in ascending order

    Example:
        >>> metrics = ["latency", "tokens_per_second"]
        >>> heatmaps = _calculate_maps(binned_responses, metrics)
    """
    heatmaps = _map_nested_dicts(
        binned_data, partial(_get_stats_from_results, metrics=metrics)
    )
    heatmaps_counts = _map_nested_dicts(binned_data, _counts_and_errors)

    heatmaps = {
        ko: {ki: {**vi, **heatmaps_counts[ko][ki]} for ki, vi in vo.items()}
        for ko, vo in heatmaps.items()
    }

    return _sort_map_labels(heatmaps)


def _sort_map_labels(heatmaps):
    """
    Sort the keys of a nested dictionary in descending order for outer keys and ascending order for inner keys.

    Args:
        heatmaps (dict): A nested dictionary to be sorted

    Returns:
        dict: A new dictionary with sorted keys where:
            - Outer keys are sorted in descending order (reverse=True)
            - Inner keys are sorted in ascending order
            - Values remain unchanged

    Example:
        >>> data = {2: {'b': 1, 'a': 2}, 1: {'d': 3, 'c': 4}}
        >>> _sort_map_labels(data)
        {2: {'a': 2, 'b': 1}, 1: {'c': 4, 'd': 3}}
    """
    sorted_heatmaps = dict(sorted(heatmaps.items(), reverse=True))
    return {k: dict(sorted(v.items())) for k, v in sorted_heatmaps.items()}


def _map_nested_dicts(ob, func):
    """
    Recursively apply a function to values in nested dictionaries.

    This function traverses a dictionary structure of arbitrary depth and applies
    the given function to all non-dictionary values. Dictionary values are
    recursively processed to transform their contents.

    Args:
        ob: The input object to process. Can be either a dictionary or a non-dictionary value.
        func: A callable that will be applied to all non-dictionary values.
            Should accept a single argument and return a transformed value.

    Returns:
        If the input is a dictionary:
            Returns a new dictionary with the same structure but transformed values.
        If the input is not a dictionary:
            Returns the result of applying func to the input value.

    Example:
        >>> def double(x):
        ...     return x * 2
        >>> data = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> _map_nested_dicts(data, double)
        {'a': 2, 'b': {'c': 4, 'd': 6}}
    """

    if isinstance(ob, dict):
        return {k: _map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)


def _get_heatmap_stats(
    heatmaps,
    search_expression: str,
):
    """Extract statistics from heatmap data using a JMESPath search expression.

    This function searches through nested heatmap data using a JMESPath expression
    to extract specific statistics.

    Args:
        heatmaps (dict): Nested dictionary containing heatmap data
        search_expression (str): JMESPath expression to search the heatmap data

    Returns:
        dict: Dictionary with same structure as input but containing only the
            values matching the search expression

    Example:
        >>> heatmaps = {'bin1': {'bin2': {'stat1': 10, 'stat2': 20}}}
        >>> get_heatmap_stats(heatmaps, 'stat1')
        {'bin1': {'bin2': 10}}
    """
    return {
        ok: {ik: jmespath.search(search_expression, iv) for ik, iv in ov.items()}
        for ok, ov in heatmaps.items()
    }
