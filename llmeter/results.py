# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cached_property
from numbers import Number
from typing import Any, Sequence

import jmespath
from upath import UPath as Path

from .endpoints import InvocationResponse
from .utils import summary_stats_from_list

logger = logging.getLogger(__name__)


def utc_datetime_serializer(obj):
    """
    Serialize datetime objects to UTC ISO format strings.

    Args:
        obj: Object to serialize. If datetime, converts to ISO format string with 'Z' timezone.
             Otherwise returns string representation.

    Returns:
        str: ISO format string with 'Z' timezone for datetime objects, or string representation
             for other objects.
    """
    if isinstance(obj, datetime):
        # Convert to UTC if timezone is set
        if obj.tzinfo is not None:
            obj = obj.astimezone(timezone.utc)
        return obj.isoformat(timespec="seconds").replace("+00:00", "Z")
    return str(obj)


@dataclass
class Result:
    """Results of an experiment run."""

    responses: list[InvocationResponse]
    total_requests: int
    clients: int
    n_requests: int
    total_test_time: float | None = None
    model_id: str | None = None
    output_path: os.PathLike | None = None
    endpoint_name: str | None = None
    provider: str | None = None
    run_name: str | None = None
    run_description: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    def __str__(self):
        return json.dumps(self.stats, indent=4, default=utc_datetime_serializer)

    def __post_init__(self):
        """Initialize the Result instance."""
        self._contributed_stats = {}

    def _update_contributed_stats(self, stats: dict[str, Number]):
        """
        Upsert externally-provided statistics for the `stats` property

        Callbacks can use this method to extend the default stats with additional key-value pairs.
        """
        if not isinstance(stats, dict):
            raise ValueError("Stats must be a dictionary")
        for key, value in stats.items():
            if not isinstance(value, Number):
                raise ValueError(
                    f"Value for key {key} must be a number, got {type(value)}"
                )
        self._contributed_stats.update(stats)

    def save(self, output_path: os.PathLike | str | None = None):
        """
        Save the results to disk or cloud storage.

        Saves the run results to the specified output path or the
        instance's default output path. It creates three files:
        1. 'summary.json': Contains the overall summary of the results.
        2. 'stats.json': Contains detailed statistics of the run.
        3. 'responses.jsonl': Contains individual invocation responses
            - Only if the responses are not already saved at the indicated path.


        Args:
            output_path (UPath | str | None, optional): The path where the result
                files will be saved. If None, the instance's default output_path
                will be used. Defaults to None.

        Raises:
            ValueError: If no output path is provided and the instance doesn't
                have a default output_path set.
            TypeError: If the provided output_path is not a valid type.
            IOError: If there's an error writing to the output files.

        Note:
            The method uses the Universal Path (UPath) library for file operations,
            which provides a unified interface for working with different file systems.
        """

        try:
            output_path = Path(self.output_path or output_path)
        except TypeError:
            raise ValueError("No output path provided")

        output_path.mkdir(parents=True, exist_ok=True)

        summary_path = output_path / "summary.json"
        stats_path = output_path / "stats.json"
        with summary_path.open("w") as f, stats_path.open("w") as s:
            f.write(self.to_json(indent=4))
            s.write(json.dumps(self.stats, indent=4, default=utc_datetime_serializer))

        responses_path = output_path / "responses.jsonl"
        if not responses_path.exists():
            with responses_path.open("w") as f:
                for response in self.responses:
                    f.write(json.dumps(asdict(response)) + "\n")

    def to_json(self, **kwargs):
        """Return the results as a JSON string."""
        summary = {
            k: o for k, o in asdict(self).items() if k not in ["responses", "stats"]
        }
        return json.dumps(summary, default=utc_datetime_serializer, **kwargs)

    def to_dict(self, include_responses: bool = False):
        """Return the results as a dictionary."""
        if include_responses:
            return asdict(self)
        return {
            k: o for k, o in asdict(self).items() if k not in ["responses", "stats"]
        }

    @classmethod
    def load(cls, result_path: os.PathLike | str):
        """
        Load run results from disk or cloud storage.

        Reads previously saved run results from the specified
        path. It expects two files to be present in the given directory:
        'responses.jsonl' containing individual invocation responses, and
        'summary.json' containing summary information.

        Args:
            result_path (UPath | str): The path to the directory containing the
                result files. Can be a string or a UPath object.

        Returns:
            Result: An instance of the Result class containing the loaded
            responses and summary data.

        Raises:
            FileNotFoundError: If either 'responses.jsonl' or 'summary.json'
                is not found in the specified directory.
            JSONDecodeError: If there's an issue parsing the JSON data in
                either file.

        """
        result_path = Path(result_path)
        responses_path = result_path / "responses.jsonl"
        summary_path = result_path / "summary.json"
        with open(responses_path, "r") as f, summary_path.open("r") as g:
            responses = [InvocationResponse(**json.loads(line)) for line in f if line]
            summary = json.load(g)
            # Convert datetime strings back to datetime objects
            for key in ["start_time", "end_time"]:
                if key in summary and summary[key] and isinstance(summary[key], str):
                    try:
                        summary[key] = datetime.fromisoformat(
                            summary[key]
                        )
                    except ValueError:
                        pass
        return cls(responses=responses, **summary)

    @cached_property
    def _builtin_stats(self) -> dict:
        """
        Default run metrics and aggregated statistics provided by LLMeter core

        Users should generally refer to the `.stats` property instead, which combines this data
        with any additional values contributed by callbacks or other extensions.

        This is a read-only and `@cached_property`, which means the result is computed once and
        then cached for subsequent accesses - improving performance.

        Returns:
            stats: A dictionary containing all computed statistics. The keys are:
                - All key-value pairs from the Result's dictionary representation
                - Test-specific statistics
                - Aggregated statistics with keys in the format "{stat_name}-{aggregation_type}"
                  where stat_name is one of the four metrics listed above, and
                  aggregation_type includes measures like mean, median, etc.
        """

        aggregation_metrics = [
            "time_to_last_token",
            "time_to_first_token",
            "num_tokens_output",
            "num_tokens_input",
        ]

        results_stats = _get_stats_from_results(
            self,
            aggregation_metrics,
        )
        return {
            **self.to_dict(),
            **_get_run_stats(self),
            **{f"{k}-{j}": v for k, o in results_stats.items() for j, v in o.items()},
        }

    @property
    def stats(self) -> dict:
        """
        Run metrics and aggregated statistics over the individual requests

        This combined view includes:
        - Basic information about the run (from the Result's dictionary representation)
        - Aggregated statistics ('average', 'p50', 'p90', 'p99') for:
            - Time to last token
            - Time to first token
            - Number of tokens output
            - Number of tokens input

        Aggregated statistics are keyed in the format "{stat_name}-{aggregation_type}"

        This property is read-only and returns a new shallow copy of the data on each access.
        Default stats provided by LLMeter are calculated on first access and then cached. Callbacks
        Callbacks or other mechanisms needing to augment stats should use the
        `_update_contributed_stats()` method.
        """
        stats = self._builtin_stats.copy()

        if self._contributed_stats:
            stats.update(self._contributed_stats)
        return stats

    def __repr__(self) -> str:
        return self.to_json()

    def get_dimension(
        self,
        dimension: str,
        filter_dimension: str | None = None,
        filter_value: Any = None,
    ):
        """
        Get the values of a specific dimension from the responses.

        Args:
            dimension (str): The name of the dimension to retrieve.
            filter_dimension (str, optional): Name of dimension to filter on. Defaults to None.
            filter_value (any, optional): Value to match for the filter dimension. Defaults to None.

        Returns:
            list: A list of values for the specified dimension across all responses.

        Raises:
            ValueError: If the specified dimension is not found in any response.
        """
        if filter_dimension is not None:
            values = [
                getattr(response, dimension)
                for response in self.responses
                if getattr(response, filter_dimension) == filter_value
            ]
        else:
            values = [getattr(response, dimension) for response in self.responses]

        if not any(values):
            # raise ValueError(f"Dimension {dimension} not found in any response")
            logger.warning(f"Dimension {dimension} not found in any response")
        return values


def _get_stats_from_results(
    results: Result | Sequence[InvocationResponse], metrics: Sequence[str]
):
    """
    Calculate statistics for specified metrics from a collection of experiment results.

    Args:
        results (Result | Sequence[InvocationResponse]): Either a Result object containing
            a run Result or a sequence of InvocationResponse objects.
        metrics (Sequence[str]): A sequence of metric names to calculate statistics for.
            These metrics should be attributes available in the InvocationResponse objects.

    Returns:
        dict: A dictionary containing calculated statistics for each specified metric.
            The dictionary is structured as {metric_name: metric_statistics}.

    Example:
        >>> results = Result(responses=[...])  # Result object with responses
        >>> metrics = ["time_to_first_token", "time_to_last_token"]
        >>> stats = _get_stats_from_results(results, metrics)
    """

    stats = {}
    data = [
        (p if isinstance(p, dict) else p.to_dict())
        for p in (results.responses if isinstance(results, Result) else results)
    ]
    for metric in metrics:
        metric_data = jmespath.search(f"[:].{metric}", data=data)
        stats[metric] = summary_stats_from_list(metric_data)
    return stats


def _get_run_stats(results: Result):
    """
    Calculate key performance statistics from a test run Result object.

    This function processes the test results to compute various performance metrics
    including failure rates and throughput measurements.

    Args:
        results (Result): A Result object containing test responses and metadata.
            Expected to have the following attributes:
            - responses: List of response objects with error information
            - total_requests: Total number of requests made
            - total_test_time: Total duration of the test run

    Returns:
        Dict[str, float]: A dictionary containing the following statistics:
            - 'failed_requests': Number of failed requests
            - 'failed_requests_rate': Ratio of failed requests to total requests
            - 'requests_per_minute': Average number of requests processed per minute

    Note:
        - Failed requests are determined by the presence of an error field in responses
        - If total_requests or total_test_time is zero, related rates will be None
        - Uses jmespath for JSON path searching in response data
    """

    stats = {}
    data = [p.to_dict() for p in results.responses]
    stats["failed_requests"] = len(jmespath.search("[:].error", data=data))
    stats["failed_requests_rate"] = (
        results.total_requests and stats["failed_requests"] / results.total_requests
    )
    stats["requests_per_minute"] = (
        results.total_test_time
        and results.total_requests / results.total_test_time * 60
    )
    stats["total_input_tokens"] = sum(
        jmespath.search("[:].num_tokens_input", data=data)
    )
    stats["total_output_tokens"] = sum(
        jmespath.search("[:].num_tokens_output", data=data)
    )
    stats["average_input_tokens_per_minute"] = (
        results.total_test_time
        and stats["total_input_tokens"] / results.total_test_time * 60
    )
    stats["average_output_tokens_per_minute"] = (
        results.total_test_time
        and stats["total_output_tokens"] / results.total_test_time * 60
    )
    return stats
