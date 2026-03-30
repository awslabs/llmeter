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
from upath.types import ReadablePathLike, WritablePathLike

from .endpoints import InvocationResponse
from .prompt_utils import LLMeterBytesEncoder
from .utils import ensure_path, summary_stats_from_list

logger = logging.getLogger(__name__)


def utc_datetime_serializer(obj: Any) -> str:
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
    if isinstance(obj, os.PathLike):
        return Path(obj).as_posix()
    return str(obj)


class InvocationResponseEncoder(LLMeterBytesEncoder):
    """Extended encoder for InvocationResponse with fallback to str() for non-serializable types.
    
    This encoder extends LLMeterBytesEncoder to handle bytes objects (via parent class)
    and adds a fallback mechanism for other non-serializable types by converting them
    to strings. This is particularly useful for InvocationResponse objects that may
    contain various non-standard types.
    
    Example:
        >>> response = InvocationResponse(input_payload={"image": {"bytes": b"\\xff\\xd8"}})
        >>> json.dumps(asdict(response), cls=InvocationResponseEncoder)
        '{"input_payload": {"image": {"bytes": {"__llmeter_bytes__": "/9g="}}}}'
    """
    
    def default(self, obj):
        """Encode objects with bytes support and str() fallback.
        
        Args:
            obj: Object to encode
            
        Returns:
            Encoded representation or None if encoding fails
        """
        # First try bytes encoding from parent
        if isinstance(obj, bytes):
            return super().default(obj)
        if isinstance(obj, os.PathLike):
            return Path(obj).as_posix()
        # Fallback to string representation for other non-serializable types
        try:
            return str(obj)
        except Exception:
            return None


@dataclass
class Result:
    """Results of a test run."""

    responses: list[InvocationResponse]
    total_requests: int
    clients: int
    n_requests: int
    total_test_time: float | None = None
    model_id: str | None = None
    output_path: WritablePathLike | None = None
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
        if not hasattr(self, "_preloaded_stats"):
            self._preloaded_stats = None

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

    def save(self, output_path: WritablePathLike | None = None):
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

        output_path = ensure_path(self.output_path or output_path)
        if output_path is None:
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
                    f.write(json.dumps(asdict(response), cls=InvocationResponseEncoder) + "\n")

    def to_json(self, **kwargs):
        """Return the results as a JSON string."""
        summary = {
            k: o for k, o in asdict(self).items() if k not in ["responses", "stats"]
        }
        return json.dumps(summary, default=utc_datetime_serializer, **kwargs)

    def to_dict(self, include_responses: bool = False):
        """Return the results as a dictionary with JSON-serializable values."""
        data = asdict(self)
        # Serialize datetime objects so stats dict is always JSON-safe
        for key in ("start_time", "end_time"):
            if key in data and isinstance(data[key], datetime):
                data[key] = utc_datetime_serializer(data[key])
        if include_responses:
            return data
        return {k: v for k, v in data.items() if k not in ["responses", "stats"]}

    def load_responses(self) -> list[InvocationResponse]:
        """
        Load individual invocation responses from disk or cloud storage.

        Reads the 'responses.jsonl' file from the result's output_path directory.
        This is useful when the Result was loaded with ``load_responses=False`` and
        you need to access the individual responses on demand.

        Returns:
            list[InvocationResponse]: The loaded responses. Also updates ``self.responses``
            in place.

        Raises:
            ValueError: If no output_path is set on this Result.
            FileNotFoundError: If 'responses.jsonl' is not found at the output_path.
        """
        if not self.output_path:
            raise ValueError(
                "No output_path set on this Result. Cannot locate responses file."
            )
        responses_path = ensure_path(self.output_path) / "responses.jsonl"
        with responses_path.open("r") as f:
            self.responses = [
                InvocationResponse(**json.loads(line)) for line in f if line
            ]
        logger.info("Loaded %d responses from %s", len(self.responses), responses_path)
        # Invalidate cached stats so they are recomputed with the loaded responses
        self.__dict__.pop("_builtin_stats", None)
        return self.responses

    @classmethod
    def load(
        cls, result_path: ReadablePathLike, load_responses: bool = True
    ) -> "Result":
        """
        Load run results from disk or cloud storage.

        Reads previously saved run results from the specified
        path. It expects 'summary.json' to be present in the given directory.
        By default, also loads 'responses.jsonl' containing individual invocation
        responses.

        Args:
            result_path (UPath | str): The path to the directory containing the
                result files. Can be a string or a UPath object.
            load_responses (bool): Whether to load individual invocation responses
                from 'responses.jsonl'. Defaults to True. When False, only the
                summary and pre-computed stats are loaded, which is significantly
                faster for large result sets. Use ``result.load_responses()`` to
                load them on demand later.

        Returns:
            Result: An instance of the Result class containing the loaded
            responses and summary data.

        Raises:
            FileNotFoundError: If required files are not found in the specified
                directory.
            JSONDecodeError: If there's an issue parsing the JSON data in
                either file.

        """
        result_path = ensure_path(result_path)
        summary_path = result_path / "summary.json"

        with summary_path.open("r") as g:
            summary = json.load(g)

        # Convert datetime strings back to datetime objects
        for key in ["start_time", "end_time"]:
            if key in summary and summary[key] and isinstance(summary[key], str):
                try:
                    summary[key] = datetime.fromisoformat(summary[key])
                except ValueError:
                    pass

        # Ensure output_path is set so load_responses() can find the files later
        if "output_path" not in summary or summary["output_path"] is None:
            summary["output_path"] = str(result_path)

        if load_responses:
            responses_path = result_path / "responses.jsonl"
            with responses_path.open("r") as f:
                responses = [
                    InvocationResponse(**json.loads(line)) for line in f if line
                ]
        else:
            responses = []
            responses_path = result_path / "responses.jsonl"
            logger.info(
                "Loaded summary only (responses not loaded). "
                "Individual responses are stored at: %s. "
                "Call result.load_responses() to load them on demand.",
                responses_path,
            )

        result = cls(responses=responses, **summary)

        # When skipping responses, load pre-computed stats from stats.json if available
        # so that result.stats works without needing the responses
        if not load_responses:
            stats_path = result_path / "stats.json"
            if stats_path.exists():
                with stats_path.open("r") as s:
                    result._preloaded_stats = json.loads(s.read())
                    # Convert datetime strings in stats
                    for key in ["start_time", "end_time"]:
                        val = result._preloaded_stats.get(key)
                        if val and isinstance(val, str):
                            try:
                                result._preloaded_stats[key] = datetime.fromisoformat(
                                    val
                                )
                            except ValueError:
                                pass
            else:
                result._preloaded_stats = None

        return result

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

        When the Result was loaded with ``load_responses=False``, pre-computed stats from
        ``stats.json`` are returned if available. Call ``load_responses()`` to load the
        individual responses and recompute stats from the raw data.
        """
        # Use preloaded stats when responses were not loaded
        if not self.responses and self._preloaded_stats is not None:
            stats = self._preloaded_stats.copy()
            if self._contributed_stats:
                stats.update(self._contributed_stats)
            return stats

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
    ) -> list:
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
