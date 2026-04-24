# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from numbers import Number
from typing import Any, Sequence

import jmespath
from upath.types import ReadablePathLike, WritablePathLike

from .endpoints import InvocationResponse
from .json_utils import llmeter_default_serializer
from .utils import ensure_path, summary_stats_from_list

logger = logging.getLogger(__name__)


@dataclass
class Result:
    """Results of a test run."""

    responses: list[InvocationResponse]
    total_requests: int | None = None
    clients: int = 1
    n_requests: int | None = None
    total_test_time: float | None = None
    model_id: str | None = None
    output_path: WritablePathLike | None = None
    endpoint_name: str | None = None
    provider: str | None = None
    run_name: str | None = None
    run_description: str | None = None
    start_time: datetime | None = None
    first_request_time: datetime | None = None
    last_request_time: datetime | None = None
    end_time: datetime | None = None

    def __str__(self):
        return json.dumps(self.stats, indent=4, default=llmeter_default_serializer)

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
            s.write(
                json.dumps(self.stats, indent=4, default=llmeter_default_serializer)
            )

        responses_path = output_path / "responses.jsonl"
        if not responses_path.exists():
            with responses_path.open("w") as f:
                for response in self.responses:
                    f.write(
                        json.dumps(asdict(response), default=llmeter_default_serializer)
                        + "\n"
                    )

    def to_json(self, default=llmeter_default_serializer, **kwargs):
        """Return the results as a JSON string.

        Args:
            default: Fallback serializer. Defaults to
                :func:`~llmeter.json_utils.llmeter_default_serializer`.
            **kwargs: Extra keyword arguments passed to :func:`json.dumps`.
        """
        summary = {
            k: o for k, o in asdict(self).items() if k not in ["responses", "stats"]
        }
        return json.dumps(summary, default=default, **kwargs)

    def to_dict(self, include_responses: bool = False):
        """Return a dictionary representation of this result.

        Returns a plain ``dict`` produced by :func:`dataclasses.asdict`,
        preserving native Python types (``datetime``, ``UPath``, etc.).
        This is suitable for programmatic access and internal data
        processing.

        For JSON output, use :meth:`to_json` which delegates to
        :func:`~llmeter.json_utils.llmeter_default_serializer` for
        non-serializable types, or pass the dict through
        ``json.dumps(result.to_dict(), default=llmeter_default_serializer)``.

        Args:
            include_responses: If ``True``, include the full list of
                :class:`~llmeter.endpoints.base.InvocationResponse` dicts
                and the ``stats`` key.  Defaults to ``False``.

        Returns:
            dict: A dictionary of result fields with native Python types.
        """
        data = asdict(self)
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
            self.responses = [InvocationResponse.from_json(line) for line in f if line]
        logger.info("Loaded %d responses from %s", len(self.responses), responses_path)
        # Recompute stats from the freshly loaded responses
        self._preloaded_stats = self._compute_stats(self)
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
        for key in [
            "start_time",
            "end_time",
            "first_request_time",
            "last_request_time",
        ]:
            if key in summary and summary[key] and isinstance(summary[key], str):
                try:
                    summary[key] = datetime.fromisoformat(
                        summary[key].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

        # Ensure output_path is set so load_responses() can find the files later
        if "output_path" not in summary or summary["output_path"] is None:
            summary["output_path"] = str(result_path)

        if load_responses:
            responses_path = result_path / "responses.jsonl"
            with responses_path.open("r") as f:
                responses = [InvocationResponse.from_json(line) for line in f if line]
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

        # Load or compute stats
        if not load_responses:
            # Use pre-computed stats from disk when responses aren't loaded
            stats_path = result_path / "stats.json"
            if stats_path.exists():
                with stats_path.open("r") as s:
                    result._preloaded_stats = json.loads(s.read())
                    # Convert datetime strings in stats
                    for key in [
                        "start_time",
                        "end_time",
                        "first_request_time",
                        "last_request_time",
                    ]:
                        val = result._preloaded_stats.get(key)
                        if val and isinstance(val, str):
                            try:
                                result._preloaded_stats[key] = datetime.fromisoformat(
                                    val.replace("Z", "+00:00")
                                )
                            except ValueError:
                                pass
            else:
                result._preloaded_stats = None
        else:
            # Compute stats from the loaded responses, but also merge any
            # contributed stats that were persisted in stats.json so they
            # survive a save/load round-trip.
            result._preloaded_stats = cls._compute_stats(result)
            stats_path = result_path / "stats.json"
            if stats_path.exists():
                with stats_path.open("r") as s:
                    saved_stats = json.loads(s.read())
                # Contributed stats are any keys in the saved file that are
                # not produced by _compute_stats (i.e. they came from callbacks).
                for key, value in saved_stats.items():
                    if key not in result._preloaded_stats:
                        result._preloaded_stats[key] = value

        return result

    @classmethod
    def _compute_stats(cls, result: "Result") -> dict:
        """Compute stats from in-memory responses.

        This is the fallback used when ``_preloaded_stats`` is not available — for
        example when a ``Result`` is constructed manually or after
        :meth:`load_responses` reloads data from disk.

        Args:
            result: A ``Result`` instance whose ``responses`` list is populated.

        Returns:
            A flat dictionary matching the ``Result.stats`` schema, containing
            run-level metrics (``failed_requests``, ``requests_per_minute``, …)
            and per-metric aggregations (``time_to_first_token-p50``, …).

        Example::

            result = Result(responses=my_responses, total_requests=100, ...)
            stats = Result._compute_stats(result)
            stats["time_to_first_token-p90"]  # 0.485
        """
        aggregation_metrics = [
            "time_to_last_token",
            "time_to_first_token",
            "num_tokens_output",
            "num_tokens_input",
            "num_tokens_input_cached",
        ]
        results_stats = _get_stats_from_results(result, aggregation_metrics)
        return {
            **result.to_dict(),
            **_get_run_stats(result),
            **{f"{k}-{j}": v for k, o in results_stats.items() for j, v in o.items()},
        }

    @property
    def stats(self) -> dict:
        """Run metrics and aggregated statistics over the individual requests.

        Returns a flat dictionary combining:

        * Basic run information (from ``to_dict()``).
        * Aggregated statistics (``average``, ``p50``, ``p90``, ``p99``) for
          ``time_to_last_token``, ``time_to_first_token``, ``num_tokens_output``,
          and ``num_tokens_input``.  Keys use the format
          ``"{metric}-{aggregation}"``.
        * Run-level throughput metrics (``requests_per_minute``,
          ``total_input_tokens``, etc.).
        * Any additional stats contributed by callbacks via
          :meth:`_update_contributed_stats`.

        During a live run, stats are computed incrementally by
        :class:`~llmeter.utils.RunningStats` and stored in ``_preloaded_stats``.
        When loading from disk with ``load_responses=False``, pre-computed stats
        from ``stats.json`` are used.  As a fallback (e.g. manually constructed
        ``Result``), stats are computed on the fly from ``self.responses``.

        Returns:
            A new shallow copy of the stats dictionary on each access.

        Example::

            result = await runner.run(payload=my_payload, clients=5)
            result.stats["time_to_first_token-p50"]   # 0.312
            result.stats["requests_per_minute"]        # 141.2
            result.stats["failed_requests"]            # 0
        """
        if self._preloaded_stats is not None:
            stats = self._preloaded_stats.copy()
        else:
            # Fallback: compute from responses (e.g. Result constructed manually)
            # Cache so subsequent accesses don't recompute.
            self._preloaded_stats = self._compute_stats(self)
            stats = self._preloaded_stats.copy()

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
    stats["total_cached_input_tokens"] = sum(
        v for v in jmespath.search("[:].num_tokens_input_cached", data=data) if v
    )
    stats["total_reasoning_output_tokens"] = sum(
        v for v in jmespath.search("[:].num_tokens_output_reasoning", data=data) if v
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
