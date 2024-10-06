# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import filterfalse
from math import isnan
import os
from statistics import StatisticsError, mean, median, quantiles
from typing import Dict, Sequence

import jmespath
from upath import UPath as Path

from .endpoints import InvocationResponse

logger = logging.getLogger(__name__)


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

    def __str__(self):
        return json.dumps(self.stats, indent=4, default=str)

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
            s.write(json.dumps(self.stats, indent=4, default=str))

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
        return json.dumps(summary, default=str, **kwargs)

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
        return cls(responses=responses, **summary)

    @cached_property
    def stats(self) -> Dict:
        """
        Calculate and return the overall run statistics.

        This property method computes various statistics based on the run results.
        It combines information from the instance's dictionary representation,
        test-specific statistics, and aggregated statistics from individual results.

        The statistics include:
        - Basic information of the run
        - Aggregated statistics for:
            - Time to last token
            - Time to first token
            - Number of tokens output
            - Number of tokens input

        Returns:
            Dict: A dictionary containing all computed statistics. The keys are:
                - All key-value pairs from the instance's dictionary representation
                - Test-specific statistics
                - Aggregated statistics with keys in the format "{stat_name}-{aggregation_type}"
                  where stat_name is one of the four metrics listed above, and
                  aggregation_type includes measures like mean, median, etc.

        Note:
            This method uses the @cached_property decorator, which means the result
            is computed once and then cached for subsequent accesses, improving
            performance for repeated calls.
        """

        results_stats = _get_stats_from_results(
            self,
            [
                "time_to_last_token",
                "time_to_first_token",
                "num_tokens_output",
                "num_tokens_input",
            ],
        )
        return {
            **self.to_dict(),
            **_get_test_stats(self),
            **{f"{k}-{j}": v for k, o in results_stats.items() for j, v in o.items()},
        }

    def __repr__(self) -> str:
        return self.to_json()


def _get_stats_from_results(
    results: Result | Sequence[InvocationResponse], metrics: Sequence[str]
):
    stats = {}
    data = [
        (p if isinstance(p, dict) else p.to_dict())
        for p in (results.responses if isinstance(results, Result) else results)
    ]
    for metric in metrics:
        metric_data = jmespath.search(f"[:].{metric}", data=data)
        stats[metric] = _get_stats_from_list(metric_data)
    return stats


def _get_test_stats(results: Result):
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
    return stats


def _get_stats_from_list(data: Sequence[int | float]):
    clean_data = list(filterfalse(isnan, data))
    try:
        return dict(
            p50=median(clean_data),
            p90=clean_data[0]
            if len(clean_data) == 1
            else quantiles(clean_data, n=10)[-1],
            p99=clean_data[0]
            if len(clean_data) == 1
            else quantiles(clean_data, n=100)[-1],
            average=mean(clean_data),
        )
    except StatisticsError:
        return {}
