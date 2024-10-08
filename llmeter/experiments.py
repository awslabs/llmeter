# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from math import ceil
from statistics import StatisticsError, quantiles
from typing import Callable

import jmespath
from tokenizers import Tokenizer
from tqdm.auto import tqdm
from upath import UPath as Path

from llmeter.results import _get_stats_from_results

from .endpoints.base import Endpoint
from .plotting import plot_heatmap, plot_sweep_results
from .prompt_utils import CreatePromptCollection
from .runner import Runner

logger = logging.getLogger(__name__)

# using a custom env variable because the TQDM one (https://github.com/tqdm/tqdm/issues/612#issuecomment-2015702344) doesn't work reliably
_disable_tqdm = False
if os.getenv("LLMETER_DISABLE_ALL_PROGRESS_BARS") == "1":
    logger.info("Disabling tqdm progress bars")
    _disable_tqdm = True


@dataclass
class LoadTest:
    endpoint: Endpoint
    payload: dict | list[dict]
    sequence_of_clients: list[int]
    min_requests_per_client: int = 1
    min_requests_per_run: int = 10
    output_path: os.PathLike | str | None = None
    tokenizer: Tokenizer | None = None
    test_name: str | None = None

    def __post_init__(self) -> None:
        self._runner = Runner(endpoint=self.endpoint, tokenizer=self.tokenizer)  # type: ignore
        self._test_name = self.test_name or f"{datetime.now():%Y%m%d-%H%M}"

    def _get_n_requests(self, clients):
        if clients * self.min_requests_per_client < self.min_requests_per_run:
            return int(ceil(self.min_requests_per_run / clients))
        return int(self.min_requests_per_client)

    async def run(self, output_path: os.PathLike | None = None):
        try:
            output_path = Path(output_path or self.output_path) / self._test_name
        except Exception:
            output_path = None
        self._results = [
            await self._runner.run(
                self.payload,
                clients=c,
                n_requests=self._get_n_requests(c),
                output_path=output_path,
                run_name=f"{c:05.0f}-clients",
            )
            for c in tqdm(
                self.sequence_of_clients, desc="Configurations", disable=_disable_tqdm
            )
        ]
        return self._results

    def plot_sweep_results(self):
        if not self._results:
            raise ValueError("No results to plot")
        return plot_sweep_results(
            self._results,
            output_path=Path(self.output_path) / self._test_name
            if self.output_path
            else None,
        )


@dataclass
class LatencyHeatmap:
    """
    Experiment to measure how latency varies by input and output token count

    This experiment uses a source text file to generate input prompts/payloads of different
    lengths, and measures how response time varies with both the input lengths and output/response
    lengths.

    Attributes:
        endpoint (Endpoint): The LLM endpoint to test.
        source_file (UPath | str): The source file from which prompts of different lengths will be
            sampled (see `llmeter.prompt_utils.CreatePromptCollection` for details).
        clients (int): The number of concurrent clients (requests) to use for the experiment. Note
            that using a high number of concurrent clients could impact observed latency.
        output_path (UPath | str | None): The (local or Cloud e.g. `s3://...`) path to save the
            results.
        input_lengths (Sequence[int]): The *approximate* input/prompt lengths to test. Since the
            locally-available `tokenizer` will often differ from the endpoint's own token counting,
            it's typically not possible to generate prompts with the exact specified token counts.
        output_lengths (Sequence[int]): The *target* output lengths to test. Since generation may
            stop early for certain prompts, and some endpoints may not report exact token counts in
            their responses, the results may not correspond exactly to these targets.
        requests_per_combination (int): The number of requests to make *for each combination* of
            input and output lengths.
        create_payload_fn (Callable | None): A function to create the actual endpoint payload for
            each invocation, from the sampled text prompt. Typically, you'll want to specify a
            prefix for your prompt in either this or the `create_payload_kwargs`. If not set, the
            endpoint's default `create_payload` method will be used.
        create_payload_kwargs (Dict): Keyword arguments to pass to the `create_payload_fn`.
        tokenizer (Tokenizer | None): A tokenizer to be used for sampling prompts of the specified
            lengths, and also estimating the generated output lengths if necessary for your
            endpoint. If not set, the `llmeter.tokenizers.DummyTokenizer` will be used.
    """

    endpoint: Endpoint
    source_file: os.PathLike | str
    clients: int = 4
    output_path: os.PathLike | str | None = None
    input_lengths: list[int] = field(default_factory=lambda: [10, 50, 200, 500])
    output_lengths: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    requests_per_combination: int = 1
    create_payload_fn: Callable[..., list[dict] | dict] | None = None
    create_payload_kwargs: dict = field(default_factory=dict)
    tokenizer: Tokenizer | None = None

    def __post_init__(self) -> None:
        _prompt_collection = CreatePromptCollection(
            requests_per_combination=self.requests_per_combination,
            input_lengths=self.input_lengths,
            output_lengths=self.output_lengths,
            source_file=Path(self.source_file),
            tokenizer=self.tokenizer,  # type: ignore
        )

        self.create_payload_fn = self.create_payload_fn or self.endpoint.create_payload
        self.payload = _prompt_collection.create_collection()
        self.payload = [
            self.create_payload_fn(
                input_text, max_tokens=max_new_tokens, **self.create_payload_kwargs
            )
            for input_text, max_new_tokens in self.payload
        ]

        self._runner = Runner(
            endpoint=self.endpoint,
            output_path=self.output_path,
            tokenizer=self.tokenizer,
        )

    async def run(self, output_path=None):
        heatmap_results = await self._runner.run(
            self.payload,
            clients=self.clients,
            n_requests=len(self.input_lengths)
            * len(self.output_lengths)
            * self.requests_per_combination
            // self.clients,
            output_path=output_path or self.output_path,
        )
        self._results = heatmap_results
        return heatmap_results

    def get_heatmaps(self):
        if not self._results:
            raise ValueError("No results to map")
        return heatmaps_from_responses(
            self._results,
            bins_input_tokens=len(self.input_lengths),
            bins_output_tokens=len(self.output_lengths),
        )

    def plot_heatmap(self):
        if not self._results:
            raise ValueError("No results to plot")
        return plot_heatmap(
            self._results,
            bins_input_tokens=len(self.input_lengths),
            bins_output_tokens=len(self.output_lengths),
            output_path=self._results.output_path,
        )


def _map_nested_dicts(ob, func):
    if isinstance(ob, dict):
        return {k: _map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)


def _cut(arr, bins: int):
    assert bins > 0
    min_val = min(arr)
    max_val = max(arr)

    width = (max_val - min_val) / bins  # Bin width
    binned = [
        (min(bins - 1, (k - min_val) // width)) * width + min_val + width / 2
        for k in arr
    ]
    return binned


def _binning(vector, bins: int | None = None) -> list:
    if not vector:
        return []

    if bins is None:
        bins = _calculate_optimal_bins(vector)

    return [x for x in _cut(vector, bins=bins)]


def _calculate_optimal_bins(vector: list) -> int:
    cardinality = len(set(vector))

    if cardinality < len(vector) / 20:
        return cardinality

    try:
        return _calculate_bins_with_iqr(vector, cardinality)
    except StatisticsError:
        return cardinality // 4 + 1


def _calculate_bins_with_iqr(vector: list, cardinality: int) -> int:
    q1, _, q3, _ = quantiles(vector, n=5)
    iqr = q3 - q1
    h = 2 * iqr / (cardinality ** (1 / 3))
    return int((max(vector) - min(vector)) // h) + 1


def heatmaps_from_responses(
    responses,
    metrics: list[str] = ["time_to_last_token", "time_to_first_token"],
    bins_output_tokens: int | None = None,
    bins_input_tokens: int | None = None,
):
    successful_responses = [r for r in responses if not r.error]
    binned_data = _bin_responses_by_tokens(
        successful_responses,
        bins_output_tokens=bins_output_tokens,
        bins_input_tokens=bins_input_tokens,
    )
    heatmaps = _calculate_maps(binned_data, metrics)
    return _add_counts_and_errors(heatmaps, binned_data)


def _bin_responses_by_tokens(
    responses,
    bins_output_tokens: int | None = None,
    bins_input_tokens: int | None = None,
):
    n_input = [r.num_tokens_input for r in responses]
    n_output = [r.num_tokens_output for r in responses]
    bins_input = [f"{k:.0f}" for k in _binning(n_input, bins_input_tokens)]
    bins_output = [f"{k:.0f}" for k in _binning(n_output, bins_output_tokens)]

    binned = defaultdict(lambda: defaultdict(list))
    for bi, bo, r in zip(bins_input, bins_output, responses):
        binned[bi][bo].append(r)

    return binned


def _calculate_maps(binned_data, metrics):
    heatmaps = _map_nested_dicts(
        binned_data, partial(_get_stats_from_results, metrics=metrics)
    )
    return _sort_map_labels(heatmaps)


def _sort_map_labels(heatmaps):
    sorted_heatmaps = dict(sorted(heatmaps.items()))
    return {k: dict(sorted(v.items())) for k, v in sorted_heatmaps.items()}


def _add_counts_and_errors(heatmaps, binned_data):
    for input_bin, output_bins in binned_data.items():
        for output_bin, responses in output_bins.items():
            heatmaps[input_bin][output_bin].update(
                {
                    "counts": len(responses),
                    "errors": sum(1 for r in responses if r.error),
                }
            )
    return heatmaps


def get_heatmap_stats(
    heatmaps,
    search_expression: str,
):
    return {
        ok: {ik: jmespath.search(search_expression, iv) for ik, iv in ov.items()}
        for ok, ov in heatmaps.items()
    }
