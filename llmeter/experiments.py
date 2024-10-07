# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from math import ceil
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from tokenizers import Tokenizer
from tqdm.auto import tqdm
from upath import UPath as Path

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

    def plot_heatmap(self):
        if not self._results:
            raise ValueError("No results to plot")
        return plot_heatmap(
            self._results,
            bins_input_tokens=len(self.input_lengths),
            bins_output_tokens=len(self.output_lengths),
            output_path=self._results.output_path,
        )
