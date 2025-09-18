# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from typing import Callable, Literal

from tqdm.auto import tqdm
from upath import UPath as Path

from llmeter.callbacks.base import Callback
from llmeter.results import Result

from .endpoints.base import Endpoint
from .plotting import plot_heatmap, plot_load_test_results, color_sequences
from .prompt_utils import CreatePromptCollection
from .runner import Runner
from .tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# using a custom env variable because the TQDM one (https://github.com/tqdm/tqdm/issues/612#issuecomment-2015702344) doesn't work reliably
_disable_tqdm = False
if os.getenv("LLMETER_DISABLE_ALL_PROGRESS_BARS") == "1":
    logger.info("Disabling tqdm progress bars")
    _disable_tqdm = True


@dataclass
class LoadTestResult:
    results: dict[int, Result]
    test_name: str
    output_path: os.PathLike | str | None = None

    def plot_results(self, show: bool = True, format: Literal["html", "png"] = "html"):
        figs = plot_load_test_results(self)

        # add individual color sequence for each plot
        c_seqs = [
            color_sequences.Bluered,
            color_sequences.Turbo,
            color_sequences.Sunsetdark_r,
            color_sequences.Blackbody,
            color_sequences.Viridis,
            color_sequences.Plasma,
        ]

        for i, (_, f) in enumerate(figs.items()):
            f.update_layout(colorway=c_seqs[i % len(c_seqs)])

        output_path = Path(self.output_path)
        if output_path:
            # save figure to the output path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            for k, f in figs.items():
                if format == "html":
                    f.write_html(output_path / f"{k}.{format}")
                else:
                    f.write_image(output_path / f"{k}.{format}")

        if show:
            [f.show() for _, f in figs.items()]
        return figs

    @classmethod
    def load(cls, load_path: Path | str | None, test_name: str | None = None) -> "LoadTestResult":
        """Load test results from a directory.

        Args:
            load_path: Directory path containing the load test results subdirectories
            test_name: Optional name for the test. If not provided, will use the directory name

        Returns:
            LoadTestResult: A LoadTestResult object containing the loaded results

        Raises:
            FileNotFoundError: If load_path does not exist or is None/empty
            ValueError: If no results are found in the directory
        """
        if not load_path:
            raise FileNotFoundError("Load path cannot be None or empty")

        if isinstance(load_path, str):
            load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Load path {load_path} does not exist")

        results = [Result.load(x) for x in load_path.iterdir() if x.is_dir()]

        if not results:
            raise ValueError(f"No results found in {load_path}")

        return LoadTestResult(
            results={r.clients: r for r in results},
            test_name=test_name or load_path.name,
            output_path=load_path.parent,
        )


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
    callbacks: list[Callback] | None = None

    def __post_init__(self) -> None:
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
        _runner = Runner(
            endpoint=self.endpoint, tokenizer=self.tokenizer, output_path=output_path
        )

        self._results = [
            await _runner.run(
                payload=self.payload,
                clients=c,
                n_requests=self._get_n_requests(c),
                run_name=f"{c:05.0f}-clients",
                callbacks=self.callbacks,
            )
            for c in tqdm(
                self.sequence_of_clients, desc="Configurations", disable=_disable_tqdm
            )
        ]
        # return self._results
        return LoadTestResult(
            results={r.clients: r for r in self._results},
            test_name=self._test_name,
            output_path=output_path,
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
            output_path=Path(self.output_path),
            tokenizer=self.tokenizer,
        )

    async def run(self, output_path=None):
        heatmap_results = await self._runner.run(
            payload=self.payload,
            clients=self.clients,
            n_requests=len(self.input_lengths)
            * len(self.output_lengths)
            * self.requests_per_combination
            // self.clients,
            output_path=Path(output_path) or self.output_path,
        )
        self._results = heatmap_results
        return heatmap_results

    def plot_heatmaps(
        self, n_bins_x: int | None, n_bins_y: int | None, show: bool = True
    ):
        if not self._results:
            raise ValueError("No results to plot")
        f1 = plot_heatmap(
            self._results,
            "time_to_first_token",
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            # output_path=self._results.output_path,
            show_scatter=True,
        )

        f2 = plot_heatmap(
            self._results,
            "time_to_last_token",
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            # output_path=self._results.output_path,
            show_scatter=True,
        )

        if show:
            f1.show()
            f2.show()

        return f1, f2
