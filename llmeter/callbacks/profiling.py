# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiling callback for LLMeter runs.

Collects per-request phase timings and produces a summary report after the run,
breaking down where time is spent: server-side prefill (TTFT), server-side
generation (decode), client-side runner overhead, and callback overhead.

Optionally saves:
- ``profile_report.json`` — aggregated summary with statistics
- ``profile_invocations.jsonl`` — per-invocation detail (one JSON object per line)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from upath.types import ReadablePathLike, WritablePathLike

from ..endpoints.base import InvocationResponse
from ..results import Result
from ..runner import _RunConfig
from ..utils import ensure_path, summary_stats_from_list
from .base import Callback

logger = logging.getLogger(__name__)


@dataclass
class InvocationProfile:
    """Per-invocation timing and metadata captured during a run.

    This is the per-request record saved to ``profile_invocations.jsonl``.
    """

    #: Sequence number within the run (0-indexed, in order of completion).
    sequence: int = 0
    #: Unique request ID (from InvocationResponse.id), if available.
    request_id: str | None = None
    #: Wall-clock time when the request was sent (ISO format).
    request_time: str | None = None
    #: Offset in seconds from the start of the run to when this request completed.
    offset_from_run_start: float = 0.0

    # --- Timing phases ---
    #: Time to first token (seconds). Server-side prefill + network round-trip.
    ttft: float | None = None
    #: Time to last token (seconds). Total response time.
    ttlt: float | None = None
    #: Generation time = TTLT - TTFT (seconds). Server-side decode (token generation).
    generation_time: float | None = None
    #: Time per output token (seconds/token) = generation_time / tokens_output.
    time_per_output_token: float | None = None
    #: Output token throughput (tokens/second) = tokens_output / generation_time.
    output_tokens_per_second: float | None = None

    # --- Token counts ---
    #: Number of input tokens.
    tokens_input: int | None = None
    #: Number of output tokens.
    tokens_output: int | None = None
    #: Number of input tokens served from cache (prompt caching).
    tokens_input_cached: int | None = None
    #: Whether this request had a cache hit (any tokens served from cache).
    cache_hit: bool = False
    #: Cache hit ratio (cached_tokens / total_input_tokens).
    cache_hit_ratio: float | None = None
    #: Number of reasoning/thinking tokens (subset of output tokens).
    tokens_output_reasoning: int | None = None

    # --- Error/retry info ---
    #: Error message, or None if successful.
    error: str | None = None
    #: Number of retries for this request.
    retries: int | None = None

    # --- Overhead ---
    #: Time spent in after_invoke callbacks (seconds). Measures profiling overhead.
    callback_overhead: float = 0.0


@dataclass
class ProfileReport:
    """Aggregated profiling report produced after a run.

    Contains both timing breakdowns and richer metadata like cache hit rates,
    retry statistics, and throughput analysis.
    """

    # --- Time accounting ---
    total_wall_clock: float = 0.0
    total_api_time: float = 0.0
    runner_overhead: float = 0.0
    api_time_fraction: float = 0.0
    runner_overhead_fraction: float = 0.0

    # --- Request counts ---
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    total_retries: int = 0

    # --- Cache stats ---
    cache_hits: int = 0
    cache_hit_rate: float = 0.0
    total_tokens_cached: int = 0

    # --- Token totals ---
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_tokens_reasoning: int = 0

    # --- Timing statistics ---
    ttft_stats: dict[str, float] = field(default_factory=dict)
    generation_time_stats: dict[str, float] = field(default_factory=dict)
    tpot_stats: dict[str, float] = field(default_factory=dict)
    tokens_per_second_stats: dict[str, float] = field(default_factory=dict)
    callback_overhead_stats: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "╭─── LLMeter Profile Report ───╮",
            f"  Wall clock:        {self.total_wall_clock:.3f}s",
            f"  API time:          {self.total_api_time:.3f}s ({self.api_time_fraction:.1%})",
            f"  Runner overhead:   {self.runner_overhead:.3f}s"
            f" ({self.runner_overhead_fraction:.1%})",
            f"  Requests:          {self.successful_requests} ok,"
            f" {self.failed_requests} failed",
        ]

        if self.retried_requests > 0:
            lines.append(
                f"  Retries:           {self.total_retries}"
                f" ({self.retried_requests} requests)"
            )

        if self.total_tokens_input > 0:
            lines.append("")
            lines.append(
                f"  Tokens in/out:     {self.total_tokens_input:,}"
                f" / {self.total_tokens_output:,}"
            )
            if self.total_tokens_reasoning > 0:
                lines.append(
                    f"  Reasoning tokens:  {self.total_tokens_reasoning:,}"
                    f" ({self.total_tokens_reasoning / max(self.total_tokens_output, 1):.0%}"
                    " of output)"
                )
            if self.cache_hits > 0:
                lines.append(
                    f"  Cache hits:        {self.cache_hits}"
                    f" ({self.cache_hit_rate:.0%}),"
                    f" {self.total_tokens_cached:,} tokens cached"
                )

        lines.append("")
        if self.ttft_stats:
            lines.append("  TTFT (server prefill + network):")
            lines.append(
                f"    avg={self.ttft_stats.get('average', 0):.3f}s  "
                f"p50={self.ttft_stats.get('p50', 0):.3f}s  "
                f"p90={self.ttft_stats.get('p90', 0):.3f}s  "
                f"p99={self.ttft_stats.get('p99', 0):.3f}s"
            )
        if self.generation_time_stats:
            lines.append("  Generation (server decode):")
            lines.append(
                f"    avg={self.generation_time_stats.get('average', 0):.3f}s  "
                f"p50={self.generation_time_stats.get('p50', 0):.3f}s  "
                f"p90={self.generation_time_stats.get('p90', 0):.3f}s  "
                f"p99={self.generation_time_stats.get('p99', 0):.3f}s"
            )
        if self.tpot_stats:
            lines.append("  Time per output token:")
            lines.append(
                f"    avg={self.tpot_stats.get('average', 0) * 1000:.1f}ms  "
                f"p50={self.tpot_stats.get('p50', 0) * 1000:.1f}ms  "
                f"p90={self.tpot_stats.get('p90', 0) * 1000:.1f}ms"
            )
        if self.tokens_per_second_stats:
            lines.append("  Output throughput:")
            lines.append(
                f"    avg={self.tokens_per_second_stats.get('average', 0):.1f} tok/s  "
                f"p50={self.tokens_per_second_stats.get('p50', 0):.1f} tok/s"
            )
        if self.callback_overhead_stats:
            avg_ms = self.callback_overhead_stats.get("average", 0) * 1000
            p99_ms = self.callback_overhead_stats.get("p99", 0) * 1000
            lines.append(
                f"  Callback overhead:  avg={avg_ms:.2f}ms  p99={p99_ms:.2f}ms"
            )

        lines.append("╰──────────────────────────────╯")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ProfileReport(wall_clock={self.total_wall_clock:.3f}s, "
            f"api_time={self.total_api_time:.3f}s ({self.api_time_fraction:.1%}), "
            f"overhead={self.runner_overhead:.3f}s ({self.runner_overhead_fraction:.1%}), "
            f"requests={self.successful_requests}ok/{self.failed_requests}fail)"
        )


@dataclass
class ProfileCallback(Callback):
    """Profiling callback that collects per-invocation timing data and produces a report.

    This callback instruments each request to capture detailed phase timings,
    token counts, cache hit information, retries, and throughput metrics.

    After the run completes:
    1. A `ProfileReport` summary is generated and contributed to ``result.stats``
    2. Optionally printed to stdout (``print_report=True``)
    3. Optionally saved to disk (``save_report=True``):
       - ``profile_report.json`` — aggregated summary
       - ``profile_invocations.jsonl`` — per-invocation detail records

    Args:
        print_report: If True (default), print the report summary to stdout after the run.
        save_report: If True (default), save report files to the result's output directory.
            Only takes effect when the result has an ``output_path`` set.

    Example::

        from llmeter.callbacks import ProfileCallback
        from llmeter.runner import Runner

        profiler = ProfileCallback()
        runner = Runner(endpoint=endpoint, callbacks=[profiler])
        result = await runner.run(payload=payload, n_requests=20)

        # Access the report
        print(profiler.report)

        # Inspect per-invocation data
        for inv in profiler.invocation_profiles:
            if inv.cache_hit:
                print(f"Request {inv.sequence}: cached {inv.tokens_input_cached} tokens")

        # Per-invocation data is also saved as profile_invocations.jsonl

    Contributed stats (prefixed with ``profile_``):

    - ``profile_total_wall_clock``: Total run wall-clock time (seconds)
    - ``profile_total_api_time``: Sum of TTLT across successful requests (seconds)
    - ``profile_runner_overhead``: Wall clock minus API time (seconds)
    - ``profile_api_time_fraction``: Fraction of wall clock in API calls
    - ``profile_runner_overhead_fraction``: Fraction of wall clock in overhead
    - ``profile_cache_hit_rate``: Fraction of requests with cached input tokens
    - ``profile_total_retries``: Total retry count across all requests
    - ``profile_ttft-average``, ``-p50``, ``-p90``, ``-p99``: TTFT statistics
    - ``profile_generation_time-average``, ``-p50``, ``-p90``, ``-p99``: Generation time stats
    - ``profile_tpot-average``, ``-p50``, ``-p90``, ``-p99``: Time per output token stats
    - ``profile_callback_overhead-average``, ``-p50``, ``-p90``, ``-p99``: Callback overhead
    - ``profile_tokens_per_second-average``, ``-p50``, ``-p90``, ``-p99``: Token throughput
    """

    print_report: bool = True
    save_report: bool = True

    def __post_init__(self):
        self._invocation_profiles: list[InvocationProfile] = []
        self._pending: dict = {}
        self._sequence: int = 0
        self._run_start_time: float = 0.0
        self._run_end_time: float = 0.0
        self._report: ProfileReport | None = None

    def __getstate__(self):
        """Support pickling/deepcopy by returning only serializable state."""
        return {
            "print_report": self.print_report,
            "save_report": self.save_report,
            "_invocation_profiles": self._invocation_profiles,
            "_sequence": self._sequence,
            "_run_start_time": self._run_start_time,
            "_run_end_time": self._run_end_time,
        }

    def __setstate__(self, state):
        """Restore from pickle/deepcopy."""
        self.print_report = state["print_report"]
        self.save_report = state.get("save_report", True)
        self._invocation_profiles = state["_invocation_profiles"]
        self._sequence = state["_sequence"]
        self._run_start_time = state["_run_start_time"]
        self._run_end_time = state["_run_end_time"]
        self._pending = {}
        self._report = None

    @property
    def invocation_profiles(self) -> list[InvocationProfile]:
        """Access the raw per-invocation profile data."""
        return self._invocation_profiles

    @property
    def report(self) -> ProfileReport | None:
        """The generated profile report (available after the run completes)."""
        return self._report

    async def before_run(self, run_config: _RunConfig) -> None:
        """Reset state and record run start time."""
        self._invocation_profiles = []
        self._pending = {}
        self._sequence = 0
        self._report = None
        self._run_start_time = time.perf_counter()

    async def before_invoke(self, payload: dict) -> None:
        """Record the timestamp just before the endpoint is called."""
        self._pending = {"before_invoke_time": time.perf_counter()}

    async def after_invoke(self, response: InvocationResponse) -> None:
        """Capture full invocation profile from the response."""
        t_enter = time.perf_counter()

        # Compute derived timing fields
        ttft = response.time_to_first_token
        ttlt = response.time_to_last_token
        generation_time = None
        tpot = None
        tps = None

        if ttft is not None and ttlt is not None and ttlt > ttft:
            generation_time = ttlt - ttft
            if response.num_tokens_output and response.num_tokens_output > 0:
                tpot = generation_time / response.num_tokens_output
                tps = response.num_tokens_output / generation_time

        # Cache info
        cached = response.num_tokens_input_cached or 0
        has_cache_hit = cached > 0
        cache_ratio = None
        if (
            has_cache_hit
            and response.num_tokens_input
            and response.num_tokens_input > 0
        ):
            cache_ratio = cached / response.num_tokens_input

        # Request time formatting
        req_time_str = None
        if response.request_time is not None:
            req_time_str = (
                response.request_time.isoformat()
                if isinstance(response.request_time, datetime)
                else str(response.request_time)
            )

        cb_overhead = time.perf_counter() - t_enter

        profile = InvocationProfile(
            sequence=self._sequence,
            request_id=response.id,
            request_time=req_time_str,
            offset_from_run_start=t_enter - self._run_start_time,
            ttft=ttft,
            ttlt=ttlt,
            generation_time=generation_time,
            time_per_output_token=tpot,
            output_tokens_per_second=tps,
            tokens_input=response.num_tokens_input,
            tokens_output=response.num_tokens_output,
            tokens_input_cached=response.num_tokens_input_cached,
            cache_hit=has_cache_hit,
            cache_hit_ratio=cache_ratio,
            tokens_output_reasoning=response.num_tokens_output_reasoning,
            error=response.error,
            retries=response.retries,
            callback_overhead=cb_overhead,
        )

        self._invocation_profiles.append(profile)
        self._sequence += 1
        self._pending = {}

    async def after_run(self, result: Result) -> None:
        """Generate the profile report, contribute stats, and optionally save to disk."""
        self._run_end_time = time.perf_counter()
        self._report = self._compute_report()

        # Contribute stats to result
        stats = self._report_to_stats(self._report)
        result._update_contributed_stats(stats)

        if self.print_report:
            print(self._report)

        if self.save_report and result.output_path:
            self._save_report(result.output_path)
            self._save_invocations(result.output_path)

    def _compute_report(self) -> ProfileReport:
        """Compute the profile report from collected invocation profiles."""
        report = ProfileReport()
        report.total_wall_clock = self._run_end_time - self._run_start_time
        report.total_requests = len(self._invocation_profiles)

        successful = [p for p in self._invocation_profiles if p.error is None]
        failed = [p for p in self._invocation_profiles if p.error is not None]
        report.successful_requests = len(successful)
        report.failed_requests = len(failed)

        # Retry stats (across all requests, including failed)
        retried = [p for p in self._invocation_profiles if p.retries and p.retries > 0]
        report.retried_requests = len(retried)
        report.total_retries = sum(p.retries for p in retried if p.retries)

        if not successful:
            return report

        # Cache stats
        cache_hits = [p for p in successful if p.cache_hit]
        report.cache_hits = len(cache_hits)
        report.cache_hit_rate = len(cache_hits) / len(successful) if successful else 0.0
        report.total_tokens_cached = sum(p.tokens_input_cached or 0 for p in successful)

        # Token totals
        report.total_tokens_input = sum(p.tokens_input or 0 for p in successful)
        report.total_tokens_output = sum(p.tokens_output or 0 for p in successful)
        report.total_tokens_reasoning = sum(
            p.tokens_output_reasoning or 0 for p in successful
        )

        # Timing metrics
        ttft_values: list[float] = []
        generation_times: list[float] = []
        tpot_values: list[float] = []
        tps_values: list[float] = []
        cb_overhead_values: list[float] = []

        for p in successful:
            if p.ttft is not None:
                ttft_values.append(p.ttft)
            if p.generation_time is not None:
                generation_times.append(p.generation_time)
            if p.time_per_output_token is not None:
                tpot_values.append(p.time_per_output_token)
            if p.output_tokens_per_second is not None:
                tps_values.append(p.output_tokens_per_second)
            cb_overhead_values.append(p.callback_overhead)

        # Total API time = sum of TTLT for successful requests
        api_times = [p.ttlt for p in successful if p.ttlt is not None]
        report.total_api_time = sum(api_times)
        report.runner_overhead = max(
            0.0, report.total_wall_clock - report.total_api_time
        )

        if report.total_wall_clock > 0:
            report.api_time_fraction = report.total_api_time / report.total_wall_clock
            report.runner_overhead_fraction = (
                report.runner_overhead / report.total_wall_clock
            )

        # Compute summary statistics for each metric
        if ttft_values:
            report.ttft_stats = summary_stats_from_list(ttft_values)
        if generation_times:
            report.generation_time_stats = summary_stats_from_list(generation_times)
        if tpot_values:
            report.tpot_stats = summary_stats_from_list(tpot_values)
        if tps_values:
            report.tokens_per_second_stats = summary_stats_from_list(tps_values)
        if cb_overhead_values:
            report.callback_overhead_stats = summary_stats_from_list(cb_overhead_values)

        return report

    @staticmethod
    def _report_to_stats(report: ProfileReport) -> dict[str, float | int]:
        """Convert a ProfileReport into a flat dict suitable for result.stats."""
        stats: dict[str, float | int] = {
            "profile_total_wall_clock": report.total_wall_clock,
            "profile_total_api_time": report.total_api_time,
            "profile_runner_overhead": report.runner_overhead,
            "profile_api_time_fraction": report.api_time_fraction,
            "profile_runner_overhead_fraction": report.runner_overhead_fraction,
            "profile_successful_requests": report.successful_requests,
            "profile_failed_requests": report.failed_requests,
            "profile_retried_requests": report.retried_requests,
            "profile_total_retries": report.total_retries,
            "profile_cache_hits": report.cache_hits,
            "profile_cache_hit_rate": report.cache_hit_rate,
            "profile_total_tokens_cached": report.total_tokens_cached,
            "profile_total_tokens_input": report.total_tokens_input,
            "profile_total_tokens_output": report.total_tokens_output,
            "profile_total_tokens_reasoning": report.total_tokens_reasoning,
        }

        for metric_name, metric_stats in [
            ("profile_ttft", report.ttft_stats),
            ("profile_generation_time", report.generation_time_stats),
            ("profile_tpot", report.tpot_stats),
            ("profile_callback_overhead", report.callback_overhead_stats),
            ("profile_tokens_per_second", report.tokens_per_second_stats),
        ]:
            for agg, value in metric_stats.items():
                stats[f"{metric_name}-{agg}"] = value

        return stats

    def _save_report(
        self,
        output_path: WritablePathLike,
        file_name: str = "profile_report.json",
    ) -> None:
        """Save the profile report as JSON to the result output directory.

        Args:
            output_path: Directory where the report file will be written.
            file_name: Name of the report file. Defaults to ``profile_report.json``.
        """
        if self._report is None:
            return

        out_dir = ensure_path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / file_name

        report_data = asdict(self._report)
        with report_path.open("w") as f:
            json.dump(report_data, f, indent=2)

        logger.info("Profile report saved to %s", report_path)

    def _save_invocations(
        self,
        output_path: WritablePathLike,
        file_name: str = "profile_invocations.jsonl",
    ) -> None:
        """Save per-invocation profiles as JSONL to the result output directory.

        Each line is a JSON object representing one invocation's profile data,
        suitable for loading into pandas or custom analysis tools.

        Args:
            output_path: Directory where the file will be written.
            file_name: Name of the file. Defaults to ``profile_invocations.jsonl``.
        """
        if not self._invocation_profiles:
            return

        out_dir = ensure_path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        invocations_path = out_dir / file_name

        with invocations_path.open("w") as f:
            for profile in self._invocation_profiles:
                f.write(json.dumps(asdict(profile)) + "\n")

        logger.info(
            "Profile invocations (%d records) saved to %s",
            len(self._invocation_profiles),
            invocations_path,
        )

    def save_to_file(self, path: WritablePathLike) -> None:
        """Save this ProfileCallback configuration to file."""
        out_path = ensure_path(path)
        config = {
            "type": "ProfileCallback",
            "print_report": self.print_report,
            "save_report": self.save_report,
        }
        with out_path.open("w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def _load_from_file(cls, path: ReadablePathLike) -> "ProfileCallback":
        """Load a ProfileCallback from file."""
        in_path = ensure_path(path)
        with in_path.open("r") as f:
            config = json.load(f)

        return cls(
            print_report=config.get("print_report", True),
            save_report=config.get("save_report", True),
        )
