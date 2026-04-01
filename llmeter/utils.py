# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import bisect
from datetime import datetime, timezone
from itertools import filterfalse
from math import isnan
from statistics import StatisticsError, mean, median, quantiles
from typing import Any, Sequence, overload

from upath import UPath
from upath.types import ReadablePathLike, WritablePathLike


class DeferredError:
    """Stores an exception and raises it at a later time if this object is accessed in any way.

    Useful to allow soft-dependencies on imports, so that the ImportError can be raised again
    later if code actually relies on the missing library.
    Lifted from https://github.com/aws/sagemaker-python-sdk/blob/e626647692d136155d72cf5b100043af41ab6c43/src/sagemaker/utils.py#L788

    Example::

        try:
            import obscurelib
        except ImportError as e:
            logger.warning("Failed to import obscurelib. Obscure features will not work.")
            obscurelib = DeferredError(e)
    """

    def __init__(self, exception):
        # # Ensure the exception is a BaseException instance
        # if isinstance(exception, BaseException):
        #     self.exc = exception
        # else:
        #     # If it's not a BaseException, wrap it in an ImportError
        #     self.exc = ImportError(str(exception))
        self.exc = exception

    def __getattr__(self, name: str):
        """Called by Python interpreter before using any method or property on the object.

        So this will short-circuit essentially any access to this object.

        Args:
            name: The name of the attribute being accessed
        """
        raise self.exc


def summary_stats_from_list(
    data: Sequence[int | float],
    percentiles: Sequence[int] = (50, 90, 99),
) -> dict[str, int | float]:
    """Calculate summary statistics for a sequence of numbers in pure Python

    Args:
        data: Sequence of numbers
        percentiles: Integer percentiles from 1-99
    Returns:
        stats: Dictionary of descriptive statistics including "average" (the arithmetic mean of the
            input `data`), and the requested `percentiles` in format "p50", "p90", "p99", etc.
    """
    clean_data = list(filterfalse(isnan, data))
    try:
        result = dict(average=mean(clean_data))
        # Rather than always using n=100-based quantiles, we'll adapt between a few specific bases
        # depending on the requested percentiles - and calculate these splits only as needed:
        q_bases: dict[int, Any] = {k: None for k in [4, 10, 100]}
        for p in percentiles:
            if len(clean_data) == 1:
                result[f"p{p}"] = clean_data[0]
            elif p == 50:
                result[f"p{p}"] = median(clean_data)
            else:
                try:
                    q_base = next(k for k in q_bases if p % (100 / k) == 0)
                except StopIteration as e:
                    raise ValueError(
                        f"Invalid percentile {p} must be a whole number between 1-99"
                    ) from e
                if q_bases[q_base] is None:
                    q_bases[q_base] = quantiles(clean_data, n=q_base)
                result[f"p{p}"] = q_bases[q_base][int(p * q_base / 100) - 1]
        return result

    except StatisticsError:
        return {}


class RunningStats:
    """Accumulate summary statistics incrementally from individual responses.

    Maintains sorted value lists per metric so that percentiles (p50, p90, p99),
    averages, and sums can be computed at any point — both mid-run (for live
    progress-bar display via :meth:`snapshot`) and at the end of a run (for the
    final :class:`~llmeter.results.Result` stats via :meth:`to_stats`).

    Args:
        metrics: Names of numeric response fields to track (e.g.
            ``"time_to_first_token"``, ``"num_tokens_output"``).

    Example::

        rs = RunningStats(metrics=["time_to_first_token", "time_to_last_token"])
        rs.update({"time_to_first_token": 0.3, "time_to_last_token": 0.8})
        rs.update({"time_to_first_token": 0.5, "time_to_last_token": 1.2, "error": None})
        rs.to_stats()
        # {'failed_requests': 0, ..., 'time_to_first_token-p50': 0.4, ...}
    """

    #: Default stats shown on the progress bar during a run.
    #: Each entry maps a short display label to a spec:
    #:
    #: * ``(metric_name, aggregation)`` — aggregation can be ``"p50"``, ``"p90"``,
    #:   ``"p99"``, ``"average"``, or ``"sum"``.
    #: * ``(metric_name, aggregation, "inv")`` — same as above but displays the
    #:   reciprocal (e.g. seconds-per-token → tokens-per-second).
    #: * The literal string ``"failed"`` for the running failure count.
    DEFAULT_SNAPSHOT_STATS: dict[str, tuple[str, ...] | str] = {
        "p50_ttft": ("time_to_first_token", "p50"),
        "p90_ttft": ("time_to_first_token", "p90"),
        "p50_ttlt": ("time_to_last_token", "p50"),
        "p90_ttlt": ("time_to_last_token", "p90"),
        "p50_tps": ("time_per_output_token", "p50", "inv"),
        "input_tokens": ("num_tokens_input", "sum"),
        "output_tokens": ("num_tokens_output", "sum"),
        "fail": "failed",
    }

    def __init__(self, metrics: Sequence[str]):
        self._metrics = list(metrics)
        self._count = 0
        self._failed = 0
        self._sums: dict[str, float] = {m: 0.0 for m in metrics}
        self._values: dict[str, list[float]] = {m: [] for m in metrics}

    def update(self, response_dict: dict[str, Any]) -> None:
        """Record one response's metric values.

        Call this once per :class:`~llmeter.endpoints.base.InvocationResponse`
        (typically via ``response.to_dict()``).  The method extracts each tracked
        metric from *response_dict*, skipping ``None`` and ``NaN`` values, and
        increments the failure counter when an ``"error"`` key is present.

        Args:
            response_dict: A flat dictionary of response fields, as returned by
                ``InvocationResponse.to_dict()``.

        Example::

            rs = RunningStats(metrics=["time_to_first_token"])
            rs.update({"time_to_first_token": 0.42, "error": None})
            rs.update({"time_to_first_token": None, "error": "timeout"})
            assert rs._failed == 1
        """
        self._count += 1
        if response_dict.get("error") is not None:
            self._failed += 1
        for m in self._metrics:
            val = response_dict.get(m)
            if val is not None and not (isinstance(val, float) and isnan(val)):
                self._sums[m] += val
                bisect.insort(self._values[m], val)

    def to_stats(
        self,
        total_requests: int | None = None,
        total_test_time: float | None = None,
        result_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute all accumulated statistics as raw numeric values.

        This is the single source of truth for stats computation.  It is called
        once at the end of a run (with all three optional arguments) to produce
        the full ``Result.stats`` dict, and also called internally by
        :meth:`snapshot` (without arguments) for mid-run progress display.

        Args:
            total_requests: Total number of requests across all clients.  When
                provided, enables ``failed_requests_rate`` and
                ``requests_per_minute`` computation.
            total_test_time: Wall-clock duration of the run in seconds.  When
                provided, enables throughput metrics (requests/min, tokens/min).
            result_dict: Base key-value pairs to include in the output (typically
                from ``Result.to_dict()``).  When ``None``, only metric
                aggregations and failure counts are returned.

        Returns:
            A flat dictionary of statistics.  Keys include:

            * ``failed_requests``, ``failed_requests_rate``, ``requests_per_minute``
            * ``total_input_tokens``, ``total_output_tokens``
            * ``average_input_tokens_per_minute``, ``average_output_tokens_per_minute``
            * ``{metric}-{agg}`` for each tracked metric and each aggregation
              (``average``, ``p50``, ``p90``, ``p99``).

        Example::

            rs = RunningStats(metrics=["time_to_first_token", "num_tokens_output"])
            for resp in responses:
                rs.update(resp.to_dict())

            # Mid-run (no run-level context):
            partial = rs.to_stats()
            partial["time_to_first_token-p50"]  # 0.312

            # End of run (full Result.stats schema):
            full = rs.to_stats(
                total_requests=100,
                total_test_time=42.5,
                result_dict=result.to_dict(),
            )
            full["requests_per_minute"]  # 141.2
        """
        stats: dict[str, Any] = {}
        if result_dict is not None:
            stats.update(result_dict)

        # Run-level stats
        stats["failed_requests"] = self._failed
        stats["failed_requests_rate"] = total_requests and self._failed / total_requests
        stats["requests_per_minute"] = (
            total_test_time and total_requests / total_test_time * 60
            if total_requests
            else None
        )
        stats["total_input_tokens"] = self._sums.get("num_tokens_input", 0)
        stats["total_output_tokens"] = self._sums.get("num_tokens_output", 0)
        stats["average_input_tokens_per_minute"] = (
            total_test_time and stats["total_input_tokens"] / total_test_time * 60
        )
        stats["average_output_tokens_per_minute"] = (
            total_test_time and stats["total_output_tokens"] / total_test_time * 60
        )

        # Per-metric aggregations
        for m in self._metrics:
            agg = summary_stats_from_list(self._values.get(m, []))
            for j, v in agg.items():
                stats[f"{m}-{j}"] = v

        return stats

    def snapshot(
        self,
        fields: dict[str, tuple[str, ...] | str] | None = None,
    ) -> dict[str, str]:
        """Format a subset of :meth:`to_stats` for progress-bar display.

        Calls :meth:`to_stats` internally and picks only the requested fields,
        formatting each value as a human-readable string.

        Args:
            fields: Mapping of ``{display_label: spec}``.  Each *spec* is one of:

                * ``(metric, aggregation)`` — a 2-tuple where *metric* is a tracked
                  metric name and *aggregation* is ``"p50"``, ``"p90"``, ``"p99"``,
                  ``"average"``, or ``"sum"``.
                * ``(metric, aggregation, "inv")`` — a 3-tuple; same as above but
                  the value is inverted before display (e.g. seconds-per-token →
                  tokens-per-second).
                * ``"failed"`` — the literal string; shows the running failure count.

                Defaults to :attr:`DEFAULT_SNAPSHOT_STATS` when ``None``.

        Returns:
            An ordered dict of ``{label: formatted_value}`` strings suitable for
            ``tqdm.set_postfix()``.

        Example::

            # Use defaults:
            rs.snapshot()
            # {'p50_ttft': '0.312s', 'p90_ttlt': '1.203s', ..., 'fail': '0'}

            # Custom selection — only p99 latency and failures:
            rs.snapshot({
                "p99_ttlt": ("time_to_last_token", "p99"),
                "fail": "failed",
            })
            # {'p99_ttlt': '2.105s', 'fail': '1'}

            # Inverted metric — tokens per second from time_per_output_token:
            rs.snapshot({
                "tps": ("time_per_output_token", "p50", "inv"),
            })
            # {'tps': '28.3 tok/s'}
        """
        if self._count == 0:
            return {}

        if fields is None:
            fields = self.DEFAULT_SNAPSHOT_STATS

        raw = self.to_stats()

        info: dict[str, str] = {}
        for label, spec in fields.items():
            if spec == "failed":
                info[label] = str(self._failed)
                continue

            metric = spec[0]
            agg = spec[1]
            invert = len(spec) > 2 and spec[2] == "inv"

            if agg == "sum":
                info[label] = f"{self._sums.get(metric, 0):.0f}"
                continue

            val = raw.get(f"{metric}-{agg}")
            if val is None:
                continue

            if invert and val > 0:
                info[label] = f"{1.0 / val:.1f} tok/s"
            elif "time" in metric:
                info[label] = f"{val:.3f}s"
            else:
                info[label] = f"{val:.1f}"

        return info


def now_utc() -> datetime:
    """Returns the current UTC datetime.

    Returns:
        datetime: Current UTC datetime object
    """
    return datetime.now(timezone.utc)


@overload
def ensure_path(path: ReadablePathLike | WritablePathLike) -> UPath: ...


@overload
def ensure_path(path: None) -> None: ...


def ensure_path(
    path: ReadablePathLike | WritablePathLike | None,
) -> UPath | None:
    """Normalize a path-like argument to a UPath instance.

    Converts strings, os.PathLike objects, and UPath instances into a
    consistent UPath representation. Passes through None unchanged.

    Args:
        path: A string, path-like object, or None.

    Returns:
        A UPath instance, or None if the input was None.
    """
    if path is None:
        return None
    return UPath(path)
