# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-updating stats display for test runs.

Renders a compact table of running statistics that updates in-place during a run.
In Jupyter notebooks, uses an HTML table via IPython.display. In terminals, falls
back to a simple printed summary that overwrites itself.
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict

logger = logging.getLogger(__name__)

#: Default grouping of stat keys for display.  Each entry is
#: ``(group_name, tuple_of_substrings)``; a stat key is assigned to the first
#: group whose substring matches (case-insensitive).  Unmatched keys fall into
#: ``"Other"``.  The tuple order defines the column order in the rendered table.
DEFAULT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Throughput", ("rpm", "tps")),
    ("TTFT", ("ttft",)),
    ("TTLT", ("ttlt",)),
    ("Tokens", ("token",)),
    ("Errors", ("fail",)),
    ("Other", ("",)),
)

#: Default stats to show on the progress bar during a run.
#:
#: Each entry maps a short display label to a *stat spec*:
#:
#: * A plain string — the canonical key in ``RunningStats.to_stats()``
#:   (e.g. ``"failed_requests"``, ``"rpm"``, ``"time_to_first_token-p50"``).
#: * A ``(stat_key, "inv")`` tuple — display the reciprocal of the value
#:   (e.g. seconds-per-token → tokens-per-second).
DEFAULT_DISPLAY_STATS: dict[str, str | tuple[str, str]] = {
    "rpm": "rpm",
    "output_tps": "output_tps",
    "p50_ttft": "time_to_first_token-p50",
    "p90_ttft": "time_to_first_token-p90",
    "p50_ttlt": "time_to_last_token-p50",
    "p90_ttlt": "time_to_last_token-p90",
    "p50_tps": ("time_per_output_token-p50", "inv"),
    "input_tokens": "num_tokens_input-sum",
    "output_tokens": "num_tokens_output-sum",
    "fail": "failed_requests",
}


def _format_stat(key: str, value: float | int, *, invert: bool = False) -> str:
    """Format a single stat value as a human-readable string.

    Args:
        key: The canonical stat key (used to infer units).
        value: The raw numeric value.
        invert: If ``True``, display ``1/value`` (e.g. time → rate).

    Returns:
        A formatted string like ``"0.312s"``, ``"28.3 tok/s"``, or ``"83"``.
    """
    if invert and value > 0:
        return f"{1.0 / value:.1f} tok/s"
    if "tps" in key or "output_tps" in key:
        return f"{value:.1f} tok/s"
    if "time" in key or "ttft" in key or "ttlt" in key:
        return f"{value:.3f}s"
    if "rpm" in key:
        return f"{value:.1f}"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return f"{value:.1f}"


def _classify(
    key: str,
    groups: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_GROUPS,
) -> str:
    """Return the group name for a stat key based on substring matching.

    Matches the key (case-insensitive) against *groups*. The first matching
    pattern determines the group. Unmatched keys are placed in ``"Other"``.

    Args:
        key (str): The stat display label to classify (e.g. ``"p50_ttft"``).
        groups: Group definitions to match against. Defaults to
            :data:`DEFAULT_GROUPS`.

    Returns:
        str: The group name (e.g. ``"TTFT"``, ``"Throughput"``, ``"Other"``).
    """
    key_lower = key.lower()
    for group_name, patterns in groups:
        for pattern in patterns:
            if pattern and pattern in key_lower:
                return group_name
    return "Other"


def _group_stats(
    stats: dict[str, str],
    groups: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_GROUPS,
) -> OrderedDict[str, list[tuple[str, str]]]:
    """Organize stats into ordered groups for display.

    Each stat key is classified via :func:`_classify` and placed into the
    corresponding group. Groups are returned in the canonical order defined
    by *groups*, with empty groups omitted.

    Args:
        stats (dict[str, str]): Mapping of stat labels to formatted values.
        groups: Group definitions controlling classification and order.
            Defaults to :data:`DEFAULT_GROUPS`.

    Returns:
        OrderedDict[str, list[tuple[str, str]]]: Groups in display order, where
        each value is a list of ``(label, formatted_value)`` tuples.
    """
    buckets: dict[str, list[tuple[str, str]]] = {}
    for k, v in stats.items():
        group = _classify(k, groups)
        buckets.setdefault(group, []).append((k, v))
    group_order = [name for name, _ in groups]
    return OrderedDict((g, buckets[g]) for g in group_order if g in buckets)


def _in_notebook() -> bool:
    """Detect if we're running inside a Jupyter/IPython notebook.

    Returns:
        bool: ``True`` if the current IPython shell is a ``ZMQInteractiveShell``
        (i.e. a Jupyter kernel), ``False`` otherwise or if IPython is not
        installed.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


class LiveStatsDisplay:
    """Live-updating stats display that works in both notebooks and terminals.

    In Jupyter notebooks, renders a grouped HTML table that updates in-place.
    Stats are automatically organized into logical groups (Throughput, TTFT,
    TTLT, Tokens, Errors) based on their display label names.

    In terminals, prints a compact grouped multi-line block using ANSI escape
    codes to overwrite previous output.

    The display owns all alias mapping and formatting.  Callers pass raw
    numeric stats (e.g. from ``RunningStats.to_stats()``) and the display
    selects, aliases, formats, and groups them for presentation.

    Args:
        disabled (bool): If ``True``, all display calls are no-ops.
        groups: Group definitions controlling how display labels are classified
            and ordered.  Defaults to :data:`DEFAULT_GROUPS`.
        display_stats: Mapping of ``{display_label: stat_spec}`` controlling
            which stats to show and how to label them.  Each *stat_spec* is
            either a plain canonical key string (e.g. ``"time_to_first_token-p50"``)
            or a ``(key, "inv")`` tuple for reciprocal display.
            Defaults to :data:`DEFAULT_DISPLAY_STATS`.

    Example::

        display = LiveStatsDisplay()
        raw = running_stats.to_stats()
        display.update(raw)
        display.close()
    """

    def __init__(
        self,
        disabled: bool = False,
        groups: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_GROUPS,
        display_stats: dict[str, str | tuple[str, str]] | None = None,
    ):
        self._disabled = disabled
        self._groups = groups
        self._display_stats = (
            display_stats if display_stats is not None else DEFAULT_DISPLAY_STATS
        )
        self._is_notebook = _in_notebook()
        self._handle = None
        self._last_line_count = 0

    def format_stats(
        self,
        raw: dict[str, object],
    ) -> dict[str, str]:
        """Select and format raw stats for display.

        Picks the stats listed in ``self._display_stats`` from *raw*, applies
        alias renaming and formatting, and returns an ordered dict of
        ``{display_label: formatted_value}`` strings.

        Args:
            raw: Flat dictionary of raw numeric stats, as returned by
                ``RunningStats.to_stats()``.

        Returns:
            Ordered dict of ``{label: formatted_string}`` suitable for
            rendering.
        """
        if not raw:
            return {label: "—" for label in self._display_stats}

        info: dict[str, str] = {}
        for label, spec in self._display_stats.items():
            if isinstance(spec, tuple):
                key, modifier = spec[0], spec[1]
                invert = modifier == "inv"
            else:
                key = spec
                invert = False

            val = raw.get(key)
            if val is None:
                info[label] = "—"
                continue

            try:
                info[label] = _format_stat(key, float(val), invert=invert)
            except (TypeError, ValueError):
                info[label] = str(val)

        return info

    def update(
        self,
        raw_stats: dict[str, object],
        extra_prefix: str = "",
    ) -> None:
        """Refresh the display with new raw stats.

        Args:
            raw_stats: Flat dictionary of raw numeric stats from
                ``RunningStats.to_stats()``.
            extra_prefix (str): Optional prefix text shown before the table
                (e.g. ``"reqs=127"`` for time-bound runs).
        """
        if self._disabled:
            return

        formatted = self.format_stats(raw_stats)
        if not formatted:
            return

        if self._is_notebook:
            self._update_notebook(formatted, extra_prefix)
        else:
            self._update_terminal(formatted, extra_prefix)

    def _update_notebook(self, stats: dict[str, str], extra_prefix: str) -> None:
        """Render stats as a grouped HTML table in a Jupyter notebook.

        Groups stats into columns (Throughput, TTFT, TTLT, Tokens, Errors) and
        renders them as an HTML ``<table>`` that updates in-place via
        ``IPython.display``.

        Args:
            stats (dict[str, str]): Mapping of label to formatted value.
            extra_prefix (str): Optional text shown above the table.
        """
        from IPython.display import HTML, display

        groups = _group_stats(stats, self._groups)
        # All columns rendered side-by-side in a single table row
        max_rows = max(len(items) for items in groups.values())

        col_htmls = []
        for group_name, items in groups.items():
            col = (
                f"<th style='padding:2px 10px;font-size:11px;color:#888;"
                f"border-bottom:1px solid #ddd;text-align:left'>"
                f"{group_name}</th>"
            )
            rows = []
            for k, v in items:
                rows.append(
                    f"<td style='padding:1px 10px;font-size:12px'>"
                    f"<span style='color:#666'>{k}</span>"
                    f"&nbsp;&nbsp;"
                    f"<span style='font-family:monospace'>{v}</span>"
                    f"</td>"
                )
            # Pad shorter columns
            for _ in range(max_rows - len(items)):
                rows.append("<td></td>")
            col_htmls.append((col, rows))

        # Assemble: header row, then data rows
        header_row = "<tr>" + "".join(c[0] for c in col_htmls) + "</tr>"
        data_rows = ""
        for i in range(max_rows):
            data_rows += "<tr>" + "".join(c[1][i] for c in col_htmls) + "</tr>"

        prefix_html = (
            f"<span style='font-size:12px;font-family:monospace;color:#555'>"
            f"{extra_prefix}</span><br>"
            if extra_prefix
            else ""
        )
        html = (
            f"{prefix_html}"
            f"<table style='border-collapse:collapse;margin:4px 0'>"
            f"{header_row}{data_rows}</table>"
        )

        if self._handle is None:
            self._handle = display(HTML(html), display_id=True)
        else:
            self._handle.update(HTML(html))

    def _update_terminal(self, stats: dict[str, str], extra_prefix: str) -> None:
        """Render stats as grouped text lines in a terminal.

        Uses ANSI escape codes to erase the previous output and overwrite it
        with the updated stats, one line per group.

        Args:
            stats (dict[str, str]): Mapping of label to formatted value.
            extra_prefix (str): Optional text shown on the first line.
        """
        # Erase previous output
        if self._last_line_count > 0:
            sys.stderr.write(f"\033[{self._last_line_count}A\033[J")

        groups = _group_stats(stats, self._groups)
        lines = []
        if extra_prefix:
            lines.append(f"  {extra_prefix}")
        for group_name, items in groups.items():
            values = "  ".join(f"{k}={v}" for k, v in items)
            lines.append(f"  {group_name}: {values}")

        output = "\n".join(lines)
        sys.stderr.write(output + "\n")
        sys.stderr.flush()
        self._last_line_count = len(lines)

    def close(self) -> None:
        """Clean up the display.

        In terminal mode, erases the stats block using ANSI escape codes.
        In notebook mode, the HTML output remains visible.
        """
        if self._disabled:
            return
        # In terminal, erase the stats block
        if not self._is_notebook and self._last_line_count > 0:
            sys.stderr.write(f"\033[{self._last_line_count}A\033[J")
            sys.stderr.flush()
            self._last_line_count = 0
