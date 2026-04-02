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

# Mapping from key substrings to (group_name, display_order).
# Stats are grouped by the first matching pattern; unmatched keys go to "Other".
_GROUP_PATTERNS: list[tuple[str, str]] = [
    ("rpm", "Throughput"),
    ("tps", "Throughput"),
    ("ttft", "TTFT"),
    ("ttlt", "TTLT"),
    ("token", "Tokens"),
    ("fail", "Errors"),
]

_GROUP_ORDER = ["Throughput", "TTFT", "TTLT", "Tokens", "Errors", "Other"]


def _classify(key: str) -> str:
    """Return the group name for a stat key based on substring matching.

    Matches the key (case-insensitive) against ``_GROUP_PATTERNS``. The first
    matching pattern determines the group. Unmatched keys are placed in
    ``"Other"``.

    Args:
        key (str): The stat display label to classify (e.g. ``"p50_ttft"``).

    Returns:
        str: The group name (e.g. ``"TTFT"``, ``"Throughput"``, ``"Other"``).
    """
    key_lower = key.lower()
    for pattern, group in _GROUP_PATTERNS:
        if pattern in key_lower:
            return group
    return "Other"


def _group_stats(stats: dict[str, str]) -> OrderedDict[str, list[tuple[str, str]]]:
    """Organize stats into ordered groups for display.

    Each stat key is classified via :func:`_classify` and placed into the
    corresponding group. Groups are returned in the canonical order defined
    by ``_GROUP_ORDER``, with empty groups omitted.

    Args:
        stats (dict[str, str]): Mapping of stat labels to formatted values.

    Returns:
        OrderedDict[str, list[tuple[str, str]]]: Groups in display order, where
        each value is a list of ``(label, formatted_value)`` tuples.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for k, v in stats.items():
        group = _classify(k)
        groups.setdefault(group, []).append((k, v))
    # Return in canonical order, skipping empty groups
    return OrderedDict((g, groups[g]) for g in _GROUP_ORDER if g in groups)


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
    TTLT, Tokens, Errors) based on their key names.

    In terminals, prints a compact grouped multi-line block using ANSI escape
    codes to overwrite previous output.

    Args:
        disabled (bool): If ``True``, all display calls are no-ops.

    Example::

        display = LiveStatsDisplay()
        display.update({"rpm": "185.9", "p50_ttft": "0.312s", "fail": "0"})
        display.update({"rpm": "190.2", "p50_ttft": "0.305s", "fail": "1"})
        display.close()
    """

    def __init__(self, disabled: bool = False):
        self._disabled = disabled
        self._is_notebook = _in_notebook()
        self._handle = None
        self._last_line_count = 0

    def update(self, stats: dict[str, str], extra_prefix: str = "") -> None:
        """Refresh the display with new stats.

        Args:
            stats (dict[str, str]): Mapping of label to formatted value.
            extra_prefix (str): Optional prefix text shown before the table
                (e.g. ``"reqs=127"`` for time-bound runs).
        """
        if self._disabled or not stats:
            return

        if self._is_notebook:
            self._update_notebook(stats, extra_prefix)
        else:
            self._update_terminal(stats, extra_prefix)

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

        groups = _group_stats(stats)

        # Build one column per group: header on top, key=value rows below
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

        groups = _group_stats(stats)
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
