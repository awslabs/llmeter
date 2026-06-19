# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plotting utilities for ProfileCallback data.

Provides ready-made Plotly figures for visualizing profiling results:
- Phase breakdown (stacked bar chart showing TTFT vs generation time per request)
- Request timeline (Gantt-style view of request concurrency)
- Throughput over time (tokens/second as the run progresses)
- Time accounting (pie/sunburst showing where wall-clock time goes)
- TPOT distribution (histogram of per-token latency)

All functions accept either a `ProfileCallback` instance (with data still in memory)
or a list of `InvocationProfile` records (e.g. loaded from `profile_invocations.jsonl`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..utils import DeferredError
from .defaults import DEFAULT_TEMPLATE

if not TYPE_CHECKING:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        go = DeferredError(e)
        make_subplots = DeferredError(e)
else:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from ..callbacks.profiling import InvocationProfile, ProfileCallback


def _get_profiles(
    source: ProfileCallback | list[InvocationProfile],
) -> list[InvocationProfile]:
    """Extract InvocationProfile list from either a callback or a raw list."""
    if isinstance(source, list):
        return source
    return source.invocation_profiles


def plot_phase_breakdown(
    source: ProfileCallback | list[InvocationProfile],
    *,
    title: str = "Per-Request Phase Breakdown",
    show_overhead: bool = False,
    **layout_kwargs: Any,
) -> go.Figure:
    """Stacked bar chart showing TTFT vs generation time for each request.

    Each bar represents one request, split into:
    - TTFT (prefill + network latency)
    - Generation (server-side decode)
    - Callback overhead (optional, usually negligible)

    Args:
        source: A ProfileCallback instance or list of InvocationProfile records.
        title: Chart title.
        show_overhead: If True, include callback overhead as a third segment.
        **layout_kwargs: Additional kwargs passed to fig.update_layout().

    Returns:
        A Plotly Figure with the stacked bar chart.
    """
    profiles = _get_profiles(source)
    successful = [p for p in profiles if p.error is None]

    x_labels = list(range(1, len(successful) + 1))
    ttft_values = [p.ttft or 0 for p in successful]
    generation_values = [p.generation_time or 0 for p in successful]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=ttft_values,
            name="TTFT (prefill + network)",
            marker_color="#636EFA",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=generation_values,
            name="Generation (server decode)",
            marker_color="#EF553B",
        )
    )

    if show_overhead:
        overhead_values = [p.callback_overhead for p in successful]
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=overhead_values,
                name="Callback overhead",
                marker_color="#00CC96",
            )
        )

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Request #",
        yaxis_title="Time (seconds)",
        template=DEFAULT_TEMPLATE,
        **layout_kwargs,
    )

    return fig


def plot_request_timeline(
    source: ProfileCallback | list[InvocationProfile],
    *,
    title: str = "Request Timeline",
    color_by: str = "phase",
    **layout_kwargs: Any,
) -> go.Figure:
    """Gantt-style timeline showing when each request was active.

    Visualizes request concurrency — overlapping bars indicate parallel requests.
    Each request is shown as a horizontal bar from its start offset to completion,
    split into TTFT and stream phases.

    Args:
        source: A ProfileCallback instance or list of InvocationProfile records.
        title: Chart title.
        color_by: How to color bars. "phase" splits TTFT/stream with different colors.
            "throughput" colors by output tokens/second.
        **layout_kwargs: Additional kwargs passed to fig.update_layout().

    Returns:
        A Plotly Figure with the timeline.
    """
    profiles = _get_profiles(source)
    successful = [p for p in profiles if p.error is None]

    if not successful:
        fig = go.Figure()
        fig.update_layout(title=title, template=DEFAULT_TEMPLATE)
        return fig

    fig = go.Figure()

    # Sort by offset for visual clarity
    sorted_profiles = sorted(successful, key=lambda p: p.offset_from_run_start)

    y_labels = [f"#{p.sequence}" for p in sorted_profiles]

    if color_by == "phase":
        # TTFT phase
        ttft_starts = []
        ttft_durations = []
        stream_starts = []
        generation_durations = []

        for p in sorted_profiles:
            ttlt = p.ttlt or 0
            ttft = p.ttft or 0
            # Request started at offset - ttlt (it completed at offset)
            req_start = p.offset_from_run_start - ttlt
            ttft_starts.append(req_start)
            ttft_durations.append(ttft)
            stream_starts.append(req_start + ttft)
            generation_durations.append(p.generation_time or 0)

        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=ttft_durations,
                base=ttft_starts,
                orientation="h",
                name="TTFT",
                marker_color="#636EFA",
                opacity=0.8,
            )
        )
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=generation_durations,
                base=stream_starts,
                orientation="h",
                name="Generation",
                marker_color="#EF553B",
                opacity=0.8,
            )
        )
    else:
        # Color by throughput
        tps_values = [p.output_tokens_per_second or 0 for p in sorted_profiles]
        starts = []
        durations = []

        for p in sorted_profiles:
            ttlt = p.ttlt or 0
            req_start = p.offset_from_run_start - ttlt
            starts.append(req_start)
            durations.append(ttlt)

        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=durations,
                base=starts,
                orientation="h",
                name="Request",
                marker=dict(
                    color=tps_values,
                    colorscale="Viridis",
                    colorbar=dict(title="tok/s"),
                ),
                opacity=0.8,
            )
        )

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Time since run start (seconds)",
        yaxis_title="Request",
        template=DEFAULT_TEMPLATE,
        height=max(300, len(successful) * 25 + 100),
        **layout_kwargs,
    )

    return fig


def plot_throughput_over_time(
    source: ProfileCallback | list[InvocationProfile],
    *,
    title: str = "Output Throughput Over Time",
    window: int = 5,
    **layout_kwargs: Any,
) -> go.Figure:
    """Scatter plot of per-request output throughput (tok/s) over the run timeline.

    Shows how throughput varies as the run progresses, with an optional rolling
    average to smooth noise.

    Args:
        source: A ProfileCallback instance or list of InvocationProfile records.
        title: Chart title.
        window: Rolling average window size (number of requests). Set to 1 to disable.
        **layout_kwargs: Additional kwargs passed to fig.update_layout().

    Returns:
        A Plotly Figure with throughput over time.
    """
    profiles = _get_profiles(source)
    successful = [
        p
        for p in profiles
        if p.error is None and p.output_tokens_per_second is not None
    ]

    if not successful:
        fig = go.Figure()
        fig.update_layout(title=title, template=DEFAULT_TEMPLATE)
        return fig

    # Sort by completion time
    sorted_profiles = sorted(successful, key=lambda p: p.offset_from_run_start)
    offsets = [p.offset_from_run_start for p in sorted_profiles]
    tps_values = [p.output_tokens_per_second or 0 for p in sorted_profiles]

    fig = go.Figure()

    # Individual points
    fig.add_trace(
        go.Scatter(
            x=offsets,
            y=tps_values,
            mode="markers",
            name="Per-request",
            marker=dict(size=6, opacity=0.5, color="#636EFA"),
        )
    )

    # Rolling average
    if window > 1 and len(tps_values) >= window:
        rolling_avg = []
        for i in range(len(tps_values)):
            start = max(0, i - window + 1)
            rolling_avg.append(sum(tps_values[start : i + 1]) / (i - start + 1))

        fig.add_trace(
            go.Scatter(
                x=offsets,
                y=rolling_avg,
                mode="lines",
                name=f"Rolling avg (window={window})",
                line=dict(color="#EF553B", width=2),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time since run start (seconds)",
        yaxis_title="Output tokens/second",
        template=DEFAULT_TEMPLATE,
        **layout_kwargs,
    )

    return fig


def plot_time_accounting(
    source: ProfileCallback | list[InvocationProfile],
    *,
    total_wall_clock: float | None = None,
    title: str = "Time Accounting",
    **layout_kwargs: Any,
) -> go.Figure:
    """Pie chart showing how wall-clock time is split between API time and overhead.

    Args:
        source: A ProfileCallback instance or list of InvocationProfile records.
        total_wall_clock: Total run wall-clock time in seconds. If source is a
            ProfileCallback with a computed report, this is extracted automatically.
        title: Chart title.
        **layout_kwargs: Additional kwargs passed to fig.update_layout().

    Returns:
        A Plotly Figure with the pie chart.
    """
    from ..callbacks.profiling import ProfileCallback as _PC

    profiles = _get_profiles(source)
    successful = [p for p in profiles if p.error is None]

    api_time = sum(p.ttlt or 0 for p in successful)

    # Try to get wall clock from report
    if total_wall_clock is None and isinstance(source, _PC) and source.report:
        total_wall_clock = source.report.total_wall_clock

    if total_wall_clock is None or total_wall_clock <= 0:
        # Can't compute overhead without wall clock
        total_wall_clock = api_time  # fallback: show API time only

    overhead = max(0.0, total_wall_clock - api_time)

    # Further break down API time into TTFT vs stream
    total_ttft = sum(p.ttft or 0 for p in successful)
    total_generation = sum(p.generation_time or 0 for p in successful)

    labels = ["TTFT (prefill)", "Generation (decode)", "Runner overhead"]
    values = [total_ttft, total_generation, overhead]
    colors = ["#636EFA", "#EF553B", "#AB63FA"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                textinfo="label+percent",
                textposition="inside",
                hole=0.3,
            )
        ]
    )

    fig.update_layout(
        title=title,
        template=DEFAULT_TEMPLATE,
        **layout_kwargs,
    )

    return fig


def plot_tpot_distribution(
    source: ProfileCallback | list[InvocationProfile],
    *,
    title: str = "Time Per Output Token Distribution",
    bin_size_ms: float | None = None,
    **layout_kwargs: Any,
) -> go.Figure:
    """Histogram of time-per-output-token (TPOT) values in milliseconds.

    Args:
        source: A ProfileCallback instance or list of InvocationProfile records.
        title: Chart title.
        bin_size_ms: Histogram bin size in milliseconds. Auto if None.
        **layout_kwargs: Additional kwargs passed to fig.update_layout().

    Returns:
        A Plotly Figure with the TPOT histogram.
    """
    profiles = _get_profiles(source)
    tpot_ms = [
        p.time_per_output_token * 1000
        for p in profiles
        if p.error is None and p.time_per_output_token is not None
    ]

    if not tpot_ms:
        fig = go.Figure()
        fig.update_layout(title=title, template=DEFAULT_TEMPLATE)
        return fig

    xbins = dict(size=bin_size_ms) if bin_size_ms else None

    fig = go.Figure(
        data=[
            go.Histogram(
                x=tpot_ms,
                name="TPOT",
                marker_color="#636EFA",
                opacity=0.8,
                xbins=xbins,
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time per output token (ms)",
        yaxis_title="Count",
        template=DEFAULT_TEMPLATE,
        **layout_kwargs,
    )

    return fig


def plot_ttft_vs_input_tokens(
    source: ProfileCallback | list[InvocationProfile],
    *,
    title: str = "TTFT vs Input Tokens",
    **layout_kwargs: Any,
) -> go.Figure:
    """Scatter plot of TTFT against input token count.

    Useful for understanding how prefill time scales with prompt length.
    Cache hits are highlighted with a different marker.

    Args:
        source: A ProfileCallback instance or list of InvocationProfile records.
        title: Chart title.
        **layout_kwargs: Additional kwargs passed to fig.update_layout().

    Returns:
        A Plotly Figure with the scatter plot.
    """
    profiles = _get_profiles(source)
    successful = [
        p
        for p in profiles
        if p.error is None and p.ttft is not None and p.tokens_input is not None
    ]

    if not successful:
        fig = go.Figure()
        fig.update_layout(title=title, template=DEFAULT_TEMPLATE)
        return fig

    # Split into cached vs non-cached
    non_cached = [p for p in successful if not p.cache_hit]
    cached = [p for p in successful if p.cache_hit]

    fig = go.Figure()

    if non_cached:
        fig.add_trace(
            go.Scatter(
                x=[p.tokens_input for p in non_cached],
                y=[p.ttft for p in non_cached],
                mode="markers",
                name="No cache",
                marker=dict(size=8, color="#636EFA", opacity=0.7),
            )
        )

    if cached:
        fig.add_trace(
            go.Scatter(
                x=[p.tokens_input for p in cached],
                y=[p.ttft for p in cached],
                mode="markers",
                name="Cache hit",
                marker=dict(size=10, color="#00CC96", opacity=0.8, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Input tokens",
        yaxis_title="TTFT (seconds)",
        template=DEFAULT_TEMPLATE,
        **layout_kwargs,
    )

    return fig


def plot_profile_summary(
    source: ProfileCallback | list[InvocationProfile],
    *,
    total_wall_clock: float | None = None,
    title: str = "Profile Summary",
) -> go.Figure:
    """Multi-panel figure combining the key profiling visualizations.

    Creates a 2x2 subplot with:
    - Phase breakdown (stacked bar)
    - Throughput over time (scatter + rolling avg)
    - TPOT distribution (histogram)
    - Time accounting (pie)

    Args:
        source: A ProfileCallback instance or list of InvocationProfile records.
        total_wall_clock: Total run wall-clock time (auto-detected from ProfileCallback).
        title: Overall figure title.

    Returns:
        A Plotly Figure with the 2x2 subplot layout.
    """
    from ..callbacks.profiling import ProfileCallback as _PC

    profiles = _get_profiles(source)
    successful = [p for p in profiles if p.error is None]

    if isinstance(source, _PC) and source.report:
        if total_wall_clock is None:
            total_wall_clock = source.report.total_wall_clock

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Phase Breakdown",
            "Throughput Over Time",
            "TPOT Distribution",
            "Time Accounting",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "pie"}],
        ],
    )

    # Panel 1: Phase breakdown
    x_labels = list(range(1, len(successful) + 1))
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=[p.ttft or 0 for p in successful],
            name="TTFT",
            marker_color="#636EFA",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=[p.generation_time or 0 for p in successful],
            name="Generation",
            marker_color="#EF553B",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Panel 2: Throughput over time
    sorted_by_time = sorted(successful, key=lambda p: p.offset_from_run_start)
    offsets = [p.offset_from_run_start for p in sorted_by_time]
    tps_values = [p.output_tokens_per_second or 0 for p in sorted_by_time]

    fig.add_trace(
        go.Scatter(
            x=offsets,
            y=tps_values,
            mode="markers+lines",
            name="tok/s",
            marker=dict(size=5, color="#00CC96"),
            line=dict(color="#00CC96", width=1),
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # Panel 3: TPOT histogram
    tpot_ms = [
        p.time_per_output_token * 1000
        for p in successful
        if p.time_per_output_token is not None
    ]
    if tpot_ms:
        fig.add_trace(
            go.Histogram(
                x=tpot_ms,
                name="TPOT (ms)",
                marker_color="#AB63FA",
                opacity=0.8,
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # Panel 4: Time accounting pie
    api_time = sum(p.ttlt or 0 for p in successful)
    total_ttft = sum(p.ttft or 0 for p in successful)
    total_generation = sum(p.generation_time or 0 for p in successful)
    overhead = max(0.0, (total_wall_clock or api_time) - api_time)

    fig.add_trace(
        go.Pie(
            labels=["TTFT", "Generation", "Overhead"],
            values=[total_ttft, total_generation, overhead],
            marker=dict(colors=["#636EFA", "#EF553B", "#AB63FA"]),
            textinfo="label+percent",
            hole=0.3,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        barmode="stack",
        title_text=title,
        template=DEFAULT_TEMPLATE,
        height=700,
        width=1100,
    )

    # Axis labels for subplots
    fig.update_xaxes(title_text="Request #", row=1, col=1)
    fig.update_yaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Run time (s)", row=1, col=2)
    fig.update_yaxes(title_text="tok/s", row=1, col=2)
    fig.update_xaxes(title_text="TPOT (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig
