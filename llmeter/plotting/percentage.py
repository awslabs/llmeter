# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard representation for percentage values as a point on a line.

A percentage that represents a fraction of a total is visualized as a dot
positioned on a horizontal line segment from 0% to 100%, with a filled segment
from the origin to the value. This gives immediate spatial context for where a
metric sits relative to its full range.
"""

from typing import TYPE_CHECKING

from ..utils import DeferredError
from .defaults import DEFAULT_TEMPLATE, get_colorway

if not TYPE_CHECKING:
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        go = DeferredError(e)
else:
    import plotly.graph_objects as go


def _build_hover(
    label: str, value: float, actual: float | None, total: float | None
) -> str:
    """Build a hovertemplate string including actual/total when available."""
    parts = [f"<b>{label}</b>: {value:.1%}"]
    if actual is not None and total is not None:
        parts.append(f"{actual:g} / {total:g}")
    elif actual is not None:
        parts.append(f"value: {actual:g}")
    elif total is not None:
        parts.append(f"total: {total:g}")
    return "<br>".join(parts) + "<extra></extra>"


def percentage_point(
    label: str,
    value: float,
    *,
    actual: float | None = None,
    total: float | None = None,
    color: str = "#636EFA",
    line_color: str = "#E0E0E0",
    show_label: bool = True,
    row: int = 0,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Render a single percentage value as a point on a [0, 1] line.

    The visualization consists of:
    - A gray background line spanning the full [0%, 100%] range
    - A colored fill segment from 0% to the value
    - A dot marker at the value position with a text label

    Args:
        label: Name of the metric.
        value: Fraction in [0, 1] (e.g. 0.73 for 73%).
        actual: The numerator value (e.g. 730 cached requests). Shown in tooltip.
        total: The denominator value (e.g. 1000 total requests). Shown in tooltip.
        color: Color for the point marker and fill segment.
        line_color: Color of the background line.
        show_label: Whether to annotate the label on the left.
        row: Vertical position index (for stacking multiple).
        fig: Existing figure to add to; creates a new one if None.

    Returns:
        The Plotly Figure with the percentage point added.

    Example:
        >>> fig = percentage_point("Cache Hit Rate", 0.73, actual=730, total=1000)
        >>> fig.show()
    """
    if fig is None:
        fig = go.Figure()
        fig.update_layout(
            template=DEFAULT_TEMPLATE,
            xaxis=dict(
                range=[-0.05, 1.05],
                tickformat=".0%",
                dtick=0.25,
                showgrid=False,
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
            height=120,
            margin=dict(l=120, r=40, t=30, b=30),
        )

    y = -row  # stack downwards

    # The baseline: a thin line from 0 to 1
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[y, y],
            mode="lines",
            line=dict(color=line_color, width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Filled segment from 0 to the value
    fig.add_trace(
        go.Scatter(
            x=[0, value],
            y=[y, y],
            mode="lines",
            line=dict(color=color, width=5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # The point representing the actual value
    fig.add_trace(
        go.Scatter(
            x=[value],
            y=[y],
            mode="markers+text",
            marker=dict(color=color, size=14, symbol="circle"),
            text=[f"{value:.0%}"],
            textposition="top center",
            textfont=dict(size=12),
            showlegend=False,
            name=label,
            hovertemplate=_build_hover(label, value, actual, total),
        )
    )

    # Label annotation on the left
    if show_label:
        fig.add_annotation(
            x=-0.03,
            y=y,
            text=label,
            showarrow=False,
            xanchor="right",
            font=dict(size=12),
        )

    return fig


def percentage_points(
    metrics: dict[str, float | tuple[float, float | None, float | None]],
    *,
    colors: list[str] | None = None,
    line_color: str = "#E0E0E0",
    title: str | None = None,
) -> go.Figure:
    """Render multiple percentage values, each as a point on its own line.

    Each metric gets its own horizontal row with a background line, a colored
    fill segment, and a dot marker.

    Args:
        metrics: Mapping of label -> value. Value can be:
            - a float (fraction 0-1), or
            - a tuple of (fraction, actual, total) where actual/total are
              shown in the tooltip. Use None to omit either.
        colors: Optional list of colors (cycles if shorter than metrics).
        line_color: Color for the background lines.
        title: Optional figure title.

    Returns:
        A Plotly Figure with one row per metric.

    Example:
        >>> fig = percentage_points({
        ...     "Cache Hit Rate": (0.73, 730, 1000),
        ...     "Error Rate": (0.02, 2, 100),
        ...     "GPU Utilization": 0.91,
        ... })
        >>> fig.show()
    """
    colors = colors or get_colorway()

    fig = go.Figure()
    n = len(metrics)

    for i, (label, raw) in enumerate(metrics.items()):
        # Unpack value, actual, total
        if isinstance(raw, tuple):
            value = raw[0]
            actual = raw[1] if len(raw) > 1 else None
            total = raw[2] if len(raw) > 2 else None
        else:
            value = raw
            actual = None
            total = None

        color = colors[i % len(colors)]
        y = -i

        # Background line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[y, y],
                mode="lines",
                line=dict(color=line_color, width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Filled segment from 0 to value
        fig.add_trace(
            go.Scatter(
                x=[0, value],
                y=[y, y],
                mode="lines",
                line=dict(color=color, width=5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Point
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[y],
                mode="markers+text",
                marker=dict(color=color, size=14, symbol="circle"),
                text=[f"{value:.0%}"],
                textposition="top center",
                textfont=dict(size=12, color=color),
                showlegend=False,
                name=label,
                hovertemplate=_build_hover(label, value, actual, total),
            )
        )

        # Label on the left
        fig.add_annotation(
            x=-0.03,
            y=y,
            text=label,
            showarrow=False,
            xanchor="right",
            font=dict(size=12),
        )

    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        xaxis=dict(
            range=[-0.05, 1.05],
            tickformat=".0%",
            dtick=0.25,
            showgrid=False,
        ),
        yaxis=dict(
            range=[-(n - 1) - 0.5, 0.5],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        height=max(120, 60 * n + 60),
        margin=dict(l=150, r=40, t=50, b=30),
        title=title,
    )

    return fig
