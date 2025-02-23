# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from upath import UPath as Path

from ..runner import Result
from ..utils import DeferredError

if not TYPE_CHECKING:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        px = DeferredError("Please install plotly to use plotting functions")
        go = DeferredError("Please install plotly to use plotting functions")
else:
    import plotly.express as px
    import plotly.graph_objects as go

try:
    import kaleido
except ModuleNotFoundError:
    kaleido = DeferredError("Please install kaleido to use plotting functions")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmeter.experiments import SweepResult


color_sequences = px.colors.sequential


def scatter_histogram_2d(
    result: Result, x_dimension: str, y_dimension: str, n_bins_x: int, n_bins_y: int
) -> go.Figure:
    """
    Generate a scatter plot with histograms for the given dimensions.

    Args:
        result (Result): The Result object containing the data.
        x_dimension (str): The dimension to be plotted on the x-axis.
        y_dimension (str): The dimension to be plotted on the y-axis.
        n_bins_x (int): The number of bins for the x-axis histogram.
        n_bins_y (int): The number of bins for the y-axis histogram.

    Returns:
        plotly.graph_objs._figure.Figure: The generated scatter plot figure.
    """

    try:
        x = result.get_dimension(x_dimension)
        y = result.get_dimension(y_dimension)
    except AttributeError:
        raise ValueError(f"Invalid dimension: {x_dimension} or {y_dimension}")

    fig = px.scatter(
        x=x,
        y=y,
        marginal_x="histogram",
        marginal_y="histogram",
        template="plotly_white",
        range_x=[0, max(x for x in x if x is not None) * 1.1],
        range_y=[0, max(y for y in y if y is not None) * 1.1],
        labels={
            "x": x_dimension.replace("_", " ").capitalize(),
            "y": y_dimension.replace("_", " ").capitalize(),
        },
        width=800,
        height=800,
    )
    fig.update_traces(nbinsx=n_bins_x, autobinx=True, selector={"type": "histogram"})
    fig.update_traces(nbinsy=n_bins_y, autobinx=True, selector={"type": "histogram"})
    return fig


def histogram_by_dimension(result, dimension: str, **histogram_kwargs) -> go.Histogram:
    x = result.get_dimension(dimension)
    name = histogram_kwargs.pop("name", None) or result.run_name
    histnorm = histogram_kwargs.pop("histnorm", None) or "probability"

    return go.Histogram(x=x, name=name, histnorm=histnorm, **histogram_kwargs)


def plot_heatmap(
    result: Result,
    dimension: str,
    n_bins_x: int | None,
    n_bins_y: int | None,
    output_path: Path | None = None,
    show_scatter=False,
) -> go.Figure:
    """
    Generate a heatmap visualization of token counts and a specified dimension.

    Args:
        result (Result): The Result object containing the response data
        dimension (str): The dimension to plot as the z-axis/color values
        n_bins_x (int): Number of bins for the x-axis histogram
        n_bins_y (int): Number of bins for the y-axis histogram
        show_scatter (bool, optional): Whether to overlay scatter points. Defaults to False.

    Returns:
        plotly.graph_objs._figure.Figure: The generated heatmap figure

    Raises:
        ValueError: If the specified dimension is not found in the result data
    """

    num_tokens_input_list = result.get_dimension("num_tokens_input")
    num_tokens_output_list = result.get_dimension("num_tokens_output")
    try:
        z = result.get_dimension(dimension)
    except AttributeError:
        raise ValueError(f"Dimension {dimension} not found in result")

    fig = px.density_heatmap(
        x=num_tokens_input_list,
        y=num_tokens_output_list,
        z=z,
        marginal_x="histogram",
        marginal_y="histogram",
        template="plotly_white",
        histfunc="avg",
        nbinsx=n_bins_x,
        nbinsy=n_bins_y,
        labels={
            "x": "Number of input tokens",
            "y": "Number of output tokens",
        },
        title=f'LLMeter: average {dimension.replace("_", " ").capitalize()}',
        width=800,
        height=800,
    )
    fig.layout["coloraxis"]["colorbar"]["title"] = "average"  # type: ignore
    fig.update_coloraxes(colorbar_tickformat=".1s")
    fig.update_coloraxes(colorbar_ticksuffix="s")
    fig.update_layout(coloraxis_colorscale="oryel")
    fig.update_traces(marker_color="grey", selector=dict(type="histogram"))
    fig.update_traces(texttemplate="%{z:.3s}s <br> %{x:.2s}, %{y:.2s}")
    fig.update_traces(
        texttemplate="%{y:.3s}", selector=dict(type="histogram", bingroup="x")
    )
    fig.update_traces(
        texttemplate="%{x:.3s}", selector=dict(type="histogram", bingroup="y")
    )
    if show_scatter:
        fig.add_trace(
            go.Scatter(
                x=num_tokens_input_list,
                y=num_tokens_output_list,
                mode="markers",
                showlegend=False,
                marker=dict(
                    symbol="x",
                    opacity=0.7,
                    color="white",
                    size=8,
                    line=dict(width=1),
                ),
            )
        )
    return fig


def boxplot_by_dimension(
    result: Result, dimension: str, name: str | None = None, **box_kwargs
) -> go.Box:
    x = result.get_dimension(dimension)
    name = box_kwargs.pop("name", None) or result.run_name

    return go.Box(x=x, name=name, **box_kwargs)


def stat_clients(sweep_result: SweepResult, stat: str, **scatter_kwargs):
    """
    Create a scatter plot trace for a given statistic vs number of clients.

    Args:
        sweep_result (SweepResult): SweepResult object containing the response data
        stat (str): The statistic to plot (e.g. 'failed_requests_rate', 'requests_per_minute')
        **scatter_kwargs: Additional keyword arguments to pass to go.Scatter

    Returns:
        plotly.graph_objects.Scatter: Scatter plot trace of the statistic vs number of clients
    """
    results_sorted = [
        sweep_result.results[k] for k in sorted(sweep_result.results.keys())
    ]
    ER = [k.stats[stat] for k in results_sorted]
    clients = [k.clients for k in results_sorted]

    default_kwargs = {
        "x": clients,
        "y": ER,
        "mode": "lines+markers",
        "name": stat.replace("_", " ").capitalize(),
        "opacity": 0.5,
        "line": dict(dash="dot"),
    }

    # scatter_kwargs take precedence over defaults
    default_kwargs.update(scatter_kwargs)

    return go.Scatter(**default_kwargs)


def error_clients_fig(sweep_result: SweepResult, log_scale=False, **scatter_kwargs):
    """
    Create a figure showing error rate vs number of clients.

    Args:
        sweep_result (SweepResult): SweepResult object containing the response data
        log_scale (bool, optional): Whether to use logarithmic scale for x-axis. Defaults to False.
        **scatter_kwargs: Additional keyword arguments to pass to stat_clients

    Returns:
        plotly.graph_objects.Figure: Figure showing error rate vs number of clients
    """
    fig = go.Figure()
    fig.add_trace(stat_clients(sweep_result, "failed_requests_rate", **scatter_kwargs))
    fig.update_layout(
        title="Error rate vs number of clients",
        xaxis_title="Number of clients",
        yaxis_title="Error rate",
    )
    fig.update_layout(template="plotly_white")
    if log_scale:
        fig.update_xaxes(type="log")
        # fig.update_yaxes(type="log")

    return fig


def rpm_clients_fig(sweep_result: SweepResult, log_scale=False, **scatter_kwargs):
    """
    Create a figure showing requests per minute vs number of clients.

    Args:
        sweep_result (SweepResult): SweepResult object containing the response data
        log_scale (bool, optional): Whether to use logarithmic scale for axes. Defaults to False.
        **scatter_kwargs: Additional keyword arguments to pass to stat_clients

    Returns:
        plotly.graph_objects.Figure: Figure showing requests per minute vs number of clients
    """
    fig = go.Figure()
    fig.add_trace(stat_clients(sweep_result, "requests_per_minute", **scatter_kwargs))
    fig.update_layout(
        title="Requests per minute vs number of clients",
        xaxis_title="Number of clients",
        # xaxis_tickformat=".2s",
        yaxis_title="Requests per minute",
        yaxis_tickformat=".2s",
    )
    fig.update_layout(template="plotly_white")

    if log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    return fig


def average_input_tokens_clients_fig(
    sweep_result: SweepResult, log_scale=False, **scatter_kwargs
):
    fig = go.Figure()
    fig.add_trace(
        stat_clients(sweep_result, "average_input_tokens_per_minute", **scatter_kwargs)
    )
    fig.update_layout(
        title="Average input tokens per minute vs number of clients",
        xaxis_title="Number of clients",
        xaxis_tickformat=".2s",
        yaxis_title="Average input tokens per minute",
        yaxis_tickformat=".2s",
    )
    fig.update_layout(template="plotly_white")
    if log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    return fig


def average_output_tokens_clients_fig(
    sweep_result: SweepResult, log_scale=False, **scatter_kwargs
):
    fig = go.Figure()
    fig.add_trace(
        stat_clients(sweep_result, "average_output_tokens_per_minute", **scatter_kwargs)
    )
    fig.update_layout(
        title="Average output tokens per minute vs number of clients",
        xaxis_title="Number of clients",
        xaxis_tickformat=".2s",
        yaxis_title="Average output tokens per minute",
        yaxis_tickformat=".2s",
    )
    fig.update_layout(template="plotly_white")
    if log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    return fig


def latency_clients(
    sweep_result: SweepResult,
    dimension: Literal["time_to_first_token", "time_to_last_token"],
    **box_kwargs,
):
    """
    Create a box plot trace showing latency distribution vs number of clients.

    Args:
        results (list[Result]): List of Result objects containing response data
        dimension (Literal["time_to_first_token", "time_to_last_token"]): The latency dimension to plot
        **box_kwargs: Additional keyword arguments to pass to go.Box

    Returns:
        plotly.graph_objects.Box: Box plot trace showing latency distribution

    Raises:
        ValueError: If results list is empty or dimension is invalid
    """

    y_box = []
    x_box = []
    r = sweep_result.results
    # for result in sorted(results, key=lambda x: x.clients, reverse=False):
    for n_clients in sorted(r.keys()):
        result = r[n_clients]
        try:
            y = result.get_dimension(dimension)
            y_box.extend(y)
            x_box.extend([n_clients] * len(y))
        except AttributeError:
            raise ValueError(f"Invalid dimension: {dimension}")

    if not y_box or not x_box:
        raise ValueError("No valid data points found in results")

    default_kwargs = dict(
        x=x_box,
        y=y_box,
        name=dimension.replace("_", " ").capitalize(),
    )
    default_kwargs.update(box_kwargs)

    return go.Box(**default_kwargs)


def latency_clients_fig(
    sweep_result: SweepResult,
    dimension: Literal["time_to_first_token", "time_to_last_token"],
    log_scale=False,
    **box_kwargs,
):
    """
    Create a figure showing latency distribution vs number of clients.

    Args:
        results (list[Result]): List of Result objects containing response data
        dimension (Literal["time_to_first_token", "time_to_last_token"]): The latency dimension to plot
        log_scale (bool, optional): Whether to use logarithmic scale for axes. Defaults to False.
        **box_kwargs: Additional keyword arguments to pass to latency_clients

    Returns:
        plotly.graph_objects.Figure: Figure showing latency distribution vs number of clients

    Raises:
        ValueError: If results list is empty or dimension is invalid
    """

    try:
        fig = go.Figure()
        if not box_kwargs.get("name"):
            box_kwargs["name"] = sweep_result.test_name
        box_trace = latency_clients(sweep_result, dimension, **box_kwargs)
        fig.add_trace(box_trace)
        fig.update_layout(
            title=f"{dimension.replace('_', ' ').capitalize()} vs number of clients",
            xaxis_title="Number of clients",
            xaxis_tickformat="s",
            yaxis_title=f'{dimension.replace("_", " ").capitalize()} (s)',
            yaxis_tickformat=".2s",
        )
        if log_scale:
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")

        fig.update_layout(template="plotly_white")
        return fig
    except Exception as e:
        raise ValueError(f"Error creating figure: {str(e)}")


def plot_sweep_results(
    sweep_result: SweepResult,
    log_scale=True,
):
    f1 = latency_clients_fig(sweep_result, "time_to_first_token", log_scale=log_scale)
    f2 = latency_clients_fig(sweep_result, "time_to_last_token", log_scale=log_scale)
    f3 = rpm_clients_fig(sweep_result, log_scale=log_scale)
    f4 = error_clients_fig(sweep_result, log_scale=log_scale)
    f5 = average_input_tokens_clients_fig(sweep_result, log_scale=log_scale)
    f6 = average_output_tokens_clients_fig(sweep_result, log_scale=log_scale)

    return {
        "time_to_first_token": f1,
        "time_to_last_token": f2,
        "requests_per_minute": f3,
        "error_rate": f4,
        "average_input_tokens_clients": f5,
        "average_output_tokens_clients": f6,
    }
