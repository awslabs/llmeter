# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
        # title="LLMeter: Number of Tokens vs. Time to Last Token",
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


def stat_clients(results: list[Result], stat: str, **scatter_kwargs):
    """
    Create a scatter plot trace for a given statistic vs number of clients.

    Args:
        results (list[Result]): List of Result objects containing the response data
        stat (str): The statistic to plot (e.g. 'failed_requests_rate', 'requests_per_minute')
        **scatter_kwargs: Additional keyword arguments to pass to go.Scatter

    Returns:
        plotly.graph_objects.Scatter: Scatter plot trace of the statistic vs number of clients
    """
    results_sorted = sorted(results, key=lambda x: x.clients, reverse=False)
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


def error_clients_fig(results: list[Result], log_scale=False, **scatter_kwargs):
    """
    Create a figure showing error rate vs number of clients.

    Args:
        results (list[Result]): List of Result objects containing the response data
        **scatter_kwargs: Additional keyword arguments to pass to stat_clients

    Returns:
        plotly.graph_objects.Figure: Figure showing error rate vs number of clients
    """
    fig = go.Figure()
    fig.add_trace(stat_clients(results, "failed_requests_rate", **scatter_kwargs))
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


def rpm_clients_fig(results: list[Result], log_scale=False, **scatter_kwargs):
    """
    Create a figure showing requests per minute vs number of clients.

    Args:
        results (list[Result]): List of Result objects containing the response data
        **scatter_kwargs: Additional keyword arguments to pass to stat_clients

    Returns:
        plotly.graph_objects.Figure: Figure showing requests per minute vs number of clients
    """
    fig = go.Figure()
    fig.add_trace(stat_clients(results, "requests_per_minute", **scatter_kwargs))
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


def latency_clients(
    results: list[Result],
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
    if not results:
        raise ValueError("Results list cannot be empty")

    y_box = []
    x_box = []
    for result in sorted(results, key=lambda x: x.clients, reverse=False):
        try:
            y = result.get_dimension(dimension)
            y_box.extend(y)
            x_box.extend([result.clients] * len(y))
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
    results: list[Result],
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
    if not results:
        raise ValueError("Results list cannot be empty")

    try:
        fig = go.Figure()
        box_trace = latency_clients(results, dimension, **box_kwargs)
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
    result: list[Result],
    output_path: Path | None,
    log_scale=True,
    format: Literal["html", "png"] = "html",
):
    f1 = latency_clients_fig(result, "time_to_first_token", log_scale=log_scale)
    f2 = latency_clients_fig(
        result,
        "time_to_last_token",
        log_scale=log_scale,
        marker_color=px.colors.qualitative.Plotly[1],
    )
    f3 = rpm_clients_fig(
        result, marker_color=px.colors.qualitative.Plotly[2], log_scale=log_scale
    )
    f4 = error_clients_fig(
        result, marker_color=px.colors.qualitative.Plotly[3], log_scale=log_scale
    )

    if output_path:
        # save figure to the output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if format == "html":
            f1.write_html(output_path / "time_to_first_token.html")
            f2.write_html(output_path / "time_to_last_token.html")
            f3.write_html(output_path / "requests_per_minute.html")
            f4.write_html(output_path / "error_rate.html")
        elif format == "png":
            f1.write_image(output_path / "time_to_first_token.png")
            f2.write_image(output_path / "time_to_last_token.png")
            f3.write_image(output_path / "requests_per_minute.png")
            f4.write_image(output_path / "error_rate.png")

    return {
        "time_to_first_token": f1,
        "time_to_last_token": f2,
        "requests_per_minute": f3,
        "error_rate": f4,
    }
