# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# from upath import UPath as Path

from .runner import Result
from .utils import DeferredError

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    plotly = DeferredError("Please install plotly to use plotting functions")


try:
    import kaleido
except ModuleNotFoundError:
    kaleido = DeferredError("Please install kaleido to use plotting functions")


def scatter_histogram(
    result: Result, x_dimension: str, y_dimension: str, n_bins_x, n_bins_y
):
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


def heatmap(result, dimension, n_bins_x, n_bins_y, show_scatter=False):
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

    num_tokens_input_list = [k.num_tokens_input for k in result.responses]
    num_tokens_output_list = [k.num_tokens_output for k in result.responses]
    try:
        z = [k.to_dict()[dimension] for k in result.responses]
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
