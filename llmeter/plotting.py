# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from functools import partial
from numbers import Real
from statistics import StatisticsError, quantiles
from typing import Sequence

from upath import UPath as Path

from .runner import Result
from .utils import DeferredError

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = DeferredError("Please install seaborn to use plotting functions")
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = DeferredError("Please install pandas to use plotting functions")
try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = DeferredError("Please install matplotlib to use plotting functions")


def binning(vector, bins: int | None = None) -> tuple[list[Real], bool]:
    """Map the elements of `vector` to a discrete set of `bins` representative values for plotting

    If the cardinality of `vector` already exactly matches the number of `bins`, the same values
    will be returned. If the number of `bins` is not specified, a heuristic is used to select one.

    Returns:
        result: Elements of `vector` mapped to the mid-points of the calculated bins
        binned: Truthy if the data was binned, falsy if it was left as-is

    TODO: Possibly extend `binned` to return the actual bin intervals instead of just True?
    """
    if len(vector) == 0:
        return [], False
    cardinality = len(set(vector))

    if bins is None:
        # https://stats.stackexchange.com/a/114497
        if cardinality < len(vector) / 20:
            m = cardinality
        else:
            try:
                q1, _, q3, _ = quantiles(vector, n=5)
                iqr = q3 - q1
                h = 2 * iqr / (cardinality ** (1 / 3))
                m = int((max(vector) - min(vector)) // h) + 1
            except StatisticsError:
                m = cardinality // 4 + 1
        return [x.mid for x in pd.cut(vector, bins=m)], True

    if cardinality == bins:
        return [x for x in vector], False

    return [x.mid for x in pd.cut(vector, bins=bins)], True


def plot_heatmap(
    result: Result,
    bins_output_tokens: int | None = None,
    bins_input_tokens: int | None = None,
    output_path: os.PathLike | str | None = None,
):
    df = pd.DataFrame([p.to_dict() for p in result.responses])
    token_out_bins, is_out_binned = binning(
        df["num_tokens_output"], bins=bins_output_tokens
    )
    token_in_bins, is_in_binned = binning(
        df["num_tokens_input"], bins=bins_input_tokens
    )
    df["num_tokens_input"] = token_in_bins
    df["num_tokens_output"] = token_out_bins

    df_plot = df.pivot_table(
        values=["time_to_last_token", "time_to_first_token"],
        index="num_tokens_output",
        columns=["num_tokens_input"],
        aggfunc=[_p50, _p90, _p99],
        observed=False,
    ).rename(
        columns={
            "time_to_last_token": "Time to last token",
            "time_to_first_token": "Time to first token",
        }
    )
    df_plot.columns.names = ["Statistics", "Metric", "num_tokens_input"]
    df_plot = df_plot.melt(ignore_index=False).reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg = sns.FacetGrid(
            df_plot,
            col="Statistics",
            row="Metric",
            gridspec_kws={"hspace": 0.4},
            margin_titles=True,
        )
        fg.map_dataframe(
            _draw_heatmap,
            "num_tokens_input",
            "num_tokens_output",
            "value",
            annot=True,
            cmap="crest",
            cbar=False,
            fmt=".03g",
        )
        fg.set_titles("{row_name}-{col_name}")
        fg.set_xlabels(f"Input tokens ({'Binned' if is_in_binned else 'Exact'})")
        fg.set_ylabels(f"Output tokens ({'Binned' if is_out_binned else 'Exact'})")
        for ax in fg.axes.ravel():
            ax.invert_yaxis()
            break
    if output_path:
        with (Path(output_path) / "heatmap.png").open("wb") as f:
            fg.savefig(f, bbox_inches="tight")
    return fg.figure, fg.axes


_p50 = partial(pd.Series.quantile, q=0.5)
_p50.__name__ = "p50"  # type: ignore
_p90 = partial(pd.Series.quantile, q=0.9)
_p90.__name__ = "p90"  # type: ignore
_p99 = partial(pd.Series.quantile, q=0.99)
_p99.__name__ = "p99"  # type: ignore


def _draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    d.reindex(sorted(d.columns), axis=1)
    d.index = d.index.map("{:.0f}".format)
    d.columns = d.columns.map("{:.0f}".format)
    sns.heatmap(d, **kwargs)


def plot_sweep_results(
    results: Sequence[Result], output_path: os.PathLike | str | None = None
):
    sweep_results_df = pd.DataFrame([k.stats for k in results])
    plot_groups = [
        sweep_results_df.filter(like="first").columns,
        sweep_results_df.filter(like="last").columns,
        ("failed_requests_rate",),
    ]

    ax = sweep_results_df.plot(
        x="requests_per_minute",
        y=[k for j in plot_groups for k in j],
        subplots=plot_groups,
        style="o-",
        ylim=[0, None],  # type: ignore
        figsize=(10, 6),
        title="Sweep results",
    )
    if output_path:
        with (Path(output_path) / "sweep_results.png").open("wb") as f:
            ax[0].get_figure().savefig(f)
    return ax[0].get_figure(), ax
