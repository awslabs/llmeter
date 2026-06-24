# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for building charts and visualizations of LLMeter results

These tools depend on [Plotly](https://plotly.com/python/), which you can either install separately
or via the `llmeter[plotting]` extra.
"""

from .plotting import (
    boxplot_by_dimension,
    color_sequences,
    histogram_by_dimension,
    plot_heatmap,
    plot_load_test_results,
    scatter_histogram_2d,
)
from .profiling import (
    plot_phase_breakdown,
    plot_profile_summary,
    plot_request_timeline,
    plot_throughput_over_time,
    plot_time_accounting,
    plot_tpot_distribution,
    plot_ttft_vs_input_tokens,
)

__all__ = [
    "boxplot_by_dimension",
    "color_sequences",
    "histogram_by_dimension",
    "plot_heatmap",
    "plot_load_test_results",
    "plot_phase_breakdown",
    "plot_profile_summary",
    "plot_request_timeline",
    "plot_throughput_over_time",
    "plot_time_accounting",
    "plot_tpot_distribution",
    "plot_ttft_vs_input_tokens",
    "scatter_histogram_2d",
]
