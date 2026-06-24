# Plotting and Visualization

LLMeter includes a set of [Plotly](https://plotly.com/python/)-based visualization primitives for charting benchmark results.

## Installation

Plotting requires the `plotting` extra:

```bash
uv add 'llmeter[plotting]'
```

This installs Plotly, Kaleido (for static image export), and pandas.

## Theme and Color Management

All LLMeter charts share a consistent visual theme and color palette managed through `llmeter.plotting.defaults`:

```python
from llmeter.plotting import DEFAULT_TEMPLATE, get_colorway

print(DEFAULT_TEMPLATE)  # "plotly_white"
print(get_colorway())    # ['#636efa', '#EF553B', '#00cc96', ...]
```

### Switching themes globally

To change the visual theme for all LLMeter charts in a session:

```python
import llmeter.plotting.defaults as plotting_defaults

plotting_defaults.DEFAULT_TEMPLATE = "plotly_dark"
```

After this, every chart function (`percentage_points`, `boxplot_by_dimension`,
`plot_load_test_results`, etc.) will use the dark template — and `get_colorway()`
will return that template's color cycle.

### Using the colorway in custom charts

When building your own Plotly figures alongside LLMeter charts, use `get_colorway()`
to stay visually consistent:

```python
import plotly.graph_objects as go
from llmeter.plotting import get_colorway

colors = get_colorway()
fig = go.Figure()
for i, (name, data) in enumerate(my_series.items()):
    fig.add_trace(go.Scatter(y=data, name=name, line=dict(color=colors[i % len(colors)])))
fig.show()
```

## Percentage Point Visualization

The `percentage_point` and `percentage_points` functions provide a standard
representation for metrics that are fractions of a total (rates, utilization,
hit ratios, etc.).

Each metric is rendered as:

- A gray background line spanning 0% to 100%
- A colored fill segment from 0% to the value
- A dot marker at the value with a text label

This gives immediate spatial context — you see where the value sits relative to
the full range.

### Single metric

```python
from llmeter.plotting import percentage_point

fig = percentage_point("Cache Hit Rate", 0.73, actual=730, total=1000)
fig.show()
```

### Multiple metrics

```python
from llmeter.plotting import percentage_points

fig = percentage_points({
    "Success Rate": (0.97, 485, 500),
    "Cached Responses": (0.34, 170, 500),
    "Responses Under SLA": (0.88, 440, 500),
    "Throttle Rate": (0.06, 30, 500),
})
fig.show()
```

The value can be a plain float or a tuple of `(fraction, actual, total)`. When
`actual` and `total` are provided, the tooltip shows both the percentage and the
raw counts (e.g., "485 / 500").

### Customizing colors

Pass explicit colors per metric:

```python
fig = percentage_points(
    {
        "Success Rate": (0.97, 485, 500),
        "Error Rate": (0.03, 15, 500),
    },
    colors=["#00CC96", "#EF553B"],
)
```

Or leave `colors=None` to cycle through the current template's colorway.

### Use with LLMeter results

A natural fit is visualizing rate-based stats from a benchmark run:

```python
from llmeter.results import Result
from llmeter.plotting import percentage_points

result = Result.load("./outputs/my-run", load_responses=True)

fig = percentage_points({
    "Success Rate": (
        1 - result.stats["failed_requests_rate"],
        result.stats["total_requests"] - result.stats["failed_requests"],
        result.stats["total_requests"],
    ),
    "Failed Requests": (
        result.stats["failed_requests_rate"],
        result.stats["failed_requests"],
        result.stats["total_requests"],
    ),
})
fig.show()
```

## Other Plotting Functions

See the [API Reference](../reference/plotting/index.md) for the full set of
charting utilities:

- `boxplot_by_dimension` — box plots for latency distributions
- `histogram_by_dimension` — histograms of any response dimension
- `scatter_histogram_2d` — scatter with marginal histograms
- `plot_heatmap` — input/output token heatmaps
- `plot_load_test_results` — full load test dashboard (6 charts)
