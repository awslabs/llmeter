import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from llmeter.plotting import plot_heatmap
from llmeter.plotting.plotting import (
    scatter_histogram_2d,
    histogram_by_dimension,
    boxplot_by_dimension,
    stat_clients,
    error_clients_fig,
    rpm_clients_fig,
    average_input_tokens_clients_fig,
    average_output_tokens_clients_fig,
    latency_clients,
    latency_clients_fig,
    plot_load_test_results,
)
from llmeter.plotting.heatmap import (
    Interval,
    Heatmap,
    _cut,
    initialize_map,
    _bin_responses_by_tokens,
    _counts_and_errors,
    _calculate_maps,
    _sort_map_labels,
    _map_nested_dicts,
    _get_heatmap_stats,
)
from llmeter.runner import Result
from llmeter.experiments import LoadTestResult
from llmeter.endpoints.base import InvocationResponse


@pytest.fixture
def sample_result():
    responses = [
        MagicMock(
            num_tokens_output=np.random.randint(10, 100),
            num_tokens_input=np.random.randint(10, 100),
            time_to_last_token=np.random.random(),
            time_to_first_token=np.random.random(),
            error=None,
        )
        for _ in range(50)
    ]
    return Result(responses=responses, total_requests=50, clients=5, n_requests=10)  # type: ignore


@pytest.fixture
def sample_load_test_result():
    """Create a sample LoadTestResult for testing."""
    results = {}
    for clients in [1, 2, 4, 8]:
        responses = [
            MagicMock(
                spec=InvocationResponse,
                num_tokens_output=np.random.randint(10, 100),
                num_tokens_input=np.random.randint(10, 100),
                time_to_last_token=np.random.random(),
                time_to_first_token=np.random.random(),
                error=None,
            )
            for _ in range(20)
        ]

        # Create a mock Result object with stats property
        stats_data = {
            "failed_requests_rate": np.random.random() * 0.1,
            "requests_per_minute": np.random.randint(100, 1000),
            "average_input_tokens_per_minute": np.random.randint(1000, 5000),
            "average_output_tokens_per_minute": np.random.randint(1000, 5000),
        }

        # Mock get_dimension method
        def mock_get_dimension(dimension):
            if dimension == "time_to_first_token":
                return [r.time_to_first_token for r in responses]
            elif dimension == "time_to_last_token":
                return [r.time_to_last_token for r in responses]
            else:
                return [getattr(r, dimension) for r in responses]

        # Create a mock result with all necessary attributes
        result = MagicMock()
        result.responses = responses
        result.total_requests = 20
        result.clients = clients
        result.n_requests = 20
        result.get_dimension = mock_get_dimension

        # Use PropertyMock for the stats property
        type(result).stats = PropertyMock(return_value=stats_data)

        results[clients] = result

    return LoadTestResult(
        results=results, test_name="test_load_test", output_path="/tmp/test"
    )


@pytest.fixture
def sample_responses():
    """Create sample InvocationResponse objects for testing."""
    return [
        MagicMock(
            spec=InvocationResponse,
            num_tokens_input=np.random.randint(10, 100),
            num_tokens_output=np.random.randint(10, 100),
            time_to_first_token=np.random.random(),
            time_to_last_token=np.random.random(),
            error=None if np.random.random() > 0.1 else "test error",
        )
        for _ in range(30)
    ]


# Test plotting.py functions


@patch("llmeter.plotting.plotting.px")
def test_plot_heatmap(mock_px, sample_result: Result):
    # Mock the density_heatmap function
    mock_fig = MagicMock()
    mock_px.density_heatmap.return_value = mock_fig

    # Call the function with required parameters
    fig = plot_heatmap(sample_result, "time_to_first_token", 10, 10)

    # Verify that px.density_heatmap was called
    assert mock_px.density_heatmap.called
    assert fig == mock_fig


@patch("llmeter.plotting.plotting.px")
def test_plot_heatmap_invalid_dimension(mock_px, sample_result: Result):
    """Test plot_heatmap with invalid dimension."""
    mock_px.density_heatmap.return_value = MagicMock()

    # Mock get_dimension to return valid data for required dimensions but fail for invalid ones
    def mock_get_dimension(dimension):
        if dimension in ["num_tokens_input", "num_tokens_output"]:
            return [1, 2, 3, 4, 5]  # Return valid data for required dimensions
        else:
            raise AttributeError("Invalid dimension")

    sample_result.get_dimension = mock_get_dimension

    with pytest.raises(
        ValueError, match="Dimension invalid_dimension not found in result"
    ):
        plot_heatmap(sample_result, "invalid_dimension", 10, 10)


@patch("llmeter.plotting.plotting.px")
def test_scatter_histogram_2d(mock_px, sample_result: Result):
    """Test scatter_histogram_2d function."""
    mock_fig = MagicMock()
    mock_px.scatter.return_value = mock_fig

    fig = scatter_histogram_2d(
        sample_result, "num_tokens_input", "num_tokens_output", 10, 10
    )

    assert mock_px.scatter.called
    assert fig == mock_fig


@patch("llmeter.plotting.plotting.px")
def test_scatter_histogram_2d_invalid_dimension(mock_px, sample_result: Result):
    """Test scatter_histogram_2d with invalid dimension."""
    sample_result.get_dimension = MagicMock(
        side_effect=AttributeError("Invalid dimension")
    )

    with pytest.raises(ValueError, match="Invalid dimension"):
        scatter_histogram_2d(sample_result, "invalid_x", "invalid_y", 10, 10)


@patch("llmeter.plotting.plotting.go")
def test_histogram_by_dimension(mock_go, sample_result: Result):
    """Test histogram_by_dimension function."""
    mock_histogram = MagicMock()
    mock_go.Histogram.return_value = mock_histogram

    # Mock get_dimension to return sample data
    sample_result.get_dimension = MagicMock(return_value=[1, 2, 3, 4, 5])
    sample_result.run_name = "test_run"

    hist = histogram_by_dimension(sample_result, "time_to_first_token")

    assert mock_go.Histogram.called
    assert hist == mock_histogram


@patch("llmeter.plotting.plotting.go")
def test_boxplot_by_dimension(mock_go, sample_result: Result):
    """Test boxplot_by_dimension function."""
    mock_box = MagicMock()
    mock_go.Box.return_value = mock_box

    # Mock get_dimension to return sample data
    sample_result.get_dimension = MagicMock(return_value=[1, 2, 3, 4, 5])
    sample_result.run_name = "test_run"

    box = boxplot_by_dimension(sample_result, "time_to_first_token")

    assert mock_go.Box.called
    assert box == mock_box


@patch("llmeter.plotting.plotting.go")
def test_stat_clients(mock_go, sample_load_test_result: LoadTestResult):
    """Test stat_clients function."""
    mock_scatter = MagicMock()
    mock_go.Scatter.return_value = mock_scatter

    scatter = stat_clients(sample_load_test_result, "failed_requests_rate")

    assert mock_go.Scatter.called
    assert scatter == mock_scatter


@patch("llmeter.plotting.plotting.go")
def test_error_clients_fig(mock_go, sample_load_test_result: LoadTestResult):
    """Test error_clients_fig function."""
    mock_fig = MagicMock()
    mock_scatter = MagicMock()
    mock_go.Figure.return_value = mock_fig
    mock_go.Scatter.return_value = mock_scatter

    fig = error_clients_fig(sample_load_test_result)

    assert mock_go.Figure.called
    assert fig == mock_fig


@patch("llmeter.plotting.plotting.go")
def test_rpm_clients_fig(mock_go, sample_load_test_result: LoadTestResult):
    """Test rpm_clients_fig function."""
    mock_fig = MagicMock()
    mock_scatter = MagicMock()
    mock_go.Figure.return_value = mock_fig
    mock_go.Scatter.return_value = mock_scatter

    fig = rpm_clients_fig(sample_load_test_result, log_scale=True)

    assert mock_go.Figure.called
    assert fig == mock_fig


@patch("llmeter.plotting.plotting.go")
def test_average_input_tokens_clients_fig(
    mock_go, sample_load_test_result: LoadTestResult
):
    """Test average_input_tokens_clients_fig function."""
    mock_fig = MagicMock()
    mock_scatter = MagicMock()
    mock_go.Figure.return_value = mock_fig
    mock_go.Scatter.return_value = mock_scatter

    fig = average_input_tokens_clients_fig(sample_load_test_result)

    assert mock_go.Figure.called
    assert fig == mock_fig


@patch("llmeter.plotting.plotting.go")
def test_average_output_tokens_clients_fig(
    mock_go, sample_load_test_result: LoadTestResult
):
    """Test average_output_tokens_clients_fig function."""
    mock_fig = MagicMock()
    mock_scatter = MagicMock()
    mock_go.Figure.return_value = mock_fig
    mock_go.Scatter.return_value = mock_scatter

    fig = average_output_tokens_clients_fig(sample_load_test_result)

    assert mock_go.Figure.called
    assert fig == mock_fig


@patch("llmeter.plotting.plotting.go")
def test_latency_clients(mock_go, sample_load_test_result: LoadTestResult):
    """Test latency_clients function."""
    mock_box = MagicMock()
    mock_go.Box.return_value = mock_box

    box = latency_clients(sample_load_test_result, "time_to_first_token")

    assert mock_go.Box.called
    assert box == mock_box


@patch("llmeter.plotting.plotting.go")
def test_latency_clients_invalid_dimension(
    mock_go, sample_load_test_result: LoadTestResult
):
    """Test latency_clients with invalid dimension."""
    # Mock get_dimension to raise AttributeError
    for result in sample_load_test_result.results.values():
        result.get_dimension = MagicMock(
            side_effect=AttributeError("Invalid dimension")
        )

    with pytest.raises(ValueError, match="Invalid dimension"):
        latency_clients(
            sample_load_test_result, "time_to_first_token"
        )  # Use valid type but mock error


@patch("llmeter.plotting.plotting.go")
def test_latency_clients_no_data(mock_go):
    """Test latency_clients with no valid data."""
    empty_load_test = LoadTestResult(results={}, test_name="empty", output_path="/tmp")

    with pytest.raises(ValueError, match="No valid data points found"):
        latency_clients(empty_load_test, "time_to_first_token")


@patch("llmeter.plotting.plotting.go")
def test_latency_clients_fig(mock_go, sample_load_test_result: LoadTestResult):
    """Test latency_clients_fig function."""
    mock_fig = MagicMock()
    mock_box = MagicMock()
    mock_go.Figure.return_value = mock_fig
    mock_go.Box.return_value = mock_box

    fig = latency_clients_fig(sample_load_test_result, "time_to_first_token")

    assert mock_go.Figure.called
    assert fig == mock_fig


@patch("llmeter.plotting.plotting.latency_clients_fig")
@patch("llmeter.plotting.plotting.rpm_clients_fig")
@patch("llmeter.plotting.plotting.error_clients_fig")
@patch("llmeter.plotting.plotting.average_input_tokens_clients_fig")
@patch("llmeter.plotting.plotting.average_output_tokens_clients_fig")
def test_plot_load_test_results(
    mock_avg_out,
    mock_avg_in,
    mock_error,
    mock_rpm,
    mock_latency,
    sample_load_test_result: LoadTestResult,
):
    """Test plot_load_test_results function."""
    # Mock all the individual plotting functions
    mock_latency.return_value = MagicMock()
    mock_rpm.return_value = MagicMock()
    mock_error.return_value = MagicMock()
    mock_avg_in.return_value = MagicMock()
    mock_avg_out.return_value = MagicMock()

    result = plot_load_test_results(sample_load_test_result)

    assert isinstance(result, dict)
    assert len(result) == 6
    assert "time_to_first_token" in result
    assert "time_to_last_token" in result
    assert "requests_per_minute" in result
    assert "error_rate" in result
    assert "average_input_tokens_clients" in result
    assert "average_output_tokens_clients" in result


# Test heatmap.py functions


def test_interval_creation():
    """Test Interval class creation and properties."""
    interval = Interval(1, 5, "both")
    assert interval.left == 1
    assert interval.right == 5
    assert interval.closed == "both"
    assert interval.mid == 3.0


def test_interval_contains():
    """Test Interval __contains__ method."""
    interval = Interval(1, 5, "both")
    assert 3 in interval
    assert 1 in interval
    assert 5 in interval
    assert 0 not in interval
    assert 6 not in interval


def test_interval_str():
    """Test Interval string representation."""
    interval_both = Interval(1, 5, "both")
    assert str(interval_both) == "[1, 5]"

    interval_right = Interval(1, 5, "right")
    assert str(interval_right) == "(1, 5]"

    interval_left = Interval(1, 5, "left")
    assert str(interval_left) == "[1, 5)"

    interval_neither = Interval(1, 5, "neither")
    assert str(interval_neither) == "(1, 5)"


def test_heatmap_initialization_with_responses(sample_responses):
    """Test Heatmap initialization with responses."""
    heatmap = Heatmap(
        responses=sample_responses, n_bins_input_tokens=5, n_bins_output_tokens=5
    )
    assert heatmap.responses == sample_responses
    assert heatmap.bins_input_tokens == 5
    assert heatmap.bins_output_tokens == 5


def test_heatmap_initialization_with_result(sample_result):
    """Test Heatmap initialization with result."""
    heatmap = Heatmap(
        result=sample_result, n_bins_input_tokens=5, n_bins_output_tokens=5
    )
    assert heatmap.responses == sample_result.responses
    assert heatmap.bins_input_tokens == 5
    assert heatmap.bins_output_tokens == 5


def test_heatmap_initialization_no_data():
    """Test Heatmap initialization with no data raises ValueError."""
    with pytest.raises(ValueError, match="Either responses or result must be provided"):
        Heatmap()


@patch("llmeter.plotting.heatmap._get_heatmap_stats")
@patch("llmeter.plotting.heatmap._calculate_maps")
def test_heatmap_get_map(mock_calc_maps, mock_get_stats, sample_responses):
    """Test Heatmap get_map method."""
    mock_calc_maps.return_value = {"test_metric": {"mean": 1.5}}
    mock_get_stats.return_value = {"bin1": {"bin2": 1.5}}

    heatmap = Heatmap(responses=sample_responses)
    result = heatmap.get_map("test_metric", "mean")

    assert mock_calc_maps.called
    assert mock_get_stats.called
    assert result == {"bin1": {"bin2": 1.5}}


def test_cut_function():
    """Test _cut function."""
    arr = [1, 2, 3, 4, 5, 6]
    bins = 3

    intervals, boundaries = _cut(arr, bins)

    assert len(intervals) == len(arr)
    assert len(boundaries) == bins
    assert all(isinstance(interval, Interval) for interval in intervals)
    assert all(isinstance(boundary, Interval) for boundary in boundaries)


def test_cut_function_edge_case():
    """Test _cut function with all same values (edge case)."""
    arr = [5, 5, 5, 5]  # All values are the same
    bins = 3

    intervals, boundaries = _cut(arr, bins)

    assert len(intervals) == len(arr)
    assert len(boundaries) == 1  # Should create only one bin when all values are same
    assert all(isinstance(interval, Interval) for interval in intervals)
    assert all(isinstance(boundary, Interval) for boundary in boundaries)

    # All intervals should be the same since all values are identical
    assert all(interval.left == 5 for interval in intervals)
    assert all(interval.right == 6 for interval in intervals)  # 5 + 1 offset


def test_initialize_map():
    """Test initialize_map function."""
    n_input, n_output = 3, 4
    result = initialize_map(n_input, n_output)

    assert len(result) == n_output
    assert all(len(row) == n_input for row in result)
    assert all(all(cell is None for cell in row) for row in result)


def test_bin_responses_by_tokens(sample_responses):
    """Test _bin_responses_by_tokens function."""
    binned, input_bins, output_bins = _bin_responses_by_tokens(sample_responses, 3, 3)

    assert isinstance(binned, dict)
    assert len(input_bins) == 3
    assert len(output_bins) == 3
    assert all(isinstance(interval, Interval) for interval in input_bins)
    assert all(isinstance(interval, Interval) for interval in output_bins)


def test_counts_and_errors():
    """Test _counts_and_errors function."""
    # Test with empty results
    result = _counts_and_errors([])
    assert result == {"counts": 0, "errors": 0, "error_rate": 0}

    # Test with mixed results
    responses = [
        MagicMock(error=None),
        MagicMock(error="error1"),
        MagicMock(error=None),
        MagicMock(error="error2"),
    ]
    result = _counts_and_errors(responses)
    assert result["counts"] == 4
    assert result["errors"] == 2
    assert result["error_rate"] == 0.5


@patch("llmeter.plotting.heatmap._get_stats_from_results")
def test_calculate_maps(mock_get_stats):
    """Test _calculate_maps function."""
    mock_get_stats.return_value = {"mean": 1.5, "std": 0.5}

    binned_data = {
        Interval(0, 10): {
            Interval(0, 5): [MagicMock(error=None), MagicMock(error="test")]
        }
    }
    metrics = ["test_metric"]

    result = _calculate_maps(binned_data, metrics)

    assert isinstance(result, dict)
    assert mock_get_stats.called


def test_sort_map_labels():
    """Test _sort_map_labels function."""
    heatmaps = {2: {"b": 1, "a": 2}, 1: {"d": 3, "c": 4}}

    result = _sort_map_labels(heatmaps)

    # Outer keys should be sorted in descending order
    assert list(result.keys()) == [2, 1]
    # Inner keys should be sorted in ascending order
    assert list(result[2].keys()) == ["a", "b"]
    assert list(result[1].keys()) == ["c", "d"]


def test_map_nested_dicts():
    """Test _map_nested_dicts function."""

    def double(x):
        return x * 2

    # Test with nested dict
    data = {"a": 1, "b": {"c": 2, "d": 3}}
    result = _map_nested_dicts(data, double)
    assert result == {"a": 2, "b": {"c": 4, "d": 6}}

    # Test with non-dict
    result = _map_nested_dicts(5, double)
    assert result == 10


@patch("llmeter.plotting.heatmap.jmespath")
def test_get_heatmap_stats(mock_jmespath):
    """Test _get_heatmap_stats function."""
    mock_jmespath.search.return_value = 1.5

    heatmaps = {"bin1": {"bin2": {"stat1": 10, "stat2": 20}}}

    result = _get_heatmap_stats(heatmaps, "stat1")

    assert mock_jmespath.search.called
    assert result == {"bin1": {"bin2": 1.5}}
