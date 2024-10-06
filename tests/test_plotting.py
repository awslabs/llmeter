from pathlib import Path
from typing import Literal
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from llmeter.plotting import binning, plot_heatmap, plot_sweep_results
from llmeter.runner import Result


@pytest.fixture
def sample_result():
    responses = [
        MagicMock(
            to_dict=lambda: {
                "num_tokens_output": np.random.randint(10, 100),
                "num_tokens_input": np.random.randint(10, 100),
                "time_to_last_token": np.random.random(),
                "time_to_first_token": np.random.random(),
            }
        )
        for _ in range(50)
    ]
    return Result(responses=responses, total_requests=50, clients=5, n_requests=10)  # type: ignore


def test_binning():
    vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Test with default bins
    result = binning(vector)
    assert len(result) == len(vector)
    assert isinstance(result, list)

    # Test with specified bins
    result = binning(vector, bins=5)
    assert len(result) == len(vector)
    assert len(set(result)) <= 5


def test_binning_with_repeated_values():
    vector = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    result = binning(vector)
    assert len(result) == len(vector)
    assert len(set(result)) <= len(set(vector))


@patch("llmeter.plotting.sns")
def test_plot_heatmap(mock_sns, sample_result: Result):
    fig, ax = plot_heatmap(sample_result)

    assert mock_sns.FacetGrid.called


@patch("llmeter.plotting.pd.DataFrame")
@patch("matplotlib.pyplot.subplots")
def test_plot_sweep_results(mock_subplots, mock_dataframe):
    mock_results = [
        MagicMock(
            stats={
                "requests_per_minute": i,
                "time_to_first_token_p50": np.random.random(),
                "time_to_last_token_p50": np.random.random(),
                "failed_requests_rate": np.random.random(),
                "num_tokens_output": np.random.randint(10, 100),
                "num_tokens_input": np.random.randint(10, 100),
                "time_to_last_token": np.random.random(),
                "time_to_first_token": np.random.random(),
            }
        )
        for i in range(5)
    ]

    mock_dataframe.return_value = pd.DataFrame([r.stats for r in mock_results])
    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_subplots.return_value = (mock_fig, [mock_ax])

    fig, ax = plot_sweep_results(mock_results)

    assert mock_dataframe.called


def test_binning_edge_cases():
    # Test with empty vector
    assert binning([]) == []

    # Test with single value
    result = binning([1])
    assert len(result) == 1
    assert result[0] == "1"

    # Test with large range of values
    large_vector = list(range(1000))
    result = binning(large_vector)
    assert len(result) == len(large_vector)
    assert len(set(result)) < len(large_vector)


@pytest.mark.parametrize("bins", [None, 5, 10, 20])
def test_binning_with_different_bin_sizes(
    bins: None | Literal[5] | Literal[10] | Literal[20],
):
    vector = list(range(100))
    result = binning(vector, bins=bins)
    assert len(result) == len(vector)
    if bins is not None:
        assert len(set(result)) <= bins


@patch("llmeter.plotting.sns")
def test_plot_heatmap_with_custom_bins(mock_sns, sample_result: Result):
    fig, ax = plot_heatmap(sample_result, bins_output_tokens=5, bins_input_tokens=5)

    assert mock_sns.FacetGrid.called


@patch("llmeter.plotting.pd.DataFrame")
@patch("matplotlib.pyplot.subplots")
def test_plot_sweep_results_with_output_path(
    mock_subplots, mock_dataframe, tmp_path: Path
):
    mock_results = [
        MagicMock(
            stats={
                "requests_per_minute": i,
                "time_to_first_token_p50": np.random.random(),
                "time_to_last_token_p50": np.random.random(),
                "failed_requests_rate": np.random.random(),
            }
        )
        for i in range(5)
    ]

    mock_dataframe.return_value = pd.DataFrame([r.stats for r in mock_results])
    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_subplots.return_value = (mock_fig, [mock_ax])

    output_path = tmp_path / "test_output"
    output_path.mkdir()

    fig, ax = plot_sweep_results(mock_results, output_path=output_path)

    assert mock_dataframe.called
    # assert mock_dataframe.plot.called


@pytest.fixture
def mock_figure():
    fig = MagicMock()
    ax = MagicMock()
    ax.get_figure.return_value = fig
    return fig, [ax]


@pytest.fixture
def mock_results():
    return [
        MagicMock(
            stats={
                "requests_per_minute": i,
                "time_to_first_token_p50": 0.1,
                "time_to_last_token_p50": 0.2,
                "failed_requests_rate": 0.01,
            }
        )
        for i in range(5)
    ]
