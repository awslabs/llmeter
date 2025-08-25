# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from math import ceil
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import os

from llmeter.experiments import LoadTest, LatencyHeatmap, LoadTestResult
from llmeter.runner import Runner
from llmeter.results import Result


@pytest.fixture
def mock_endpoint():
    return MagicMock()


@pytest.fixture
def mock_runner():
    return AsyncMock(spec=Runner)


def test_load_test_post_init(mock_endpoint):
    load_test = LoadTest(
        endpoint=mock_endpoint,
        payload={"input": "test"},
        sequence_of_clients=[1, 2, 3],
    )
    # LoadTest doesn't create Runner in __post_init__, only in run()
    assert load_test.endpoint == mock_endpoint
    assert load_test.payload == {"input": "test"}
    assert load_test.sequence_of_clients == [1, 2, 3]


# mock the run method of the Runner class
@patch("llmeter.experiments.Runner.run", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_load_test_run(mock_endpoint, mock_runner):
    with patch("llmeter.experiments.Runner", return_value=mock_runner):
        load_test = LoadTest(
            endpoint=mock_endpoint,
            payload={"input": "test"},
            sequence_of_clients=[1, 2, 3],
        )

        results = await load_test.run()

        # LoadTest.run() returns a LoadTestResult, check it has results dict
        assert hasattr(results, 'results')
        # The results dict has one entry per unique client count, not per run
        assert len(results.results) >= 1
        mock_runner.run.assert_called()
        assert mock_runner.run.call_count == 3


def test_load_test_with_different_params(mock_endpoint):
    load_test = LoadTest(
        endpoint=mock_endpoint,
        payload=[{"input": "test1"}, {"input": "test2"}],
        sequence_of_clients=[1, 5, 10],
        min_requests_per_client=20,
        min_requests_per_run=100,
        output_path="test_output.json",
        tokenizer=MagicMock(),
    )
    assert load_test.min_requests_per_client == 20
    assert load_test.min_requests_per_run == 100
    assert load_test.output_path == "test_output.json"
    assert load_test.tokenizer is not None


def test_latency_heatmap_post_init(mock_endpoint):
    with patch("llmeter.experiments.CreatePromptCollection") as mock_create_prompt:
        with patch("llmeter.experiments.Runner"):
            mock_create_prompt.return_value.create_collection.return_value = [
                ("input1", 128),
                ("input2", 256),
            ]
            heatmap = LatencyHeatmap(
                endpoint=mock_endpoint,
                source_file="test_file.txt",
                clients=2,
                requests_per_combination=1,
                output_path="/tmp/test_output",
            )
            assert len(heatmap.payload) == 2
            mock_create_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_latency_heatmap_run(mock_endpoint, mock_runner):
    with patch("llmeter.experiments.Runner", return_value=mock_runner):
        with patch("llmeter.experiments.CreatePromptCollection") as mock_create_prompt:
            mock_create_prompt.return_value.create_collection.return_value = [
                ("input1", 128),
                ("input2", 256),
            ]

            heatmap = LatencyHeatmap(
                endpoint=mock_endpoint,
                source_file="test_file.txt",
                clients=2,
                requests_per_combination=1,
                input_lengths=[10, 50],
                output_lengths=[128, 256],
                output_path="/tmp/test_output",
            )

            results = await heatmap.run()

            mock_runner.run.assert_called_once()
            assert results == mock_runner.run.return_value


def test_latency_heatmap_with_different_params(mock_endpoint, tmp_path):
    (tmp_path / "test_file.txt").write_text("test")

    def create_payload_fn(input_text, max_tokens, **kwargs):
        return {"input": input_text, "max_tokens": max_tokens, **kwargs}

    with patch("llmeter.experiments.CreatePromptCollection") as mock_create_prompt:
        mock_create_prompt.return_value.create_collection.return_value = [
            ("input1", 128),
            ("input2", 256),
        ]
        heatmap = LatencyHeatmap(
            endpoint=mock_endpoint,
            source_file=tmp_path / "test_file.txt",
            clients=4,
            requests_per_combination=2,
            input_lengths=[100, 200, 300],
            output_lengths=[512, 1024],
            create_payload_fn=create_payload_fn,
            create_payload_kwargs={"temperature": 0.7},
            tokenizer=MagicMock(),
            output_path="/tmp/test_output",
        )
        assert heatmap.clients == 4
        assert heatmap.requests_per_combination == 2
        assert heatmap.input_lengths == [100, 200, 300]
        assert heatmap.output_lengths == [512, 1024]
        assert heatmap.create_payload_kwargs == {"temperature": 0.7}
        assert heatmap.tokenizer is not None


# Test LoadTestResult class
class TestLoadTestResult:
    """Test the LoadTestResult class."""

    def test_load_test_result_init(self):
        """Test LoadTestResult initialization."""
        mock_result1 = MagicMock(spec=Result)
        mock_result1.clients = 1
        mock_result2 = MagicMock(spec=Result)
        mock_result2.clients = 2
        
        results = {1: mock_result1, 2: mock_result2}  # type: ignore
        load_test_result = LoadTestResult(
            results=results,
            test_name="test_experiment",
            output_path="/tmp/test"
        )
        
        assert load_test_result.results == results
        assert load_test_result.test_name == "test_experiment"
        assert load_test_result.output_path == "/tmp/test"

    @patch('llmeter.experiments.plot_load_test_results')
    def test_plot_results_html(self, mock_plot):
        """Test plot_results with HTML format."""
        mock_fig1 = MagicMock()
        mock_fig2 = MagicMock()
        mock_plot.return_value = {"fig1": mock_fig1, "fig2": mock_fig2}
        
        load_test_result = LoadTestResult(
            results={},
            test_name="test",
            output_path="/tmp/test"
        )
        
        figs = load_test_result.plot_results(show=False, format="html")
        
        assert figs == {"fig1": mock_fig1, "fig2": mock_fig2}
        mock_plot.assert_called_once_with(load_test_result)
        mock_fig1.write_html.assert_called_once()
        mock_fig2.write_html.assert_called_once()

    @patch('llmeter.experiments.plot_load_test_results')
    def test_plot_results_png(self, mock_plot):
        """Test plot_results with PNG format."""
        mock_fig = MagicMock()
        mock_plot.return_value = {"test_fig": mock_fig}
        
        load_test_result = LoadTestResult(
            results={},
            test_name="test",
            output_path="/tmp/test"
        )
        
        figs = load_test_result.plot_results(show=False, format="png")
        
        mock_fig.write_html.assert_called_once_with(Path("/tmp/test") / "test_fig.png")

    @patch('llmeter.experiments.plot_load_test_results')
    def test_plot_results_show(self, mock_plot):
        """Test plot_results with show=True."""
        mock_fig = MagicMock()
        mock_plot.return_value = {"test_fig": mock_fig}
        
        load_test_result = LoadTestResult(
            results={},
            test_name="test",
            output_path=None
        )
        
        load_test_result.plot_results(show=True)
        
        mock_fig.show.assert_called_once()

    def test_load_test_result_load_none_path(self):
        """Test LoadTestResult.load with None path."""
        with pytest.raises(FileNotFoundError, match="Load path cannot be None or empty"):
            LoadTestResult.load(None)

    def test_load_test_result_load_empty_path(self):
        """Test LoadTestResult.load with empty path."""
        with pytest.raises(FileNotFoundError, match="Load path cannot be None or empty"):
            LoadTestResult.load("")

    def test_load_test_result_load_nonexistent_path(self):
        """Test LoadTestResult.load with nonexistent path."""
        with pytest.raises(FileNotFoundError, match="Load path .* does not exist"):
            LoadTestResult.load("/nonexistent/path")

    def test_load_test_result_load_no_results(self, tmp_path):
        """Test LoadTestResult.load with directory containing no results."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No results found in"):
            LoadTestResult.load(empty_dir)

    @patch('llmeter.experiments.Result.load')
    def test_load_test_result_load_success(self, mock_result_load, tmp_path):
        """Test successful LoadTestResult.load."""
        # Create mock results
        mock_result1 = MagicMock(spec=Result)
        mock_result1.clients = 1
        mock_result2 = MagicMock(spec=Result)
        mock_result2.clients = 2
        mock_result_load.side_effect = [mock_result1, mock_result2]
        
        # Create test directories
        result_dir1 = tmp_path / "result1"
        result_dir1.mkdir()
        result_dir2 = tmp_path / "result2"
        result_dir2.mkdir()
        
        # Create a file in tmp_path to ensure iterdir works correctly
        (tmp_path / "file.txt").write_text("test")
        
        result = LoadTestResult.load(tmp_path, "custom_test_name")
        
        assert result.test_name == "custom_test_name"
        assert result.output_path == tmp_path.parent
        assert len(result.results) == 2
        assert 1 in result.results
        assert 2 in result.results
        assert mock_result_load.call_count == 2

    @patch('llmeter.experiments.Result.load')
    def test_load_test_result_load_with_string_path(self, mock_result_load, tmp_path):
        """Test LoadTestResult.load with string path."""
        mock_result = MagicMock(spec=Result)
        mock_result.clients = 1
        mock_result_load.return_value = mock_result
        
        result_dir = tmp_path / "result"
        result_dir.mkdir()
        
        result = LoadTestResult.load(str(tmp_path))
        
        assert result.test_name == tmp_path.name
        assert result.output_path == tmp_path.parent


# Test LatencyHeatmap additional functionality
class TestLatencyHeatmapExtended:
    """Extended tests for LatencyHeatmap class."""

    @patch('llmeter.experiments.CreatePromptCollection')
    @patch('llmeter.experiments.Runner')
    def test_latency_heatmap_run_with_output_path(self, mock_runner_class, mock_create_prompt, tmp_path):
        """Test LatencyHeatmap.run with custom output_path."""
        mock_create_prompt.return_value.create_collection.return_value = [("input", 128)]
        mock_runner = AsyncMock()
        mock_runner_class.return_value = mock_runner
        
        heatmap = LatencyHeatmap(
            endpoint=MagicMock(),
            source_file="test_file.txt",
            output_path=str(tmp_path)
        )
        
        # Test run with custom output path
        import asyncio
        async def run_test():
            return await heatmap.run(output_path=str(tmp_path / "custom"))
        
        result = asyncio.run(run_test())
        
        # Check that runner was called with the custom path
        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["output_path"] == Path(tmp_path / "custom")

    @patch('llmeter.experiments.CreatePromptCollection')
    @patch('llmeter.experiments.Runner')
    def test_latency_heatmap_run_none_output_path(self, mock_runner_class, mock_create_prompt):
        """Test LatencyHeatmap.run with None output_path."""
        mock_create_prompt.return_value.create_collection.return_value = [("input", 128)]
        mock_runner = AsyncMock()
        mock_runner_class.return_value = mock_runner
        
        heatmap = LatencyHeatmap(
            endpoint=MagicMock(),
            source_file="test_file.txt",
            output_path=None
        )
        
        import asyncio
        async def run_test():
            return await heatmap.run()
        
        asyncio.run(run_test())
        
        # Check that runner was called with None path
        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["output_path"] is None

    @patch('llmeter.experiments.plot_heatmap')
    def test_plot_heatmaps_success(self, mock_plot_heatmap, mock_endpoint):
        """Test plot_heatmaps method."""
        # Create a heatmap with results
        heatmap = LatencyHeatmap.__new__(LatencyHeatmap)  # Create without __init__
        heatmap._results = MagicMock()  # Mock results
        
        mock_fig1 = MagicMock()
        mock_fig2 = MagicMock()
        mock_plot_heatmap.side_effect = [mock_fig1, mock_fig2]
        
        f1, f2 = heatmap.plot_heatmaps(n_bins_x=10, n_bins_y=15, show=False)
        
        assert f1 == mock_fig1
        assert f2 == mock_fig2
        assert mock_plot_heatmap.call_count == 2
        
        # Check the calls
        calls = mock_plot_heatmap.call_args_list
        assert calls[0][0] == (heatmap._results, "time_to_first_token")
        assert calls[0][1] == {"n_bins_x": 10, "n_bins_y": 15, "show_scatter": True}
        assert calls[1][0] == (heatmap._results, "time_to_last_token")
        assert calls[1][1] == {"n_bins_x": 10, "n_bins_y": 15, "show_scatter": True}

    def test_plot_heatmaps_no_results(self, mock_endpoint):
        """Test plot_heatmaps with no results."""
        heatmap = LatencyHeatmap.__new__(LatencyHeatmap)
        heatmap._results = None  # type: ignore
        
        with pytest.raises(ValueError, match="No results to plot"):
            heatmap.plot_heatmaps(None, None)

    @patch('llmeter.experiments.plot_heatmap')
    def test_plot_heatmaps_with_show(self, mock_plot_heatmap, mock_endpoint):
        """Test plot_heatmaps with show=True."""
        heatmap = LatencyHeatmap.__new__(LatencyHeatmap)
        heatmap._results = MagicMock()
        
        mock_fig1 = MagicMock()
        mock_fig2 = MagicMock()
        mock_plot_heatmap.side_effect = [mock_fig1, mock_fig2]
        
        heatmap.plot_heatmaps(None, None, show=True)
        
        mock_fig1.show.assert_called_once()
        mock_fig2.show.assert_called_once()


# Test LoadTest additional functionality
class TestLoadTestExtended:
    """Extended tests for LoadTest class."""

    @patch('llmeter.experiments.Runner')
    @pytest.mark.asyncio
    async def test_load_test_run_with_output_path_exception(self, mock_runner_class, mock_endpoint):
        """Test LoadTest.run when output_path creation fails."""
        mock_runner = AsyncMock()
        mock_runner_class.return_value = mock_runner
        
        load_test = LoadTest(
            endpoint=mock_endpoint,
            payload={"input": "test"},
            sequence_of_clients=[1],
            output_path=None  # This will cause Path(None) to fail
        )
        
        # Mock Path to raise an exception
        with patch('llmeter.experiments.Path') as mock_path:
            mock_path.side_effect = Exception("Path error")
            
            await load_test.run()
            
            # Should still call runner with output_path=None
            mock_runner.run.assert_called_once()
            call_kwargs = mock_runner.run.call_args[1]
            assert call_kwargs["output_path"] is None

    def test_load_test_post_init_with_test_name(self, mock_endpoint):
        """Test LoadTest.__post_init__ with custom test_name."""
        load_test = LoadTest(
            endpoint=mock_endpoint,
            payload={"input": "test"},
            sequence_of_clients=[1],
            test_name="custom_test"
        )
        
        assert load_test._test_name == "custom_test"

    def test_load_test_post_init_without_test_name(self, mock_endpoint):
        """Test LoadTest.__post_init__ without test_name (uses timestamp)."""
        with patch('llmeter.experiments.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.__format__ = MagicMock(return_value="20240101-1200")
            mock_datetime.now.return_value = mock_now
            
            load_test = LoadTest(
                endpoint=mock_endpoint,
                payload={"input": "test"},
                sequence_of_clients=[1]
            )
            
            assert load_test._test_name == "20240101-1200"


@pytest.fixture
def experiment_runner():
    # Create an ExperimentRunner instance with some default values
    return LoadTest(
        endpoint=mock_endpoint,  # type: ignore
        payload={"input": "test"},
        sequence_of_clients=[1, 2, 3],
        min_requests_per_client=10,
        min_requests_per_run=50,
    )


def test_get_n_requests_below_min_requests_per_run(experiment_runner):
    # Test when clients * min_requests_per_client < min_requests_per_run
    clients = 4  # 4 * 10 = 40, which is less than min_requests_per_run (50)
    result = experiment_runner._get_n_requests(clients)
    expected = ceil(experiment_runner.min_requests_per_run / clients)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_n_requests_above_min_requests_per_run(experiment_runner):
    # Test when clients * min_requests_per_client >= min_requests_per_run
    clients = 6  # 6 * 10 = 60, which is more than min_requests_per_run (50)
    result = experiment_runner._get_n_requests(clients)
    expected = experiment_runner.min_requests_per_client
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_n_requests_exact_min_requests_per_run(experiment_runner):
    # Test when clients * min_requests_per_client == min_requests_per_run
    clients = 5  # 5 * 10 = 50, which is equal to min_requests_per_run (50)
    result = experiment_runner._get_n_requests(clients)
    expected = experiment_runner.min_requests_per_client
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_n_requests_single_client(experiment_runner):
    # Test with a single client
    clients = 1
    result = experiment_runner._get_n_requests(clients)
    expected = experiment_runner.min_requests_per_run
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_n_requests_large_number_of_clients(experiment_runner):
    # Test with a large number of clients
    clients = 1000
    result = experiment_runner._get_n_requests(clients)
    expected = experiment_runner.min_requests_per_client
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_n_requests_returns_integer(experiment_runner):
    # Test that the function always returns an integer
    clients = 7  # This will cause a division that's not a whole number
    result = experiment_runner._get_n_requests(clients)
    assert isinstance(result, int), f"Expected an integer, but got {type(result)}"


@pytest.mark.parametrize(
    "min_requests_per_client, min_requests_per_run, clients, expected",
    [
        (10, 50, 4, 13),
        (10, 50, 6, 10),
        (5, 100, 10, 10),
        (20, 50, 2, 25),
    ],
)
def test_get_n_requests_parametrized(
    min_requests_per_client, min_requests_per_run, clients, expected
):
    # Parametrized test to cover multiple scenarios
    runner = LoadTest(
        endpoint=mock_endpoint,  # type: ignore
        payload={"input": "test"},
        sequence_of_clients=[1, 2, 3],
        min_requests_per_client=min_requests_per_client,
        min_requests_per_run=min_requests_per_run,
    )
    result = runner._get_n_requests(clients)
    assert result == expected, f"Expected {expected}, but got {result}"
