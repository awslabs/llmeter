# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest

from llmeter.endpoints.base import InvocationResponse
from llmeter.experiments import LoadTestResult
from llmeter.results import Result


@pytest.fixture
def mock_results():
    """Create mock results for testing."""
    mock_result = Mock(spec=Result)
    mock_result.clients = 10
    return {10: mock_result}


@pytest.fixture
def realistic_load_test_result(tmp_path):
    """Create LoadTestResult with realistic mock data for integration testing."""
    results = {}
    for clients in [1, 4]:
        responses = [
            MagicMock(
                spec=InvocationResponse,
                num_tokens_output=np.random.randint(10, 100),
                num_tokens_input=np.random.randint(10, 100),
                time_to_last_token=np.random.random(),
                time_to_first_token=np.random.random(),
                error=None,
            )
            for _ in range(10)
        ]

        stats_data = {
            "failed_requests_rate": np.random.random() * 0.1,
            "requests_per_minute": np.random.randint(100, 1000),
            "average_input_tokens_per_minute": np.random.randint(1000, 5000),
            "average_output_tokens_per_minute": np.random.randint(1000, 5000),
        }

        def mock_get_dimension(dimension):
            if dimension == "time_to_first_token":
                return [r.time_to_first_token for r in responses]
            elif dimension == "time_to_last_token":
                return [r.time_to_last_token for r in responses]
            else:
                return [getattr(r, dimension) for r in responses]

        result = MagicMock()
        result.responses = responses
        result.total_requests = 10
        result.clients = clients
        result.n_requests = 10
        result.get_dimension = mock_get_dimension
        type(result).stats = PropertyMock(return_value=stats_data)

        results[clients] = result

    return LoadTestResult(
        results=results, test_name="test_output", output_path=tmp_path / "output"
    )


@pytest.fixture
def load_test_result(mock_results, tmp_path):
    """Create LoadTestResult with simple mock data for basic testing."""
    return LoadTestResult(
        results=mock_results, test_name="test_output", output_path=tmp_path / "output"
    )


class TestLoadTestResultFileOutput:
    """Test file output functionality of LoadTestResult.plot_results()."""

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_creates_html_files(self, mock_plot, load_test_result):
        """Test that HTML files are created when format='html'."""
        mock_fig = Mock()
        mock_plot.return_value = {
            "error_rate": mock_fig,
            "requests_per_minute": mock_fig,
        }

        load_test_result.plot_results(show=False, format="html")

        assert mock_fig.write_image.call_count == 2
        calls = mock_fig.write_image.call_args_list
        assert str(calls[0][0][0]).endswith("error_rate.html")
        assert str(calls[1][0][0]).endswith("requests_per_minute.html")

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_creates_png_files(self, mock_plot, load_test_result):
        """Test that PNG files are created when format='png'."""
        mock_fig = Mock()
        mock_plot.return_value = {
            "time_to_first_token": mock_fig,
            "time_to_last_token": mock_fig,
        }

        load_test_result.plot_results(show=False, format="png")

        assert mock_fig.write_image.call_count == 2
        calls = mock_fig.write_image.call_args_list
        assert str(calls[0][0][0]).endswith("time_to_first_token.png")
        assert str(calls[1][0][0]).endswith("time_to_last_token.png")

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_creates_all_expected_files(self, mock_plot, load_test_result):
        """Test that all expected plot files are created."""
        mock_fig = Mock()
        # Return all expected plot types from plot_load_test_results
        expected_plots = {
            "time_to_first_token": mock_fig,
            "time_to_last_token": mock_fig,
            "requests_per_minute": mock_fig,
            "error_rate": mock_fig,
            "average_input_tokens_clients": mock_fig,
            "average_output_tokens_clients": mock_fig,
        }
        mock_plot.return_value = expected_plots

        load_test_result.plot_results(show=False, format="png")

        # Should create 6 files (one for each plot type)
        assert mock_fig.write_image.call_count == 6

        # Verify all expected file names are created
        calls = mock_fig.write_image.call_args_list
        created_files = [str(call[0][0]) for call in calls]

        for plot_name in expected_plots.keys():
            assert any(f"{plot_name}.png" in filename for filename in created_files)

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_creates_output_directory(
        self, mock_plot, mock_results, tmp_path
    ):
        """Test that output directory is created if it doesn't exist."""
        mock_fig = Mock()
        mock_plot.return_value = {"test_chart": mock_fig}

        output_path = tmp_path / "new_dir" / "nested"
        load_test_result = LoadTestResult(
            results=mock_results, test_name="test", output_path=output_path
        )

        load_test_result.plot_results(show=False, format="png")

        mock_fig.write_image.assert_called_once()
        call_path = mock_fig.write_image.call_args[0][0]
        assert call_path.parent == output_path

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_no_output_when_path_none(self, mock_plot, mock_results):
        """Test that no files are written when output_path is None."""
        mock_fig = Mock()
        mock_plot.return_value = {"test_chart": mock_fig}

        load_test_result = LoadTestResult(
            results=mock_results, test_name="test", output_path=None
        )

        load_test_result.plot_results(show=False, format="png")

        mock_fig.write_image.assert_not_called()

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_applies_color_sequences(self, mock_plot, load_test_result):
        """Test that color sequences are applied to figures."""
        mock_figs = {f"fig_{i}": MagicMock() for i in range(6)}
        mock_plot.return_value = mock_figs

        load_test_result.plot_results(show=False, format="png")

        # Verify that update_layout was called on each figure (for color sequences)
        for fig in mock_figs.values():
            fig.update_layout.assert_called()
            # Check that colorway was set in the update_layout call
            call_args = fig.update_layout.call_args
            assert "colorway" in call_args[1]

    def test_plot_results_integration_with_realistic_data(
        self, realistic_load_test_result
    ):
        """Integration test with realistic data (no mocking of plot_load_test_results)."""
        # This test uses the actual plot_load_test_results function
        # but mocks the underlying plotting functions to avoid plotly dependencies
        with patch(
            "llmeter.plotting.plotting.latency_clients_fig"
        ) as mock_latency, patch(
            "llmeter.plotting.plotting.rpm_clients_fig"
        ) as mock_rpm, patch(
            "llmeter.plotting.plotting.error_clients_fig"
        ) as mock_error, patch(
            "llmeter.plotting.plotting.average_input_tokens_clients_fig"
        ) as mock_avg_in, patch(
            "llmeter.plotting.plotting.average_output_tokens_clients_fig"
        ) as mock_avg_out:
            # Mock all plotting functions to return mock figures
            mock_fig = MagicMock()
            for mock_func in [
                mock_latency,
                mock_rpm,
                mock_error,
                mock_avg_in,
                mock_avg_out,
            ]:
                mock_func.return_value = mock_fig

            # Call plot_results - this should work end-to-end
            result = realistic_load_test_result.plot_results(show=False, format="png")

            # Verify the integration worked
            assert isinstance(result, dict)
            assert len(result) == 6

            # Verify files would be written (6 plots)
            assert mock_fig.write_image.call_count == 6

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_handles_format_parameter_correctly(
        self, mock_plot, load_test_result
    ):
        """Test that format parameter is correctly passed to write_image."""
        mock_fig = Mock()
        mock_plot.return_value = {"test_plot": mock_fig}

        # Test HTML format
        load_test_result.plot_results(show=False, format="html")
        call_path = mock_fig.write_image.call_args[0][0]
        assert str(call_path).endswith(".html")

        mock_fig.reset_mock()

        # Test PNG format
        load_test_result.plot_results(show=False, format="png")
        call_path = mock_fig.write_image.call_args[0][0]
        assert str(call_path).endswith(".png")

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_validates_format_consistency_all_files(
        self, mock_plot, load_test_result
    ):
        """Test that ALL generated files match the specified format."""
        mock_fig = Mock()
        expected_plots = {
            "time_to_first_token": mock_fig,
            "time_to_last_token": mock_fig,
            "requests_per_minute": mock_fig,
            "error_rate": mock_fig,
            "average_input_tokens_clients": mock_fig,
            "average_output_tokens_clients": mock_fig,
        }
        mock_plot.return_value = expected_plots

        # Test HTML format - all files should be .html
        load_test_result.plot_results(show=False, format="html")
        calls = mock_fig.write_image.call_args_list
        assert len(calls) == 6
        for call in calls:
            file_path = str(call[0][0])
            assert file_path.endswith(".html"), f"Expected .html file, got: {file_path}"

        mock_fig.reset_mock()

        # Test PNG format - all files should be .png
        load_test_result.plot_results(show=False, format="png")
        calls = mock_fig.write_image.call_args_list
        assert len(calls) == 6
        for call in calls:
            file_path = str(call[0][0])
            assert file_path.endswith(".png"), f"Expected .png file, got: {file_path}"

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_format_validation_with_specific_plot_names(
        self, mock_plot, load_test_result
    ):
        """Test format validation for each specific plot type."""
        mock_fig = Mock()
        expected_plots = {
            "time_to_first_token": mock_fig,
            "time_to_last_token": mock_fig,
            "requests_per_minute": mock_fig,
            "error_rate": mock_fig,
            "average_input_tokens_clients": mock_fig,
            "average_output_tokens_clients": mock_fig,
        }
        mock_plot.return_value = expected_plots

        # Test with HTML format
        load_test_result.plot_results(show=False, format="html")
        calls = mock_fig.write_image.call_args_list

        # Verify each expected plot file has correct format
        created_files = [str(call[0][0]) for call in calls]
        expected_files = [
            "time_to_first_token.html",
            "time_to_last_token.html",
            "requests_per_minute.html",
            "error_rate.html",
            "average_input_tokens_clients.html",
            "average_output_tokens_clients.html",
        ]

        for expected_file in expected_files:
            assert any(
                expected_file in created_file for created_file in created_files
            ), f"Expected file {expected_file} not found in {created_files}"

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_format_parameter_case_insensitive(
        self, mock_plot, load_test_result
    ):
        """Test that format parameter works with different cases."""
        mock_fig = Mock()
        mock_plot.return_value = {"test_plot": mock_fig}

        # Test uppercase format
        load_test_result.plot_results(show=False, format="PNG")
        call_path = mock_fig.write_image.call_args[0][0]
        assert str(call_path).lower().endswith(".png")

        mock_fig.reset_mock()

        # Test mixed case format
        load_test_result.plot_results(show=False, format="Html")
        call_path = mock_fig.write_image.call_args[0][0]
        assert str(call_path).lower().endswith(".html")

    @patch("llmeter.experiments.plot_load_test_results")
    def test_plot_results_show_parameter(self, mock_plot, load_test_result):
        """Test that show parameter controls figure display."""
        mock_fig = Mock()
        mock_plot.return_value = {"test_plot": mock_fig}

        # Test show=True
        load_test_result.plot_results(show=True, format="png")
        mock_fig.show.assert_called_once()

        mock_fig.reset_mock()

        # Test show=False
        load_test_result.plot_results(show=False, format="png")
        mock_fig.show.assert_not_called()


class TestLoadTestResultIntegration:
    """Integration tests for LoadTestResult with plot_load_test_results."""

    def test_plot_results_return_value_structure(self, realistic_load_test_result):
        """Test that plot_results returns the expected structure from plot_load_test_results."""
        with patch(
            "llmeter.plotting.plotting.latency_clients_fig"
        ) as mock_latency, patch(
            "llmeter.plotting.plotting.rpm_clients_fig"
        ) as mock_rpm, patch(
            "llmeter.plotting.plotting.error_clients_fig"
        ) as mock_error, patch(
            "llmeter.plotting.plotting.average_input_tokens_clients_fig"
        ) as mock_avg_in, patch(
            "llmeter.plotting.plotting.average_output_tokens_clients_fig"
        ) as mock_avg_out:
            # Create distinct mock figures
            mock_figs = [MagicMock() for _ in range(6)]
            mock_latency.side_effect = mock_figs[:2]  # TTFT and TTLT
            mock_rpm.return_value = mock_figs[2]
            mock_error.return_value = mock_figs[3]
            mock_avg_in.return_value = mock_figs[4]
            mock_avg_out.return_value = mock_figs[5]

            result = realistic_load_test_result.plot_results(show=False, format="png")

            # Verify the structure matches plot_load_test_results output
            expected_keys = {
                "time_to_first_token",
                "time_to_last_token",
                "requests_per_minute",
                "error_rate",
                "average_input_tokens_clients",
                "average_output_tokens_clients",
            }
            assert set(result.keys()) == expected_keys

            # Verify each value is a figure object
            for fig in result.values():
                assert hasattr(fig, "write_image")  # Should be a plotly figure
                assert hasattr(fig, "show")
