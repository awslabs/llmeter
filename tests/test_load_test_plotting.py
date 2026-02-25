# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from llmeter.endpoints.base import InvocationResponse
from llmeter.experiments import LoadTestResult
from llmeter.plotting.plotting import plot_load_test_results


@pytest.fixture
def sample_load_test_result():
    """Create a comprehensive LoadTestResult for testing plot_load_test_results."""
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

        # Create realistic stats data
        stats_data = {
            "failed_requests_rate": np.random.random() * 0.1,
            "requests_per_minute": np.random.randint(100, 1000),
            "average_input_tokens_per_minute": np.random.randint(1000, 5000),
            "average_output_tokens_per_minute": np.random.randint(1000, 5000),
        }

        # Mock get_dimension method to return appropriate data
        def mock_get_dimension(dimension):
            if dimension == "time_to_first_token":
                return [r.time_to_first_token for r in responses]
            elif dimension == "time_to_last_token":
                return [r.time_to_last_token for r in responses]
            else:
                return [getattr(r, dimension) for r in responses]

        # Create a comprehensive mock result
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


class TestPlotLoadTestResults:
    """Comprehensive tests for plot_load_test_results function."""

    @patch("llmeter.plotting.plotting.latency_clients_fig")
    @patch("llmeter.plotting.plotting.rpm_clients_fig")
    @patch("llmeter.plotting.plotting.error_clients_fig")
    @patch("llmeter.plotting.plotting.average_input_tokens_clients_fig")
    @patch("llmeter.plotting.plotting.average_output_tokens_clients_fig")
    def test_plot_load_test_results_returns_all_expected_plots(
        self,
        mock_avg_out,
        mock_avg_in,
        mock_error,
        mock_rpm,
        mock_latency,
        sample_load_test_result,
    ):
        """Test that plot_load_test_results returns all expected plot types."""
        # Mock all the individual plotting functions
        mock_latency.return_value = MagicMock()
        mock_rpm.return_value = MagicMock()
        mock_error.return_value = MagicMock()
        mock_avg_in.return_value = MagicMock()
        mock_avg_out.return_value = MagicMock()

        result = plot_load_test_results(sample_load_test_result)

        # Verify return structure
        assert isinstance(result, dict)
        assert len(result) == 6

        # Verify all expected keys are present
        expected_keys = {
            "time_to_first_token",
            "time_to_last_token",
            "requests_per_minute",
            "error_rate",
            "average_input_tokens_clients",
            "average_output_tokens_clients",
        }
        assert set(result.keys()) == expected_keys

        # Verify all plotting functions were called
        assert mock_latency.call_count == 2  # Called for both TTFT and TTLT
        mock_rpm.assert_called_once()
        mock_error.assert_called_once()
        mock_avg_in.assert_called_once()
        mock_avg_out.assert_called_once()

    @patch("llmeter.plotting.plotting.latency_clients_fig")
    @patch("llmeter.plotting.plotting.rpm_clients_fig")
    @patch("llmeter.plotting.plotting.error_clients_fig")
    @patch("llmeter.plotting.plotting.average_input_tokens_clients_fig")
    @patch("llmeter.plotting.plotting.average_output_tokens_clients_fig")
    def test_plot_load_test_results_passes_log_scale_parameter(
        self,
        mock_avg_out,
        mock_avg_in,
        mock_error,
        mock_rpm,
        mock_latency,
        sample_load_test_result,
    ):
        """Test that log_scale parameter is properly passed to all plotting functions."""
        # Mock all the individual plotting functions
        for mock_func in [
            mock_latency,
            mock_rpm,
            mock_error,
            mock_avg_in,
            mock_avg_out,
        ]:
            mock_func.return_value = MagicMock()

        # Test with log_scale=False
        plot_load_test_results(sample_load_test_result, log_scale=False)

        # Verify log_scale parameter was passed correctly
        for call in mock_latency.call_args_list:
            assert call[1]["log_scale"] is False

        assert mock_rpm.call_args[1]["log_scale"] is False
        assert mock_error.call_args[1]["log_scale"] is False
        assert mock_avg_in.call_args[1]["log_scale"] is False
        assert mock_avg_out.call_args[1]["log_scale"] is False

    @patch("llmeter.plotting.plotting.latency_clients_fig")
    @patch("llmeter.plotting.plotting.rpm_clients_fig")
    @patch("llmeter.plotting.plotting.error_clients_fig")
    @patch("llmeter.plotting.plotting.average_input_tokens_clients_fig")
    @patch("llmeter.plotting.plotting.average_output_tokens_clients_fig")
    def test_plot_load_test_results_with_default_log_scale(
        self,
        mock_avg_out,
        mock_avg_in,
        mock_error,
        mock_rpm,
        mock_latency,
        sample_load_test_result,
    ):
        """Test that default log_scale=True is used when not specified."""
        # Mock all the individual plotting functions
        for mock_func in [
            mock_latency,
            mock_rpm,
            mock_error,
            mock_avg_in,
            mock_avg_out,
        ]:
            mock_func.return_value = MagicMock()

        # Call without specifying log_scale (should default to True)
        plot_load_test_results(sample_load_test_result)

        # Verify log_scale parameter defaults to True
        for call in mock_latency.call_args_list:
            assert call[1]["log_scale"] is True

        assert mock_rpm.call_args[1]["log_scale"] is True
        assert mock_error.call_args[1]["log_scale"] is True
        assert mock_avg_in.call_args[1]["log_scale"] is True
        assert mock_avg_out.call_args[1]["log_scale"] is True

    @patch("llmeter.plotting.plotting.latency_clients_fig")
    def test_plot_load_test_results_calls_latency_with_correct_dimensions(
        self, mock_latency, sample_load_test_result
    ):
        """Test that latency_clients_fig is called with correct dimension parameters."""
        mock_latency.return_value = MagicMock()

        with patch("llmeter.plotting.plotting.rpm_clients_fig"), patch(
            "llmeter.plotting.plotting.error_clients_fig"
        ), patch("llmeter.plotting.plotting.average_input_tokens_clients_fig"), patch(
            "llmeter.plotting.plotting.average_output_tokens_clients_fig"
        ):
            plot_load_test_results(sample_load_test_result)

        # Verify latency_clients_fig was called twice with correct dimensions
        assert mock_latency.call_count == 2
        calls = mock_latency.call_args_list

        # First call should be for time_to_first_token
        assert calls[0][0] == (sample_load_test_result, "time_to_first_token")
        # Second call should be for time_to_last_token
        assert calls[1][0] == (sample_load_test_result, "time_to_last_token")

    def test_plot_load_test_results_with_empty_results(self):
        """Test plot_load_test_results behavior with empty results."""
        empty_load_test = LoadTestResult(
            results={}, test_name="empty", output_path="/tmp/empty"
        )

        with patch("llmeter.plotting.plotting.latency_clients_fig") as mock_latency:
            mock_latency.side_effect = ValueError("No valid data points found")

            with pytest.raises(ValueError, match="No valid data points found"):
                plot_load_test_results(empty_load_test)

    @patch("llmeter.plotting.plotting.latency_clients_fig")
    @patch("llmeter.plotting.plotting.rpm_clients_fig")
    @patch("llmeter.plotting.plotting.error_clients_fig")
    @patch("llmeter.plotting.plotting.average_input_tokens_clients_fig")
    @patch("llmeter.plotting.plotting.average_output_tokens_clients_fig")
    def test_plot_load_test_results_preserves_figure_objects(
        self,
        mock_avg_out,
        mock_avg_in,
        mock_error,
        mock_rpm,
        mock_latency,
        sample_load_test_result,
    ):
        """Test that the actual figure objects are preserved in the return dict."""
        # Create distinct mock figures for each plot type
        mock_ttft_fig = MagicMock()
        mock_ttft_fig.name = "ttft_figure"

        mock_ttlt_fig = MagicMock()
        mock_ttlt_fig.name = "ttlt_figure"

        mock_rpm_fig = MagicMock()
        mock_rpm_fig.name = "rpm_figure"

        mock_error_fig = MagicMock()
        mock_error_fig.name = "error_figure"

        mock_input_fig = MagicMock()
        mock_input_fig.name = "input_figure"

        mock_output_fig = MagicMock()
        mock_output_fig.name = "output_figure"

        # Set up mocks to return different figures for different calls
        mock_latency.side_effect = [mock_ttft_fig, mock_ttlt_fig]
        mock_rpm.return_value = mock_rpm_fig
        mock_error.return_value = mock_error_fig
        mock_avg_in.return_value = mock_input_fig
        mock_avg_out.return_value = mock_output_fig

        result = plot_load_test_results(sample_load_test_result)

        # Verify that the correct figure objects are returned
        assert result["time_to_first_token"] is mock_ttft_fig
        assert result["time_to_last_token"] is mock_ttlt_fig
        assert result["requests_per_minute"] is mock_rpm_fig
        assert result["error_rate"] is mock_error_fig
        assert result["average_input_tokens_clients"] is mock_input_fig
        assert result["average_output_tokens_clients"] is mock_output_fig

    @patch("llmeter.plotting.plotting.latency_clients_fig")
    @patch("llmeter.plotting.plotting.rpm_clients_fig")
    @patch("llmeter.plotting.plotting.error_clients_fig")
    @patch("llmeter.plotting.plotting.average_input_tokens_clients_fig")
    @patch("llmeter.plotting.plotting.average_output_tokens_clients_fig")
    def test_plot_load_test_results_figure_format_compatibility(
        self,
        mock_avg_out,
        mock_avg_in,
        mock_error,
        mock_rpm,
        mock_latency,
        sample_load_test_result,
    ):
        """Test that returned figures are compatible with different output formats."""
        # Create mock figures with format-specific methods
        mock_figs = []
        for i in range(6):
            mock_fig = MagicMock()
            mock_fig.write_image = MagicMock()
            mock_fig.write_html = MagicMock()
            mock_fig.to_image = MagicMock()
            mock_figs.append(mock_fig)

        # Set up mocks to return format-compatible figures
        mock_latency.side_effect = mock_figs[:2]  # TTFT and TTLT
        mock_rpm.return_value = mock_figs[2]
        mock_error.return_value = mock_figs[3]
        mock_avg_in.return_value = mock_figs[4]
        mock_avg_out.return_value = mock_figs[5]

        result = plot_load_test_results(sample_load_test_result)

        # Verify all returned figures have the necessary methods for format output
        for plot_name, fig in result.items():
            assert hasattr(
                fig, "write_image"
            ), f"Figure {plot_name} missing write_image method"
            assert hasattr(
                fig, "write_html"
            ), f"Figure {plot_name} missing write_html method"
            assert hasattr(
                fig, "to_image"
            ), f"Figure {plot_name} missing to_image method"

    def test_plot_load_test_results_validates_return_structure_for_file_formats(
        self, sample_load_test_result
    ):
        """Test that plot_load_test_results returns structure suitable for different file formats."""
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
            # Create mock figures that simulate plotly figure behavior
            def create_format_compatible_figure():
                fig = MagicMock()
                fig.write_image = MagicMock()
                fig.write_html = MagicMock()
                fig.to_image = MagicMock(return_value=b"fake_image_data")
                fig.to_html = MagicMock(return_value="<html>fake_html</html>")
                return fig

            mock_figs = [create_format_compatible_figure() for _ in range(6)]
            mock_latency.side_effect = mock_figs[:2]
            mock_rpm.return_value = mock_figs[2]
            mock_error.return_value = mock_figs[3]
            mock_avg_in.return_value = mock_figs[4]
            mock_avg_out.return_value = mock_figs[5]

            result = plot_load_test_results(sample_load_test_result)

            # Test that each figure can be used for PNG format
            for plot_name, fig in result.items():
                # Simulate PNG export
                fig.write_image(f"/tmp/{plot_name}.png")

            # Verify PNG calls were made
            for i, fig in enumerate(mock_figs):
                fig.write_image.assert_called()

            # Reset mocks and test HTML format
            for fig in mock_figs:
                fig.reset_mock()

            # Test that each figure can be used for HTML format
            for plot_name, fig in result.items():
                # Simulate HTML export
                fig.write_html(f"/tmp/{plot_name}.html")

            # Verify HTML calls were made
            for i, fig in enumerate(mock_figs):
                fig.write_html.assert_called()

    @patch("llmeter.plotting.plotting.latency_clients_fig")
    @patch("llmeter.plotting.plotting.rpm_clients_fig")
    @patch("llmeter.plotting.plotting.error_clients_fig")
    @patch("llmeter.plotting.plotting.average_input_tokens_clients_fig")
    @patch("llmeter.plotting.plotting.average_output_tokens_clients_fig")
    def test_plot_load_test_results_consistent_figure_interface(
        self,
        mock_avg_out,
        mock_avg_in,
        mock_error,
        mock_rpm,
        mock_latency,
        sample_load_test_result,
    ):
        """Test that all returned figures have consistent interface for format operations."""
        # Create mock figures with consistent interface
        mock_figs = []
        for i in range(6):
            fig = MagicMock()
            # Ensure all figures have the same interface
            fig.write_image = MagicMock()
            fig.write_html = MagicMock()
            fig.show = MagicMock()
            fig.update_layout = MagicMock()
            mock_figs.append(fig)

        mock_latency.side_effect = mock_figs[:2]
        mock_rpm.return_value = mock_figs[2]
        mock_error.return_value = mock_figs[3]
        mock_avg_in.return_value = mock_figs[4]
        mock_avg_out.return_value = mock_figs[5]

        result = plot_load_test_results(sample_load_test_result)

        # Verify all figures have consistent interface
        required_methods = ["write_image", "write_html", "show", "update_layout"]
        for plot_name, fig in result.items():
            for method in required_methods:
                assert hasattr(
                    fig, method
                ), f"Figure {plot_name} missing {method} method"
                assert callable(
                    getattr(fig, method)
                ), f"Figure {plot_name} {method} is not callable"
