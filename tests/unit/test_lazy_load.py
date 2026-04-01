# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from upath import UPath

from llmeter.endpoints.base import InvocationResponse
from llmeter.experiments import LoadTestResult
from llmeter.results import Result


@pytest.fixture
def sample_responses():
    return [
        InvocationResponse(
            id=f"test_{i}",
            response_text=f"Response {i}",
            input_prompt=f"Prompt {i}",
            time_to_first_token=0.1 * i,
            time_to_last_token=0.2 * i,
            num_tokens_output=10 * i,
            num_tokens_input=5 * i,
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def sample_result(sample_responses):
    return Result(
        responses=sample_responses,
        total_requests=5,
        clients=1,
        n_requests=5,
        total_test_time=1.0,
        run_name="Test Run",
    )


@pytest.fixture
def saved_result(sample_result, tmp_path):
    """A sample result saved to disk, returns the output path."""
    output_path = UPath(tmp_path) / "test_output"
    sample_result.save(output_path)
    return output_path


# --- Result.load with load_responses=False ---


class TestLoadWithoutResponses:
    def test_load_no_responses_returns_empty_responses(self, saved_result):
        loaded = Result.load(saved_result, load_responses=False)
        assert loaded.responses == []

    def test_load_no_responses_preserves_summary(self, sample_result, saved_result):
        loaded = Result.load(saved_result, load_responses=False)
        assert loaded.total_requests == sample_result.total_requests
        assert loaded.clients == sample_result.clients
        assert loaded.n_requests == sample_result.n_requests
        assert loaded.run_name == sample_result.run_name

    def test_load_no_responses_stats_from_stats_json(self, sample_result, saved_result):
        original_stats = sample_result.stats
        loaded = Result.load(saved_result, load_responses=False)
        loaded_stats = loaded.stats

        assert loaded_stats["total_requests"] == original_stats["total_requests"]
        assert loaded_stats["time_to_last_token-average"] == pytest.approx(
            original_stats["time_to_last_token-average"]
        )
        assert loaded_stats["time_to_first_token-p50"] == pytest.approx(
            original_stats["time_to_first_token-p50"]
        )
        assert loaded_stats["requests_per_minute"] == pytest.approx(
            original_stats["requests_per_minute"]
        )

    def test_load_no_responses_sets_output_path(self, saved_result):
        loaded = Result.load(saved_result, load_responses=False)
        assert loaded.output_path is not None
        assert str(saved_result) in str(loaded.output_path)

    def test_load_no_responses_without_stats_json(self, saved_result):
        """When stats.json is missing, stats should still work (empty responses)."""
        (saved_result / "stats.json").unlink()
        loaded = Result.load(saved_result, load_responses=False)
        # Should not raise, but stats will be computed from empty responses
        stats = loaded.stats
        assert stats["total_requests"] == 5
        assert stats["failed_requests"] == 0

    def test_load_with_responses_true_is_default(self, saved_result):
        loaded = Result.load(saved_result)
        assert len(loaded.responses) == 5

    def test_load_no_responses_with_string_path(self, saved_result):
        loaded = Result.load(str(saved_result), load_responses=False)
        assert loaded.responses == []
        assert loaded.total_requests == 5

    def test_load_no_responses_contributed_stats(self, saved_result):
        loaded = Result.load(saved_result, load_responses=False)
        loaded._update_contributed_stats({"custom_metric": 42})
        assert loaded.stats["custom_metric"] == 42


# --- Result.load_responses() on-demand loading ---


class TestLoadResponsesOnDemand:
    def test_load_responses_populates_responses(self, saved_result):
        loaded = Result.load(saved_result, load_responses=False)
        assert loaded.responses == []

        responses = loaded.load_responses()
        assert len(responses) == 5
        assert len(loaded.responses) == 5

    def test_load_responses_returns_correct_data(self, sample_responses, saved_result):
        loaded = Result.load(saved_result, load_responses=False)
        loaded.load_responses()

        for orig, loaded_resp in zip(sample_responses, loaded.responses):
            assert orig.id == loaded_resp.id
            assert orig.response_text == loaded_resp.response_text
            assert orig.time_to_first_token == loaded_resp.time_to_first_token
            assert orig.time_to_last_token == loaded_resp.time_to_last_token

    def test_load_responses_recomputes_stats(self, saved_result):
        loaded = Result.load(saved_result, load_responses=True)
        original_stats = loaded._preloaded_stats.copy()

        loaded.load_responses()
        # Stats should be recomputed (same values, but a fresh dict)
        assert loaded._preloaded_stats is not original_stats
        assert loaded._preloaded_stats == original_stats

    def test_load_responses_stats_match_full_load(self, saved_result):
        full = Result.load(saved_result, load_responses=True)
        lazy = Result.load(saved_result, load_responses=False)
        lazy.load_responses()

        full_stats = full.stats
        lazy_stats = lazy.stats

        for key in [
            "time_to_last_token-average",
            "time_to_first_token-p50",
            "num_tokens_output-p90",
            "failed_requests",
            "requests_per_minute",
        ]:
            assert lazy_stats[key] == pytest.approx(full_stats[key]), (
                f"Mismatch on {key}"
            )

    def test_load_responses_no_output_path_raises(self):
        result = Result(
            responses=[], total_requests=0, clients=0, n_requests=0, output_path=None
        )
        with pytest.raises(ValueError, match="No output_path set"):
            result.load_responses()

    def test_load_responses_missing_file_raises(self, tmp_path):
        result = Result(
            responses=[],
            total_requests=0,
            clients=0,
            n_requests=0,
            output_path=str(tmp_path),
        )
        with pytest.raises(FileNotFoundError):
            result.load_responses()


# --- LoadTestResult.load with load_responses ---


class TestLoadTestResultWithoutResponses:
    @patch("llmeter.experiments.Result.load")
    def test_passes_load_responses_false(self, mock_load, tmp_path):
        mock_result = MagicMock(spec=Result)
        mock_result.clients = 1
        mock_load.return_value = mock_result

        result_dir = tmp_path / "result1"
        result_dir.mkdir()

        LoadTestResult.load(tmp_path, load_responses=False)

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs["load_responses"] is False

    @patch("llmeter.experiments.Result.load")
    def test_passes_load_responses_true_by_default(self, mock_load, tmp_path):
        mock_result = MagicMock(spec=Result)
        mock_result.clients = 1
        mock_load.return_value = mock_result

        result_dir = tmp_path / "result1"
        result_dir.mkdir()

        LoadTestResult.load(tmp_path)

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs["load_responses"] is True
