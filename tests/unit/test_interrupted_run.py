# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for graceful save/load of interrupted runs (issue #73)."""

import asyncio
import json
import os
import signal

import pytest
from upath import UPath

from llmeter.endpoints.base import InvocationResponse
from llmeter.results import Result
from llmeter.runner import _GracefulShutdown


@pytest.fixture
def sample_responses():
    """Create a set of sample responses as would be found in responses.jsonl."""
    from datetime import datetime, timezone

    return [
        InvocationResponse(
            id=f"resp_{i}",
            response_text=f"Response text {i}",
            input_prompt=f"Test prompt {i}",
            time_to_first_token=0.1 * (i + 1),
            time_to_last_token=0.3 * (i + 1),
            num_tokens_input=10 * (i + 1),
            num_tokens_output=20 * (i + 1),
            request_time=datetime(
                2025, 6, 1, 10, 0, i * 2, tzinfo=timezone.utc
            ),
        )
        for i in range(5)
    ]


@pytest.fixture
def interrupted_run_dir(tmp_path, sample_responses):
    """Create a directory simulating an interrupted run (no summary.json).

    Contains only responses.jsonl and run_config.json, as would be left behind
    when a run is interrupted before Result.save() completes.
    """
    from llmeter.json_utils import llmeter_default_serializer

    run_dir = UPath(tmp_path / "interrupted_run")
    run_dir.mkdir(parents=True)

    # Write responses.jsonl (written incrementally during the run)
    responses_path = run_dir / "responses.jsonl"
    with responses_path.open("w") as f:
        for resp in sample_responses:
            f.write(resp.to_json() + "\n")

    # Write run_config.json (written at the start of the run)
    config = {
        "endpoint": {
            "model_id": "test-model-id",
            "endpoint_name": "test-endpoint",
            "provider": "test-provider",
        },
        "tokenizer": {"type": "DummyTokenizer"},
        "clients": 3,
        "n_requests": 10,
        "run_name": "interrupted-test-run",
        "run_description": "A test run that was interrupted",
        "payload": "payload.jsonl",
        "timeout": 60,
        "low_memory": False,
        "run_duration": None,
        "callbacks": None,
        "progress_bar_stats": None,
        "disable_per_client_progress_bar": False,
        "disable_clients_progress_bar": False,
    }
    config_path = run_dir / "run_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, default=llmeter_default_serializer)

    return run_dir


class TestLoadWithoutSummary:
    """Tests for Result.load() when summary.json is missing."""

    def test_load_recovers_responses(self, interrupted_run_dir, sample_responses):
        """Should load responses from responses.jsonl when summary.json is absent."""
        result = Result.load(interrupted_run_dir)

        assert len(result.responses) == len(sample_responses)
        for orig, loaded in zip(sample_responses, result.responses):
            assert orig.id == loaded.id
            assert orig.response_text == loaded.response_text
            assert orig.num_tokens_input == loaded.num_tokens_input
            assert orig.num_tokens_output == loaded.num_tokens_output

    def test_load_extracts_config_metadata(self, interrupted_run_dir):
        """Should extract endpoint and run metadata from run_config.json."""
        result = Result.load(interrupted_run_dir)

        assert result.clients == 3
        assert result.n_requests == 10
        assert result.run_name == "interrupted-test-run"
        assert result.run_description == "A test run that was interrupted"
        assert result.model_id == "test-model-id"
        assert result.endpoint_name == "test-endpoint"
        assert result.provider == "test-provider"

    def test_load_derives_timestamps_from_responses(self, interrupted_run_dir):
        """start_time and end_time should be derived from response request_time."""
        from datetime import datetime, timezone

        result = Result.load(interrupted_run_dir)

        # First response has request_time 10:00:00, last has 10:00:08
        assert result.start_time == datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert result.first_request_time == datetime(
            2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc
        )
        assert result.last_request_time == datetime(
            2025, 6, 1, 10, 0, 8, tzinfo=timezone.utc
        )
        assert result.end_time == datetime(
            2025, 6, 1, 10, 0, 8, tzinfo=timezone.utc
        )

    def test_load_sets_total_requests_from_responses(self, interrupted_run_dir):
        """total_requests should be set to the number of recovered responses."""
        result = Result.load(interrupted_run_dir)
        assert result.total_requests == 5

    def test_load_computes_stats_from_responses(self, interrupted_run_dir):
        """Stats should be computed from the recovered responses."""
        result = Result.load(interrupted_run_dir)
        stats = result.stats

        assert "time_to_first_token-p50" in stats
        assert "time_to_last_token-average" in stats
        assert "num_tokens_output-p90" in stats
        assert stats["failed_requests"] == 0

    def test_load_sets_output_path(self, interrupted_run_dir):
        """output_path should be set so load_responses() works later."""
        result = Result.load(interrupted_run_dir)
        assert result.output_path is not None
        assert str(interrupted_run_dir) in str(result.output_path)

    def test_load_without_responses_flag(self, interrupted_run_dir):
        """Should work with load_responses=False (metadata only)."""
        result = Result.load(interrupted_run_dir, load_responses=False)

        assert result.responses == []
        assert result.clients == 3
        assert result.run_name == "interrupted-test-run"
        # Stats should be None since we have no stats.json and didn't load responses
        assert result._preloaded_stats is None

    def test_load_only_responses_no_config(self, tmp_path, sample_responses):
        """Should work even if run_config.json is missing."""
        run_dir = UPath(tmp_path / "responses_only")
        run_dir.mkdir(parents=True)

        responses_path = run_dir / "responses.jsonl"
        with responses_path.open("w") as f:
            for resp in sample_responses:
                f.write(resp.to_json() + "\n")

        result = Result.load(run_dir)

        assert len(result.responses) == 5
        assert result.total_requests == 5
        # Config fields should be defaults when run_config.json is missing
        assert result.clients == 1  # default
        assert result.model_id is None

    def test_load_with_stats_json_no_responses(self, tmp_path):
        """Should load pre-computed stats even without responses.jsonl."""
        run_dir = UPath(tmp_path / "stats_only")
        run_dir.mkdir(parents=True)

        stats = {
            "total_requests": 100,
            "failed_requests": 2,
            "time_to_first_token-p50": 0.25,
            "time_to_last_token-average": 0.8,
        }
        stats_path = run_dir / "stats.json"
        with stats_path.open("w") as f:
            json.dump(stats, f)

        result = Result.load(run_dir)

        assert result._preloaded_stats is not None
        assert result._preloaded_stats["total_requests"] == 100
        assert result._preloaded_stats["failed_requests"] == 2
        assert result._preloaded_stats["time_to_first_token-p50"] == 0.25

    def test_load_empty_directory_raises(self, tmp_path):
        """Should raise FileNotFoundError when no useful files are present."""
        empty_dir = UPath(tmp_path / "empty")
        empty_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="no data to recover"):
            Result.load(empty_dir)

    def test_load_nonexistent_directory_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent paths."""
        with pytest.raises(FileNotFoundError):
            Result.load(tmp_path / "does_not_exist")

    def test_load_recovers_interrupted_run_with_stats(self, tmp_path, sample_responses):
        """Simulates the case where both stats.json and responses.jsonl exist
        (e.g. the interrupt handler managed to write stats before exiting)."""
        from llmeter.json_utils import llmeter_default_serializer

        run_dir = UPath(tmp_path / "partial_with_stats")
        run_dir.mkdir(parents=True)

        responses_path = run_dir / "responses.jsonl"
        with responses_path.open("w") as f:
            for resp in sample_responses:
                f.write(resp.to_json() + "\n")

        stats = {
            "total_requests": 5,
            "failed_requests": 0,
            "interrupted": True,
            "time_to_first_token-p50": 0.3,
        }
        stats_path = run_dir / "stats.json"
        with stats_path.open("w") as f:
            json.dump(stats, f, default=llmeter_default_serializer)

        result = Result.load(run_dir)

        # When both are present and we load responses, stats are recomputed
        # from responses (which is more accurate)
        assert len(result.responses) == 5
        assert result.stats["failed_requests"] == 0

    def test_load_responses_after_recovery(self, interrupted_run_dir):
        """load_responses() should work on a recovered result."""
        result = Result.load(interrupted_run_dir, load_responses=False)
        assert result.responses == []

        responses = result.load_responses()
        assert len(responses) == 5
        assert responses[0].id == "resp_0"


class TestInterruptedRunSave:
    """Tests for saving partial results during interrupt."""

    def test_save_interrupted_result_creates_all_files(
        self, tmp_path, sample_responses
    ):
        """A partial result from an interrupted run should save correctly."""
        from datetime import datetime, timezone

        result = Result(
            responses=sample_responses,
            total_requests=5,
            clients=2,
            n_requests=5,
            total_test_time=None,  # Unknown for interrupted runs
            start_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 6, 1, 10, 0, 3, tzinfo=timezone.utc),
            model_id="test-model",
            run_name="partial-run",
        )
        result._preloaded_stats = Result._compute_stats(result)
        result._preloaded_stats["interrupted"] = True

        output_path = UPath(tmp_path / "partial_output")
        result.save(output_path)

        assert (output_path / "summary.json").exists()
        assert (output_path / "stats.json").exists()
        assert (output_path / "responses.jsonl").exists()

        # Verify stats.json contains the interrupted marker
        with (output_path / "stats.json").open() as f:
            stats = json.load(f)
        assert stats.get("interrupted") is True

    def test_round_trip_interrupted_result(self, tmp_path, sample_responses):
        """A saved interrupted result should load back correctly."""
        from datetime import datetime, timezone

        result = Result(
            responses=sample_responses,
            total_requests=5,
            clients=2,
            n_requests=5,
            total_test_time=None,
            start_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2025, 6, 1, 10, 0, 3, tzinfo=timezone.utc),
            model_id="test-model",
            run_name="partial-run",
        )
        result._preloaded_stats = Result._compute_stats(result)
        result._preloaded_stats["interrupted"] = True

        output_path = UPath(tmp_path / "rt_output")
        result.save(output_path)

        loaded = Result.load(output_path)
        assert loaded.total_requests == 5
        assert loaded.run_name == "partial-run"
        assert len(loaded.responses) == 5
        assert loaded.stats.get("interrupted") is True


class TestLoadWithoutSummaryEdgeCases:
    """Edge cases for _load_without_summary."""

    def test_responses_without_request_time(self, tmp_path):
        """Timestamps should be None when responses lack request_time."""
        run_dir = UPath(tmp_path / "no_timestamps")
        run_dir.mkdir(parents=True)

        responses = [
            InvocationResponse(
                id=f"r{i}",
                response_text=f"text {i}",
                num_tokens_input=5,
                num_tokens_output=10,
                # No request_time set
            )
            for i in range(3)
        ]
        responses_path = run_dir / "responses.jsonl"
        with responses_path.open("w") as f:
            for resp in responses:
                f.write(resp.to_json() + "\n")

        result = Result.load(run_dir)

        assert result.start_time is None
        assert result.first_request_time is None
        assert result.last_request_time is None
        assert result.end_time is None
        assert result.total_requests == 3

    def test_corrupted_run_config_json(self, tmp_path, sample_responses):
        """Should recover even if run_config.json is corrupted."""
        run_dir = UPath(tmp_path / "bad_config")
        run_dir.mkdir(parents=True)

        responses_path = run_dir / "responses.jsonl"
        with responses_path.open("w") as f:
            for resp in sample_responses:
                f.write(resp.to_json() + "\n")

        # Write invalid JSON
        config_path = run_dir / "run_config.json"
        with config_path.open("w") as f:
            f.write("{invalid json content!!")

        result = Result.load(run_dir)

        # Should still load responses and use defaults
        assert len(result.responses) == 5
        assert result.clients == 1  # default
        assert result.model_id is None

    def test_corrupted_stats_json(self, tmp_path, sample_responses):
        """Should fall back to computing stats when stats.json is corrupted."""
        run_dir = UPath(tmp_path / "bad_stats")
        run_dir.mkdir(parents=True)

        responses_path = run_dir / "responses.jsonl"
        with responses_path.open("w") as f:
            for resp in sample_responses:
                f.write(resp.to_json() + "\n")

        stats_path = run_dir / "stats.json"
        with stats_path.open("w") as f:
            f.write("not valid json")

        result = Result.load(run_dir)

        # Should compute stats from responses as fallback
        assert len(result.responses) == 5
        assert result.stats["failed_requests"] == 0
        assert "time_to_first_token-p50" in result.stats

    def test_partial_responses_with_errors(self, tmp_path):
        """Recovery should handle responses with errors correctly."""
        from datetime import datetime, timezone

        run_dir = UPath(tmp_path / "with_errors")
        run_dir.mkdir(parents=True)

        responses = [
            InvocationResponse(
                id="ok_1",
                response_text="good response",
                num_tokens_input=10,
                num_tokens_output=20,
                time_to_first_token=0.1,
                time_to_last_token=0.3,
                request_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
            ),
            InvocationResponse(
                id="err_1",
                error="timeout",
                response_text="",
                request_time=datetime(2025, 6, 1, 10, 0, 1, tzinfo=timezone.utc),
            ),
            InvocationResponse(
                id="ok_2",
                response_text="another good one",
                num_tokens_input=8,
                num_tokens_output=15,
                time_to_first_token=0.2,
                time_to_last_token=0.5,
                request_time=datetime(2025, 6, 1, 10, 0, 2, tzinfo=timezone.utc),
            ),
        ]
        responses_path = run_dir / "responses.jsonl"
        with responses_path.open("w") as f:
            for resp in responses:
                f.write(resp.to_json() + "\n")

        result = Result.load(run_dir)

        assert result.total_requests == 3
        assert result.stats["failed_requests"] == 1
        assert result.start_time == datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert result.end_time == datetime(2025, 6, 1, 10, 0, 2, tzinfo=timezone.utc)

    def test_single_response(self, tmp_path):
        """Recovery should work with a single response."""
        from datetime import datetime, timezone

        run_dir = UPath(tmp_path / "single")
        run_dir.mkdir(parents=True)

        responses = [
            InvocationResponse(
                id="only",
                response_text="single",
                num_tokens_input=5,
                num_tokens_output=10,
                time_to_first_token=0.2,
                time_to_last_token=0.4,
                request_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
            )
        ]
        responses_path = run_dir / "responses.jsonl"
        with responses_path.open("w") as f:
            for resp in responses:
                f.write(resp.to_json() + "\n")

        result = Result.load(run_dir)

        assert result.total_requests == 1
        assert result.start_time == result.end_time
        assert result.first_request_time == result.last_request_time


class TestGracefulShutdown:
    """Tests for the _GracefulShutdown context manager."""

    @pytest.mark.asyncio
    async def test_registers_and_removes_signal_handlers(self):
        """Handlers should be installed on enter and removed on exit."""
        loop = asyncio.get_running_loop()
        shutdown = _GracefulShutdown(loop)

        with shutdown:
            # Handlers are registered
            assert len(shutdown._registered) == 2
            assert signal.SIGINT in shutdown._registered
            assert signal.SIGTERM in shutdown._registered

        # Handlers are removed after exit
        assert shutdown._registered == []

    @pytest.mark.asyncio
    async def test_received_signal_initially_none(self):
        """received_signal should be None before any signal arrives."""
        loop = asyncio.get_running_loop()
        shutdown = _GracefulShutdown(loop)

        with shutdown:
            assert shutdown.received_signal is None

    @pytest.mark.asyncio
    async def test_handle_sets_received_signal(self):
        """Calling _handle should set received_signal."""
        loop = asyncio.get_running_loop()
        shutdown = _GracefulShutdown(loop)

        # Use a separate task so cancelling it doesn't kill the test
        async def target():
            await asyncio.sleep(10)

        task = asyncio.ensure_future(target())
        shutdown._task = task

        shutdown._handle(signal.SIGTERM)
        assert shutdown.received_signal == signal.SIGTERM

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_handle_cancels_task(self):
        """_handle should request cancellation of the captured task."""
        loop = asyncio.get_running_loop()
        shutdown = _GracefulShutdown(loop)

        async def target():
            await asyncio.sleep(10)

        task = asyncio.ensure_future(target())
        shutdown._task = task

        shutdown._handle(signal.SIGINT)
        assert task.cancelling() > 0

        # Cleanup
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self):
        """Signal handlers should be removed even if an exception occurs."""
        loop = asyncio.get_running_loop()
        shutdown = _GracefulShutdown(loop)

        with pytest.raises(RuntimeError):
            with shutdown:
                assert len(shutdown._registered) == 2
                raise RuntimeError("boom")

        assert shutdown._registered == []


class TestRunnerInterruptFlow:
    """Tests for the full Runner interrupt flow via signal simulation."""

    @pytest.fixture
    def mock_endpoint(self):
        from unittest.mock import MagicMock

        from llmeter.endpoints.base import Endpoint

        endpoint = MagicMock(spec=Endpoint)
        endpoint.endpoint_name = "test-endpoint"
        endpoint.model_id = "test-model"
        endpoint.provider = "test-provider"
        endpoint.invoke.return_value = InvocationResponse(
            id="1", input_prompt="test", response_text="response"
        )
        endpoint.to_dict.return_value = {
            "endpoint_name": "test-endpoint",
            "model_id": "test-model",
            "provider": "test-provider",
            "endpoint_type": "mock",
        }
        return endpoint

    @pytest.mark.asyncio
    async def test_sigint_saves_partial_result(self, tmp_path, mock_endpoint):
        """SIGINT during a run should save partial results and return."""
        from llmeter.runner import Runner
        from llmeter.tokenizers import DummyTokenizer

        output_path = UPath(tmp_path / "sigint_run")

        runner = Runner(
            endpoint=mock_endpoint,
            tokenizer=DummyTokenizer(),
            output_path=output_path,
            clients=1,
            n_requests=200,
            payload=[{"prompt": "hello"}],
        )

        # Fast enough to collect several before the signal, but enough total
        # that the run won't finish before the signal arrives
        call_count = 0

        def fast_invoke(payload):
            nonlocal call_count
            import time

            call_count += 1
            time.sleep(0.01)
            return InvocationResponse(
                id=f"r{call_count}",
                input_prompt="test",
                response_text="resp",
                num_tokens_input=5,
                num_tokens_output=10,
            )

        mock_endpoint.invoke = fast_invoke

        # Give time for some requests to complete before interrupting
        loop = asyncio.get_running_loop()
        loop.call_later(0.5, os.kill, os.getpid(), signal.SIGINT)

        result = await runner.run()

        # Result should be marked as interrupted
        assert result.stats.get("interrupted") is True
        # Should NOT have completed all requests
        assert result.total_requests < 200

        # Output files should exist
        run_output = output_path / result.run_name
        assert (run_output / "summary.json").exists()
        assert (run_output / "stats.json").exists()

    @pytest.mark.asyncio
    async def test_sigterm_saves_partial_result(self, tmp_path, mock_endpoint):
        """SIGTERM during a run should save partial results and return."""
        from llmeter.runner import Runner
        from llmeter.tokenizers import DummyTokenizer

        output_path = UPath(tmp_path / "sigterm_run")

        runner = Runner(
            endpoint=mock_endpoint,
            tokenizer=DummyTokenizer(),
            output_path=output_path,
            clients=1,
            n_requests=200,
            payload=[{"prompt": "hello"}],
        )

        call_count = 0

        def fast_invoke(payload):
            nonlocal call_count
            import time

            call_count += 1
            time.sleep(0.01)
            return InvocationResponse(
                id=f"r{call_count}",
                input_prompt="test",
                response_text="resp",
                num_tokens_input=5,
                num_tokens_output=10,
            )

        mock_endpoint.invoke = fast_invoke

        loop = asyncio.get_running_loop()
        loop.call_later(0.5, os.kill, os.getpid(), signal.SIGTERM)

        result = await runner.run()

        assert result.stats.get("interrupted") is True
        assert result.total_requests < 200

    @pytest.mark.asyncio
    async def test_unrelated_cancellation_propagates(self, mock_endpoint):
        """CancelledError not from our signal handler should propagate."""
        from llmeter.runner import Runner
        from llmeter.tokenizers import DummyTokenizer

        runner = Runner(
            endpoint=mock_endpoint,
            tokenizer=DummyTokenizer(),
            clients=1,
            n_requests=200,
            payload=[{"prompt": "hello"}],
        )

        def slow_invoke(payload):
            import time

            time.sleep(0.05)
            return InvocationResponse(
                id="x", input_prompt="test", response_text="resp"
            )

        mock_endpoint.invoke = slow_invoke

        # Run in a task and cancel from outside (not via our signal handler)
        task = asyncio.ensure_future(runner.run())
        await asyncio.sleep(0.1)  # Let it start
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_interrupted_result_has_correct_metadata(
        self, tmp_path, mock_endpoint
    ):
        """Interrupted result should have timing info from collected responses."""
        from llmeter.runner import Runner
        from llmeter.tokenizers import DummyTokenizer

        output_path = UPath(tmp_path / "meta_run")

        runner = Runner(
            endpoint=mock_endpoint,
            tokenizer=DummyTokenizer(),
            output_path=output_path,
            clients=1,
            n_requests=200,
            payload=[{"prompt": "hello"}],
        )

        call_count = 0

        def counting_invoke(payload):
            nonlocal call_count
            import time

            call_count += 1
            time.sleep(0.01)
            return InvocationResponse(
                id=f"r{call_count}",
                input_prompt="test",
                response_text="resp",
                num_tokens_input=5,
                num_tokens_output=10,
            )

        mock_endpoint.invoke = counting_invoke

        loop = asyncio.get_running_loop()
        loop.call_later(0.5, os.kill, os.getpid(), signal.SIGINT)

        result = await runner.run()

        assert result.start_time is not None
        assert result.end_time is not None
        assert result.end_time >= result.start_time
        assert result.total_test_time is None  # Unknown for interrupted runs
        assert result.clients == 1
        assert result.model_id == "test-model"
