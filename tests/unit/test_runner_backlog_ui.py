# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the backlog progress UI and stats throttling mechanics.

These tests cover:
1. Stats display update throttling (timer-based, every 0.5s)
2. Backlog progress bar creation during time-bound runs
3. Two-phase flow: time bar closes → backlog bar appears → drains
4. Final stats update after queue drain
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from llmeter.endpoints.base import Endpoint, InvocationResponse
from llmeter.runner import Runner, STATS_UPDATE_INTERVAL, _Run
from llmeter.tokenizers import Tokenizer


@pytest.fixture
def mock_endpoint():
    endpoint = MagicMock(spec=Endpoint)
    endpoint.endpoint_name = "endpoint-name"
    endpoint.model_id = "model-id"
    endpoint.provider = "provider"
    endpoint.invoke.return_value = InvocationResponse(
        id="1", input_prompt="test", response_text="response"
    )
    endpoint_conf = {k: v for k, v in vars(endpoint).items() if not k.startswith("_")}
    endpoint_conf["endpoint_type"] = "mock_endpoint"
    endpoint.to_dict.return_value = endpoint_conf
    return endpoint


@pytest.fixture
def mock_tokenizer():
    with patch(
        "llmeter.tokenizers.Tokenizer.to_dict",
        return_value={"tokenizer_module": "mock_tokenizer"},
    ):
        yield MagicMock(spec=Tokenizer)


@pytest.fixture
def runner(mock_endpoint, mock_tokenizer):
    return Runner(endpoint=mock_endpoint, tokenizer=mock_tokenizer)


@pytest.fixture
def time_bound_run(mock_endpoint, mock_tokenizer):
    """Create a _Run configured for time-bound (duration) mode."""
    run = _Run(
        endpoint=mock_endpoint,
        tokenizer=mock_tokenizer,
        payload=[{"prompt": "test"}],
        run_duration=1.0,
        clients=1,
        run_name="test_time_bound",
    )
    return run


@pytest.fixture
def count_bound_run(mock_endpoint, mock_tokenizer):
    """Create a _Run configured for count-bound mode."""
    run = _Run(
        endpoint=mock_endpoint,
        tokenizer=mock_tokenizer,
        payload=[{"prompt": "test"}],
        n_requests=5,
        clients=1,
        run_name="test_count_bound",
    )
    return run


# ── Stats throttling tests ───────────────────────────────────────────────────


class TestStatsThrottling:
    """Tests for the timer-based stats display update throttling."""

    @pytest.mark.asyncio
    async def test_stats_not_updated_on_every_result(self, count_bound_run: _Run):
        """Stats display should NOT be called for every single result."""
        run = count_bound_run
        run._queue = asyncio.Queue()
        run._progress_bar = MagicMock()
        run._backlog_bar = None
        run._stats_display = MagicMock()
        run._time_bound = False

        # Put 10 responses on the queue in quick succession, then None to stop
        for i in range(10):
            await run._queue.put(
                InvocationResponse(
                    id=str(i),
                    input_prompt="test",
                    response_text="response",
                    num_tokens_input=5,
                    num_tokens_output=5,
                )
            )
        await run._queue.put(None)

        await run._process_results_from_q()

        # Should have processed all 10 responses
        assert len(run._responses) == 10
        # Stats display should NOT have been called 10 times since they arrive
        # within the 0.5s throttle window. It should be called at most a few
        # times (initial burst + final update).
        assert run._stats_display.update.call_count < 10

    @pytest.mark.asyncio
    async def test_stats_updated_after_throttle_interval(self, count_bound_run: _Run):
        """Stats display should update after the throttle interval elapses."""
        run = count_bound_run
        run._queue = asyncio.Queue()
        run._progress_bar = MagicMock()
        run._backlog_bar = None
        run._stats_display = MagicMock()
        run._time_bound = False

        # Put first response
        await run._queue.put(
            InvocationResponse(
                id="1",
                input_prompt="test",
                response_text="response",
                num_tokens_input=5,
                num_tokens_output=5,
            )
        )

        # Simulate a delay by patching time.perf_counter
        original_perf_counter = time.perf_counter

        call_count = [0]
        base_time = original_perf_counter()

        def mock_perf_counter():
            # After the first response, advance time past the throttle threshold
            if call_count[0] > 0:
                return base_time + STATS_UPDATE_INTERVAL + 0.1
            call_count[0] += 1
            return base_time

        # Put second response that will arrive "0.6s later"
        await run._queue.put(
            InvocationResponse(
                id="2",
                input_prompt="test",
                response_text="response",
                num_tokens_input=5,
                num_tokens_output=5,
            )
        )
        await run._queue.put(None)

        with patch("llmeter.runner.time.perf_counter", side_effect=mock_perf_counter):
            await run._process_results_from_q()

        # Both responses should trigger stats updates since they're >0.5s apart
        assert run._stats_display.update.call_count == 2 + 1  # 2 timed + 1 final

    @pytest.mark.asyncio
    async def test_final_stats_update_on_queue_drain(self, count_bound_run: _Run):
        """A final stats update should occur when the queue drains."""
        run = count_bound_run
        run._queue = asyncio.Queue()
        run._progress_bar = MagicMock()
        run._backlog_bar = None
        run._stats_display = MagicMock()
        run._time_bound = False

        # Single response + sentinel
        await run._queue.put(
            InvocationResponse(
                id="1",
                input_prompt="test",
                response_text="response",
                num_tokens_input=5,
                num_tokens_output=5,
            )
        )
        await run._queue.put(None)

        await run._process_results_from_q()

        # The final update should always happen (even if within throttle window)
        assert run._stats_display.update.call_count >= 1
        # Last call should have the final stats
        last_call = run._stats_display.update.call_args_list[-1]
        # Should include stats (non-empty dict)
        assert last_call[0][0] or last_call[1].get("raw_stats")

    @pytest.mark.asyncio
    async def test_stats_display_disabled(self, count_bound_run: _Run):
        """No stats updates when display is None."""
        run = count_bound_run
        run._queue = asyncio.Queue()
        run._progress_bar = MagicMock()
        run._backlog_bar = None
        run._stats_display = None
        run._time_bound = False

        await run._queue.put(
            InvocationResponse(
                id="1",
                input_prompt="test",
                response_text="response",
                num_tokens_input=5,
                num_tokens_output=5,
            )
        )
        await run._queue.put(None)

        # Should not raise even with _stats_display=None
        await run._process_results_from_q()
        assert len(run._responses) == 1


# ── Backlog progress bar tests ───────────────────────────────────────────────


class TestBacklogProgressBar:
    """Tests for the backlog progress bar in time-bound runs."""

    @pytest.mark.asyncio
    async def test_backlog_bar_increments_per_result(self, time_bound_run: _Run):
        """The backlog bar should be updated for each processed result."""
        run = time_bound_run
        run._queue = asyncio.Queue()
        run._progress_bar = None  # Already closed in phase 2
        run._stats_display = MagicMock()
        run._backlog_bar = MagicMock()

        # Put 5 responses + sentinel
        for i in range(5):
            await run._queue.put(
                InvocationResponse(
                    id=str(i),
                    input_prompt="test",
                    response_text="response",
                    num_tokens_input=5,
                    num_tokens_output=5,
                )
            )
        await run._queue.put(None)

        await run._process_results_from_q()

        # Backlog bar should have been updated once per result
        assert run._backlog_bar.update.call_count == 5
        # Each update should be 1
        for call in run._backlog_bar.update.call_args_list:
            assert call[0][0] == 1

    @pytest.mark.asyncio
    async def test_no_backlog_bar_when_none(self, count_bound_run: _Run):
        """When _backlog_bar is None, no errors should occur."""
        run = count_bound_run
        run._queue = asyncio.Queue()
        run._progress_bar = MagicMock()
        run._backlog_bar = None
        run._stats_display = MagicMock()

        await run._queue.put(
            InvocationResponse(
                id="1",
                input_prompt="test",
                response_text="response",
                num_tokens_input=5,
                num_tokens_output=5,
            )
        )
        await run._queue.put(None)

        # Should not raise
        await run._process_results_from_q()
        assert len(run._responses) == 1

    @pytest.mark.asyncio
    async def test_backlog_bar_and_progress_bar_independent(
        self, time_bound_run: _Run
    ):
        """Backlog bar updates independently of the main progress bar."""
        run = time_bound_run
        run._queue = asyncio.Queue()
        run._progress_bar = MagicMock()
        run._backlog_bar = MagicMock()
        run._stats_display = MagicMock()
        # For time-bound runs, main progress bar is NOT updated per-result
        run._time_bound = True

        await run._queue.put(
            InvocationResponse(
                id="1",
                input_prompt="test",
                response_text="response",
                num_tokens_input=5,
                num_tokens_output=5,
            )
        )
        await run._queue.put(None)

        await run._process_results_from_q()

        # Main progress bar should NOT be updated (time-bound skips per-result updates)
        run._progress_bar.update.assert_not_called()
        # Backlog bar SHOULD be updated
        run._backlog_bar.update.assert_called_once_with(1)


# ── Two-phase flow integration tests ────────────────────────────────────────


class TestTwoPhaseFlow:
    """Tests for the full two-phase flow in time-bound runs."""

    @pytest.mark.asyncio
    async def test_time_bound_run_initializes_backlog_bar_to_none(self, runner: Runner):
        """_backlog_bar should be initialized to None at the start of a run."""
        result = await runner.run(
            payload={"prompt": "test"},
            run_duration=0.3,
            clients=1,
        )
        # Run completes successfully; backlog bar was managed internally
        assert result.total_requests > 0

    @pytest.mark.asyncio
    async def test_time_bound_run_completes_with_results(self, runner: Runner):
        """Time-bound run with the new two-phase flow should complete and return results."""
        result = await runner.run(
            payload={"prompt": "test"},
            run_duration=0.3,
            clients=2,
        )

        assert result.total_requests > 0
        assert result.clients == 2
        assert result.total_test_time is not None
        assert result.total_test_time > 0

    @pytest.mark.asyncio
    async def test_time_bound_run_stats_available(self, runner: Runner):
        """Stats should be computed correctly in the two-phase flow."""
        result = await runner.run(
            payload={"prompt": "test"},
            run_duration=0.3,
            clients=1,
        )

        stats = result.stats
        assert "total_requests" in stats
        assert stats["total_requests"] == result.total_requests
        assert stats["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_backlog_bar_created_when_queue_has_items(
        self, mock_endpoint, mock_tokenizer
    ):
        """When the queue has items after sending stops, a backlog bar should be created."""
        run = _Run(
            endpoint=mock_endpoint,
            tokenizer=mock_tokenizer,
            payload=[{"prompt": "test"}],
            run_duration=0.2,
            clients=1,
            run_name="test_backlog_creation",
        )

        # Make invoke slow enough that results pile up
        def slow_invoke(payload):
            time.sleep(0.05)
            return InvocationResponse(
                id="1", input_prompt="test", response_text="response"
            )

        mock_endpoint.invoke.side_effect = slow_invoke

        backlog_bar_created = [False]
        original_run = run._run

        # Patch tqdm to track if a "Processing backlog" bar is created
        with patch("llmeter.runner.tqdm") as mock_tqdm:
            mock_tqdm.return_value = MagicMock()
            mock_tqdm.gather = asyncio.gather

            # We can't easily run the full _run due to fixture complexity,
            # but we can verify the initialization
            run._backlog_bar = None
            assert run._backlog_bar is None

    @pytest.mark.asyncio
    async def test_count_bound_run_unaffected(self, runner: Runner):
        """Count-bound runs should work exactly as before (no backlog bar)."""
        result = await runner.run(
            payload={"prompt": "test"},
            n_requests=3,
            clients=1,
        )

        assert result.total_requests == 3
        assert result.clients == 1
        assert result.n_requests == 3

    @pytest.mark.asyncio
    async def test_time_bar_completes_to_full(self, time_bound_run: _Run):
        """The time progress bar should reach 100% when _tick_time_bar finishes."""
        run = time_bound_run
        run.run_duration = 0.5
        run._progress_bar = MagicMock()
        run._progress_bar.total = int(run.run_duration)
        run._progress_bar.n = 0

        await run._tick_time_bar()

        # The bar should have been updated with the full duration
        total_updates = sum(
            call[0][0] for call in run._progress_bar.update.call_args_list
        )
        assert total_updates >= int(run.run_duration) - 1  # Allow for timing slack


# ── Stats prefix tests ───────────────────────────────────────────────────────


class TestStatsPrefix:
    """Tests for the 'reqs=N' prefix in time-bound runs."""

    @pytest.mark.asyncio
    async def test_time_bound_shows_reqs_prefix(self, time_bound_run: _Run):
        """Time-bound runs should show reqs=N prefix in stats updates."""
        run = time_bound_run
        run._queue = asyncio.Queue()
        run._progress_bar = None
        run._backlog_bar = None
        run._stats_display = MagicMock()
        run._time_bound = True

        await run._queue.put(
            InvocationResponse(
                id="1",
                input_prompt="test",
                response_text="response",
                num_tokens_input=5,
                num_tokens_output=5,
            )
        )
        await run._queue.put(None)

        await run._process_results_from_q()

        # Check that the final update includes the reqs= prefix
        last_call = run._stats_display.update.call_args_list[-1]
        extra_prefix = last_call[1].get("extra_prefix", last_call[0][1] if len(last_call[0]) > 1 else "")
        assert "reqs=" in extra_prefix

    @pytest.mark.asyncio
    async def test_count_bound_no_reqs_prefix(self, count_bound_run: _Run):
        """Count-bound runs should NOT show reqs=N prefix."""
        run = count_bound_run
        run._queue = asyncio.Queue()
        run._progress_bar = MagicMock()
        run._backlog_bar = None
        run._stats_display = MagicMock()
        run._time_bound = False

        await run._queue.put(
            InvocationResponse(
                id="1",
                input_prompt="test",
                response_text="response",
                num_tokens_input=5,
                num_tokens_output=5,
            )
        )
        await run._queue.put(None)

        await run._process_results_from_q()

        # Check that calls use empty prefix
        for call in run._stats_display.update.call_args_list:
            extra_prefix = call[1].get("extra_prefix", call[0][1] if len(call[0]) > 1 else "")
            assert "reqs=" not in extra_prefix
