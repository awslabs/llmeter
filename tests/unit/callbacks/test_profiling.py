# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import pickle
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llmeter.callbacks.profiling import (
    ProfileCallback,
)
from llmeter.endpoints.base import InvocationResponse
from llmeter.results import Result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeRunConfig:
    pass


def _make_response(
    *,
    idx: int = 0,
    ttft: float = 0.25,
    ttlt: float = 3.0,
    tokens_in: int = 100,
    tokens_out: int = 80,
    cached: int | None = None,
    reasoning: int | None = None,
    error: str | None = None,
    retries: int | None = None,
) -> InvocationResponse:
    return InvocationResponse(
        response_text="output" if not error else None,
        id=f"req-{idx:03d}",
        request_time=datetime(2025, 6, 19, 10, 0, idx, tzinfo=timezone.utc),
        time_to_first_token=ttft if not error else None,
        time_to_last_token=ttlt if not error else None,
        num_tokens_input=tokens_in if not error else None,
        num_tokens_output=tokens_out if not error else None,
        num_tokens_input_cached=cached,
        num_tokens_output_reasoning=reasoning,
        error=error,
        retries=retries,
    )


@pytest.fixture
def profiler():
    return ProfileCallback(print_report=False, save_report=False)


@pytest.fixture
def profiler_with_data():
    """Returns a profiler that has run through a simulated run."""

    async def _setup():
        cb = ProfileCallback(print_report=False, save_report=True)
        await cb.before_run(FakeRunConfig())

        responses = [
            _make_response(idx=0, ttft=0.2, ttlt=2.5, tokens_in=100, tokens_out=80),
            _make_response(
                idx=1, ttft=0.08, ttlt=2.0, tokens_in=100, tokens_out=60, cached=70
            ),
            _make_response(
                idx=2,
                ttft=0.5,
                ttlt=6.0,
                tokens_in=200,
                tokens_out=300,
                reasoning=150,
            ),
            _make_response(
                idx=3, ttft=0.9, ttlt=4.0, tokens_in=80, tokens_out=50, retries=2
            ),
            _make_response(idx=4, error="ThrottlingException", retries=3),
        ]

        for resp in responses:
            await cb.before_invoke({"messages": [{"role": "user", "content": "test"}]})
            await cb.after_invoke(resp)

        result = Result(
            responses=[],
            total_requests=5,
            clients=1,
            n_requests=5,
            total_test_time=20.0,
        )
        await cb.after_run(result)
        return cb, result

    return asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Serialization / deepcopy tests
# ---------------------------------------------------------------------------


class TestProfileCallbackSerialization:
    def test_deepcopy(self, profiler):
        copied = copy.deepcopy(profiler)
        assert copied.print_report == profiler.print_report
        assert copied.save_report == profiler.save_report

    def test_pickle_roundtrip(self, profiler):
        data = pickle.dumps(profiler)
        restored = pickle.loads(data)
        assert restored.print_report is False
        assert restored.save_report is False

    def test_asdict(self, profiler):
        d = asdict(profiler)
        assert d == {"print_report": False, "save_report": False}

    def test_save_and_load_config(self, profiler):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiler.json"
            profiler.save_to_file(path)

            with open(path) as f:
                config = json.load(f)
            assert config["type"] == "ProfileCallback"
            assert config["print_report"] is False
            assert config["save_report"] is False

            restored = ProfileCallback._load_from_file(path)
            assert restored.print_report is False
            assert restored.save_report is False


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestProfileCallbackLifecycle:
    def test_before_run_resets_state(self, profiler):
        async def _test():
            await profiler.before_run(FakeRunConfig())
            assert profiler._invocation_profiles == []
            assert profiler._sequence == 0
            assert profiler._report is None
            assert profiler._run_start_time > 0

        asyncio.run(_test())

    def test_captures_invocation_data(self, profiler):
        async def _test():
            await profiler.before_run(FakeRunConfig())
            await profiler.before_invoke({"messages": []})
            await profiler.after_invoke(
                _make_response(idx=0, ttft=0.3, ttlt=2.5, tokens_in=100, tokens_out=80)
            )

            assert len(profiler.invocation_profiles) == 1
            p = profiler.invocation_profiles[0]
            assert p.sequence == 0
            assert p.request_id == "req-000"
            assert p.ttft == 0.3
            assert p.ttlt == 2.5
            assert p.generation_time == pytest.approx(2.2)
            assert p.tokens_input == 100
            assert p.tokens_output == 80
            assert p.error is None
            assert p.callback_overhead >= 0

        asyncio.run(_test())

    def test_captures_cache_info(self, profiler):
        async def _test():
            await profiler.before_run(FakeRunConfig())
            await profiler.before_invoke({"messages": []})
            await profiler.after_invoke(
                _make_response(idx=0, cached=60, tokens_in=100, tokens_out=50)
            )

            p = profiler.invocation_profiles[0]
            assert p.cache_hit is True
            assert p.tokens_input_cached == 60
            assert p.cache_hit_ratio == pytest.approx(0.6)

        asyncio.run(_test())

    def test_captures_reasoning_tokens(self, profiler):
        async def _test():
            await profiler.before_run(FakeRunConfig())
            await profiler.before_invoke({"messages": []})
            await profiler.after_invoke(
                _make_response(idx=0, tokens_out=200, reasoning=120)
            )

            p = profiler.invocation_profiles[0]
            assert p.tokens_output_reasoning == 120

        asyncio.run(_test())

    def test_captures_errors_and_retries(self, profiler):
        async def _test():
            await profiler.before_run(FakeRunConfig())
            await profiler.before_invoke({"messages": []})
            await profiler.after_invoke(
                _make_response(idx=0, error="timeout", retries=3)
            )

            p = profiler.invocation_profiles[0]
            assert p.error == "timeout"
            assert p.retries == 3
            assert p.ttft is None
            assert p.generation_time is None

        asyncio.run(_test())

    def test_sequence_increments(self, profiler):
        async def _test():
            await profiler.before_run(FakeRunConfig())
            for i in range(5):
                await profiler.before_invoke({"messages": []})
                await profiler.after_invoke(_make_response(idx=i))

            sequences = [p.sequence for p in profiler.invocation_profiles]
            assert sequences == [0, 1, 2, 3, 4]

        asyncio.run(_test())

    def test_reuse_across_runs(self, profiler):
        """A profiler should reset between runs."""

        async def _test():
            await profiler.before_run(FakeRunConfig())
            await profiler.before_invoke({"messages": []})
            await profiler.after_invoke(_make_response(idx=0))
            result1 = Result(responses=[], total_requests=1)
            await profiler.after_run(result1)

            assert len(profiler.invocation_profiles) == 1

            # Second run
            await profiler.before_run(FakeRunConfig())
            assert len(profiler.invocation_profiles) == 0
            assert profiler.report is None

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Report computation tests
# ---------------------------------------------------------------------------


class TestProfileReport:
    def test_report_computed(self, profiler_with_data):
        cb, result = profiler_with_data
        report = cb.report
        assert report is not None
        assert report.total_requests == 5
        assert report.successful_requests == 4
        assert report.failed_requests == 1

    def test_report_cache_stats(self, profiler_with_data):
        cb, _ = profiler_with_data
        report = cb.report
        assert report.cache_hits == 1
        assert report.cache_hit_rate == pytest.approx(0.25)
        assert report.total_tokens_cached == 70

    def test_report_retry_stats(self, profiler_with_data):
        cb, _ = profiler_with_data
        report = cb.report
        assert report.retried_requests == 2  # idx=3 and idx=4
        assert report.total_retries == 5  # 2 + 3

    def test_report_token_totals(self, profiler_with_data):
        cb, _ = profiler_with_data
        report = cb.report
        assert report.total_tokens_input == 480  # 100+100+200+80
        assert report.total_tokens_output == 490  # 80+60+300+50
        assert report.total_tokens_reasoning == 150

    def test_report_timing_stats(self, profiler_with_data):
        cb, _ = profiler_with_data
        report = cb.report
        assert "average" in report.ttft_stats
        assert "p50" in report.ttft_stats
        assert "average" in report.generation_time_stats
        assert "average" in report.tpot_stats
        assert "average" in report.tokens_per_second_stats

    def test_report_str(self, profiler_with_data):
        cb, _ = profiler_with_data
        text = str(cb.report)
        assert "Profile Report" in text
        assert "4 ok" in text
        assert "1 failed" in text
        assert "Cache hits" in text
        assert "Reasoning tokens" in text
        assert "Retries" in text

    def test_report_repr(self, profiler_with_data):
        cb, _ = profiler_with_data
        text = repr(cb.report)
        assert "ProfileReport" in text


# ---------------------------------------------------------------------------
# Stats contribution tests
# ---------------------------------------------------------------------------


class TestProfileStatsContribution:
    def test_contributes_to_result_stats(self, profiler_with_data):
        _, result = profiler_with_data
        stats = result.stats
        assert "profile_total_wall_clock" in stats
        assert "profile_total_api_time" in stats
        assert "profile_runner_overhead" in stats
        assert "profile_api_time_fraction" in stats
        assert "profile_successful_requests" in stats
        assert stats["profile_successful_requests"] == 4
        assert stats["profile_failed_requests"] == 1

    def test_contributes_cache_stats(self, profiler_with_data):
        _, result = profiler_with_data
        assert result.stats["profile_cache_hit_rate"] == pytest.approx(0.25)
        assert result.stats["profile_total_tokens_cached"] == 70

    def test_contributes_timing_stats(self, profiler_with_data):
        _, result = profiler_with_data
        assert "profile_ttft-average" in result.stats
        assert "profile_ttft-p50" in result.stats
        assert "profile_generation_time-average" in result.stats
        assert "profile_tpot-average" in result.stats
        assert "profile_tokens_per_second-average" in result.stats


# ---------------------------------------------------------------------------
# File saving tests
# ---------------------------------------------------------------------------


class TestProfileSaving:
    def test_saves_report_json(self, profiler_with_data):
        cb, _ = profiler_with_data
        with tempfile.TemporaryDirectory() as tmpdir:
            cb._save_report(tmpdir)
            report_path = Path(tmpdir) / "profile_report.json"
            assert report_path.exists()

            with open(report_path) as f:
                data = json.load(f)
            assert data["successful_requests"] == 4
            assert data["failed_requests"] == 1
            assert "ttft_stats" in data

    def test_saves_invocations_jsonl(self, profiler_with_data):
        cb, _ = profiler_with_data
        with tempfile.TemporaryDirectory() as tmpdir:
            cb._save_invocations(tmpdir)
            inv_path = Path(tmpdir) / "profile_invocations.jsonl"
            assert inv_path.exists()

            with open(inv_path) as f:
                records = [json.loads(line) for line in f]
            assert len(records) == 5
            assert records[0]["sequence"] == 0
            assert records[0]["request_id"] == "req-000"
            assert records[1]["cache_hit"] is True
            assert records[4]["error"] == "ThrottlingException"

    def test_save_report_false_skips(self):
        async def _test():
            cb = ProfileCallback(print_report=False, save_report=False)
            await cb.before_run(FakeRunConfig())
            await cb.before_invoke({"messages": []})
            await cb.after_invoke(_make_response(idx=0))

            with tempfile.TemporaryDirectory() as tmpdir:
                result = Result(responses=[], total_requests=1, output_path=tmpdir)
                await cb.after_run(result)
                assert not (Path(tmpdir) / "profile_report.json").exists()
                assert not (Path(tmpdir) / "profile_invocations.jsonl").exists()

        asyncio.run(_test())

    def test_no_output_path_no_error(self):
        async def _test():
            cb = ProfileCallback(print_report=False, save_report=True)
            await cb.before_run(FakeRunConfig())
            await cb.before_invoke({"messages": []})
            await cb.after_invoke(_make_response(idx=0))

            result = Result(responses=[], total_requests=1, output_path=None)
            await cb.after_run(result)  # Should not raise

        asyncio.run(_test())

    def test_stats_persist_save_load(self, profiler_with_data):
        _, result = profiler_with_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result.output_path = tmpdir
            result.save()

            loaded = Result.load(tmpdir, load_responses=False)
            assert "profile_total_wall_clock" in loaded.stats
            assert "profile_ttft-average" in loaded.stats
            assert loaded.stats["profile_cache_hit_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestProfileEdgeCases:
    def test_all_requests_fail(self, profiler):
        async def _test():
            await profiler.before_run(FakeRunConfig())
            for i in range(3):
                await profiler.before_invoke({"messages": []})
                await profiler.after_invoke(_make_response(idx=i, error="fail"))

            result = Result(responses=[], total_requests=3)
            await profiler.after_run(result)

            assert profiler.report.successful_requests == 0
            assert profiler.report.failed_requests == 3
            assert profiler.report.ttft_stats == {}

        asyncio.run(_test())

    def test_no_requests(self, profiler):
        async def _test():
            await profiler.before_run(FakeRunConfig())
            result = Result(responses=[], total_requests=0)
            await profiler.after_run(result)

            assert profiler.report.total_requests == 0
            assert profiler.report.successful_requests == 0

        asyncio.run(_test())

    def test_missing_ttft(self, profiler):
        """Requests with None TTFT should not break computation."""

        async def _test():
            await profiler.before_run(FakeRunConfig())
            await profiler.before_invoke({"messages": []})
            resp = InvocationResponse(
                response_text="ok",
                time_to_first_token=None,
                time_to_last_token=2.0,
                num_tokens_input=50,
                num_tokens_output=30,
                error=None,
            )
            await profiler.after_invoke(resp)

            result = Result(responses=[], total_requests=1)
            await profiler.after_run(result)

            # Should not crash, TTFT stats should be empty
            assert profiler.report.ttft_stats == {}

        asyncio.run(_test())
