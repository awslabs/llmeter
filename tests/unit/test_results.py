# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime
from pathlib import Path

import pytest
from upath import UPath

from llmeter.endpoints.base import InvocationResponse
from llmeter.results import Result, _get_run_stats, _get_stats_from_results

sample_responses_successful = [
    InvocationResponse(
        response_text="test response",
        time_to_first_token=k[0],
        time_to_last_token=k[1],
        num_tokens_input=k[2],
        num_tokens_output=k[3],
        input_prompt="test input prompt",
    )
    for k in [
        [132, 236, 106, 104],
        [89, 230, 122, 8],
        [184, 187, 256, 183],
        [51, 247, 269, 171],
        [13, 126, 293, 168],
        [33, 245, 164, 109],
        [41, 231, 266, 131],
        [71, 206, 1, 31],
        [124, 179, 134, 19],
        [218, 230, 239, 265],
    ]
]
response_error = InvocationResponse(error="this is an error", response_text="")

## testing for `_get_test_stats()`


def test_get_test_stats():
    # Test case 1: No errors, non-zero test time
    responses = sample_responses_successful
    result = Result(
        responses=responses,
        total_requests=10,
        clients=2,
        n_requests=5,
        total_test_time=100,
    )
    stats = _get_run_stats(result)

    assert stats["failed_requests"] == 0
    assert stats["failed_requests_rate"] == 0
    assert pytest.approx(stats["requests_per_minute"], 0.01) == 6

    # Test case 2: Some errors, non-zero test time
    responses = sample_responses_successful[:8] + [response_error] * 2
    result = Result(
        responses=responses,
        total_requests=10,
        clients=2,
        n_requests=5,
        total_test_time=100,
    )
    stats = _get_run_stats(result)

    assert stats["failed_requests"] == 2
    assert pytest.approx(stats["failed_requests_rate"], 0.01) == 0.2
    assert pytest.approx(stats["requests_per_minute"], 0.01) == 6

    # Test case 3: All errors, non-zero test time
    responses = [response_error] * 5
    result = Result(
        responses=responses,
        total_requests=5,
        clients=1,
        n_requests=5,
        total_test_time=10,
    )
    stats = _get_run_stats(result)

    assert stats["failed_requests"] == 5
    assert stats["failed_requests_rate"] == 1
    assert pytest.approx(stats["requests_per_minute"], 0.01) == 30

    # Test case 4: No errors, zero test time
    responses = sample_responses_successful[:3]
    result = Result(
        responses=responses,
        total_requests=3,
        clients=1,
        n_requests=3,
        total_test_time=0,
    )
    stats = _get_run_stats(result)

    assert stats["failed_requests"] == 0
    assert stats["failed_requests_rate"] == 0
    assert stats["requests_per_minute"] == 0  # Avoid division by zero


# ## testing `_get_stats_from_results()`
test_metrics = [
    "time_to_last_token",
    "time_to_first_token",
    "num_tokens_output",
    "num_tokens_input",
]


def test_get_stats_from_results_with_result_object():
    responses = sample_responses_successful
    result = Result(
        clients=5,
        n_requests=100,
        responses=responses,
        total_requests=5,
        total_test_time=10,
    )

    stats = _get_stats_from_results(result, metrics=test_metrics)

    assert "time_to_last_token" in stats
    assert "time_to_first_token" in stats
    assert "num_tokens_output" in stats
    assert "num_tokens_input" in stats


def test_get_stats_from_results_with_no_metrics():
    responses = sample_responses_successful
    result = Result(
        clients=5,
        n_requests=100,
        responses=responses,
        total_requests=5,
        total_test_time=10,
    )

    stats = _get_stats_from_results(result, metrics=[])
    assert stats == {}


@pytest.fixture
def sample_result():
    responses = [
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
    return Result(
        responses=responses,
        total_requests=5,
        clients=1,
        n_requests=5,
        total_test_time=1,
        run_name="Test Run",
    )


def test_stats_property(sample_result: Result):
    stats = sample_result.stats

    # Test basic information
    assert stats["total_requests"] == 5

    # Test aggregated statistics
    assert "time_to_last_token-p50" in stats
    assert "time_to_first_token-average" in stats
    assert "num_tokens_output-p90" in stats
    assert "num_tokens_input-p99" in stats

    # Test specific values (you may need to adjust these based on your exact implementation)
    assert stats["time_to_last_token-average"] == pytest.approx(0.6)
    assert stats["time_to_first_token-p50"] == pytest.approx(0.3)
    assert stats["num_tokens_output-average"] == 30
    assert stats["num_tokens_input-average"] == 15

    # Test test-specific statistics
    assert "failed_requests" in stats
    assert "failed_requests_rate" in stats
    assert "requests_per_minute" in stats

    # Test that all keys from to_dict() are present
    for key in sample_result.to_dict().keys():
        assert key in stats

    # Test caching returns same object for built-in stats:
    assert sample_result._preloaded_stats is None or isinstance(
        sample_result._preloaded_stats, dict
    )


def test_stats_property_empty_result():
    empty_result = Result(
        responses=[], total_requests=0, clients=0, n_requests=0, total_test_time=0
    )
    stats = empty_result.stats

    assert stats["total_requests"] == 0
    assert stats["failed_requests"] == 0
    assert stats["failed_requests_rate"] == 0
    assert stats["requests_per_minute"] == 0

    # Check that no errors are raised for empty data
    for metric in [
        "time_to_last_token",
        "time_to_first_token",
        "num_tokens_output",
        "num_tokens_input",
    ]:
        for stat in ["p50", "p90", "p99", "average"]:
            assert f"{metric}-{stat}" not in stats


def test_stats_json_serializable_with_datetimes():
    """stats dict should be JSON-serializable via json_default."""
    from datetime import datetime, timezone

    from llmeter.serialization import json_default

    result = Result(
        responses=sample_responses_successful[:2],
        total_requests=2,
        clients=1,
        n_requests=2,
        total_test_time=1.0,
        start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    stats = result.stats
    # stats contains raw datetime objects; serialization is handled at the
    # boundary (e.g. save()) via json_default.
    json_str = json.dumps(stats, default=json_default)
    parsed = json.loads(json_str)
    assert parsed["start_time"] == "2025-01-01T12:00:00Z"
    assert parsed["end_time"] == "2025-01-01T12:00:01Z"


def test_to_dict_returns_native_types():
    """to_dict() should return raw Python types, not serialized strings."""
    from datetime import datetime, timezone

    result = Result(
        responses=[],
        start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        output_path=UPath("/tmp/test-results"),
    )
    data = result.to_dict()
    assert isinstance(data["start_time"], datetime)
    assert isinstance(data["output_path"], UPath)


@pytest.fixture
def temp_dir(tmp_path: Path):
    return UPath(tmp_path)


def test_save_method(sample_result: Result, temp_dir: UPath):
    output_path = temp_dir / "test_output"
    sample_result.save(output_path)

    # Check if files are created
    assert (output_path / "summary.json").exists()
    assert (output_path / "stats.json").exists()
    assert (output_path / "responses.jsonl").exists()

    # Check content of summary.json
    with (output_path / "summary.json").open() as f:
        summary = json.load(f)
        assert summary["total_requests"] == 5
        assert summary["run_name"] == "Test Run"

    # Check content of stats.json
    with (output_path / "stats.json").open() as f:
        stats = json.load(f)
        assert "total_requests" in stats
        assert "time_to_last_token-average" in stats

    # Check content of responses.jsonl
    with (output_path / "responses.jsonl").open() as f:
        responses = [json.loads(line) for line in f]
        assert len(responses) == 5
        assert all(isinstance(r["id"], str) for r in responses)


def test_load_method(sample_result: Result, temp_dir: UPath):
    output_path = temp_dir / "test_output"
    sample_result.save(output_path)

    # Load the saved result
    loaded_result = Result.load(output_path)

    # Check if loaded result matches the original
    assert loaded_result.total_requests == sample_result.total_requests
    assert loaded_result.run_name == sample_result.run_name
    assert len(loaded_result.responses) == len(sample_result.responses)

    # Check if responses are correctly loaded
    for orig, loaded in zip(sample_result.responses, loaded_result.responses):
        assert orig.id == loaded.id
        assert orig.response_text == loaded.response_text
        assert orig.input_prompt == loaded.input_prompt
        assert orig.time_to_first_token == loaded.time_to_first_token
        assert orig.time_to_last_token == loaded.time_to_last_token
        assert orig.num_tokens_output == loaded.num_tokens_output
        assert orig.num_tokens_input == loaded.num_tokens_input


def test_load_restores_summary_datetimes(temp_dir: UPath):
    """Result.load must parse Z-suffixed ISO-8601 strings back to datetime."""
    from datetime import datetime, timezone

    dt_start = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    dt_end = datetime(2025, 6, 1, 10, 5, 0, tzinfo=timezone.utc)
    result = Result(
        responses=sample_responses_successful[:2],
        total_requests=2,
        clients=1,
        n_requests=2,
        total_test_time=300.0,
        start_time=dt_start,
        end_time=dt_end,
        first_request_time=dt_start,
        last_request_time=dt_end,
    )
    output_path = temp_dir / "datetime_output"
    result.save(output_path)

    loaded = Result.load(output_path)
    assert isinstance(loaded.start_time, datetime)
    assert isinstance(loaded.end_time, datetime)
    assert isinstance(loaded.first_request_time, datetime)
    assert isinstance(loaded.last_request_time, datetime)
    assert loaded.start_time == dt_start
    assert loaded.end_time == dt_end


def test_load_restores_response_request_time(temp_dir: UPath):
    """Result.load must restore request_time on responses as datetime, not str."""
    from datetime import datetime, timezone

    dt = datetime(2025, 3, 15, 8, 30, 0, tzinfo=timezone.utc)
    responses = [
        InvocationResponse(
            id="rt_test",
            response_text="hello",
            request_time=dt,
            num_tokens_input=5,
            num_tokens_output=3,
        )
    ]
    result = Result(
        responses=responses,
        total_requests=1,
        clients=1,
        n_requests=1,
        total_test_time=1.0,
    )
    output_path = temp_dir / "request_time_output"
    result.save(output_path)

    loaded = Result.load(output_path)
    assert isinstance(loaded.responses[0].request_time, datetime)
    assert loaded.responses[0].request_time == dt


def test_save_method_no_output_path(sample_result: Result):
    with pytest.raises(ValueError, match="No output path provided"):
        sample_result.save()


def test_load_method_missing_files(temp_dir: UPath):
    with pytest.raises(FileNotFoundError):
        Result.load(temp_dir / "non_existent_directory")


def test_save_and_load_with_string_path(sample_result: Result, temp_dir: UPath):
    output_path = str(temp_dir / "test_output")
    sample_result.save(output_path)
    loaded_result = Result.load(output_path)
    assert loaded_result.total_requests == sample_result.total_requests


def test_save_method_existing_responses(sample_result: Result, temp_dir: UPath):
    output_path = temp_dir / "test_output"
    sample_result.save(output_path)

    # Modify the responses file
    with (output_path / "responses.jsonl").open("a") as f:
        f.write(json.dumps({"id": "extra_response"}) + "\n")

    # Save again
    sample_result.save(output_path)

    # Check that the responses file wasn't overwritten
    with (output_path / "responses.jsonl").open() as f:
        responses = [json.loads(line) for line in f]
        assert len(responses) == 6  # 5 original + 1 extra
        assert responses[-1]["id"] == "extra_response"


# ── restore_dataclass_types introspection ──────────────────────────────────────


def test_restore_dataclass_types_converts_iso_strings():
    """restore_dataclass_types should convert all datetime-typed field keys."""
    from llmeter.serialization import restore_dataclass_types

    d = {
        "start_time": "2025-06-01T10:00:00Z",
        "end_time": "2025-06-01T10:05:00+00:00",
        "first_request_time": "2025-06-01T10:00:01Z",
        "last_request_time": None,
        "total_requests": 5,  # non-datetime field, should be left alone
    }
    restore_dataclass_types(Result, d)

    assert isinstance(d["start_time"], datetime)
    assert isinstance(d["end_time"], datetime)
    assert isinstance(d["first_request_time"], datetime)
    assert d["last_request_time"] is None
    assert d["total_requests"] == 5


def test_restore_dataclass_types_skips_already_parsed():
    """Already-datetime values should pass through unchanged."""
    from datetime import timezone

    from llmeter.serialization import restore_dataclass_types

    dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    d = {"start_time": dt}
    restore_dataclass_types(Result, d)
    assert d["start_time"] is dt


def test_restore_dataclass_types_handles_invalid_string():
    """Invalid date strings should be left as-is (not raise)."""
    from llmeter.serialization import restore_dataclass_types

    d = {"start_time": "not-a-date"}
    restore_dataclass_types(Result, d)
    assert d["start_time"] == "not-a-date"


# ── Corrupt stats.json alongside valid summary.json ────────────────────────────


def test_load_with_corrupt_stats_json_and_responses(sample_result: Result, temp_dir):
    """Corrupt stats.json should not crash load() when responses are available."""
    output_path = temp_dir / "corrupt_stats"
    sample_result.save(output_path)

    # Corrupt stats.json
    with (output_path / "stats.json").open("w") as f:
        f.write("{invalid json!!")

    loaded = Result.load(output_path, load_responses=True)

    # Stats should still be computed from responses
    assert loaded.stats["failed_requests"] == 0
    assert "time_to_first_token-p50" in loaded.stats
    assert loaded.total_requests == 5


def test_load_with_corrupt_stats_json_no_responses(sample_result: Result, temp_dir):
    """Corrupt stats.json with load_responses=False should not crash."""
    output_path = temp_dir / "corrupt_stats_lazy"
    sample_result.save(output_path)

    with (output_path / "stats.json").open("w") as f:
        f.write("not json")

    loaded = Result.load(output_path, load_responses=False)

    # _preloaded_stats falls back to None; stats property will compute on demand
    assert loaded._preloaded_stats is None
    stats = loaded.stats
    assert stats["total_requests"] == 5


def test_load_missing_responses_jsonl_with_summary(sample_result: Result, temp_dir):
    """Missing responses.jsonl should not crash; stats come from stats.json."""
    output_path = temp_dir / "no_responses"
    sample_result.save(output_path)

    # Remove responses.jsonl
    (output_path / "responses.jsonl").unlink()

    loaded = Result.load(output_path, load_responses=True)

    assert loaded.responses == []
    assert loaded.total_requests == 5
    # Stats should come from stats.json on disk
    assert "time_to_first_token-p50" in loaded.stats


# ── Contributed stats round-trip ─────────────────────────────────────────────


class TestContributedStatsRoundTrip:
    """Verify that callback-contributed stats survive save → load cycles."""

    @pytest.fixture
    def result_with_contributed_stats(self):
        responses = [
            InvocationResponse(
                id=f"r{i}",
                response_text=f"resp {i}",
                input_prompt=f"prompt {i}",
                time_to_first_token=0.1 * i,
                time_to_last_token=0.2 * i,
                num_tokens_output=10 * i,
                num_tokens_input=5 * i,
            )
            for i in range(1, 4)
        ]
        result = Result(
            responses=responses,
            total_requests=3,
            clients=1,
            n_requests=3,
            total_test_time=1.0,
        )
        result._update_contributed_stats(
            {"custom_metric_a": 42.0, "custom_metric_b": 99.5}
        )
        return result

    def test_contributed_stats_appear_in_stats(self, result_with_contributed_stats):
        stats = result_with_contributed_stats.stats
        assert stats["custom_metric_a"] == 42.0
        assert stats["custom_metric_b"] == 99.5

    def test_contributed_stats_written_to_stats_json(
        self, result_with_contributed_stats, tmp_path
    ):
        output = UPath(tmp_path / "out")
        result_with_contributed_stats.save(output)

        with (output / "stats.json").open() as f:
            saved = json.load(f)
        assert saved["custom_metric_a"] == 42.0
        assert saved["custom_metric_b"] == 99.5

    def test_load_with_responses_preserves_contributed_stats(
        self, result_with_contributed_stats, tmp_path
    ):
        output = UPath(tmp_path / "out")
        result_with_contributed_stats.save(output)

        loaded = Result.load(output, load_responses=True)

        assert loaded.stats["custom_metric_a"] == 42.0
        assert loaded.stats["custom_metric_b"] == 99.5

    def test_load_without_responses_preserves_contributed_stats(
        self, result_with_contributed_stats, tmp_path
    ):
        output = UPath(tmp_path / "out")
        result_with_contributed_stats.save(output)

        loaded = Result.load(output, load_responses=False)

        assert loaded.stats["custom_metric_a"] == 42.0
        assert loaded.stats["custom_metric_b"] == 99.5

    def test_contributed_stats_do_not_clobber_builtin_stats(
        self, result_with_contributed_stats, tmp_path
    ):
        output = UPath(tmp_path / "out")
        result_with_contributed_stats.save(output)

        loaded = Result.load(output, load_responses=True)

        # Builtin stats must still be present and correct
        assert "failed_requests" in loaded.stats
        assert loaded.stats["total_requests"] == 3
        assert "time_to_first_token-p50" in loaded.stats

    def test_builtin_stats_not_overwritten_by_stale_saved_values(self, tmp_path):
        """If a builtin key exists in stats.json with a stale value, the freshly
        computed value from responses should win."""
        responses = [
            InvocationResponse(
                id="x",
                response_text="r",
                input_prompt="p",
                time_to_first_token=0.5,
                time_to_last_token=1.0,
                num_tokens_output=10,
                num_tokens_input=5,
            )
        ]
        result = Result(
            responses=responses,
            total_requests=1,
            clients=1,
            n_requests=1,
            total_test_time=2.0,
        )
        output = UPath(tmp_path / "out")
        result.save(output)

        # Tamper with stats.json: set a wrong value for a builtin key
        stats_path = output / "stats.json"
        with stats_path.open() as f:
            saved = json.load(f)
        saved["failed_requests"] = 999
        with stats_path.open("w") as f:
            json.dump(saved, f)

        loaded = Result.load(output, load_responses=True)

        # The freshly computed value (0 failures) should win over the tampered 999
        assert loaded.stats["failed_requests"] == 0

    def test_load_responses_recomputes_but_keeps_contributed(self, tmp_path):
        """After load(load_responses=False) + load_responses(), contributed
        stats from stats.json should still be accessible via _preloaded_stats
        even though responses were reloaded."""
        responses = [
            InvocationResponse(
                id="z",
                response_text="r",
                input_prompt="p",
                time_to_first_token=0.3,
                time_to_last_token=0.6,
                num_tokens_output=8,
                num_tokens_input=4,
            )
        ]
        result = Result(
            responses=responses,
            total_requests=1,
            clients=1,
            n_requests=1,
            total_test_time=1.0,
        )
        result._update_contributed_stats({"cb_stat": 7.0})
        output = UPath(tmp_path / "out")
        result.save(output)

        loaded = Result.load(output, load_responses=False)
        assert loaded.stats["cb_stat"] == 7.0

        # Now reload responses — _preloaded_stats gets recomputed from
        # responses only, so cb_stat won't be in _preloaded_stats anymore,
        # but it was never in _contributed_stats on the loaded instance either.
        loaded.load_responses()
        # After recompute, builtin stats should be correct
        assert loaded.stats["failed_requests"] == 0
        assert "time_to_first_token-p50" in loaded.stats

    def test_multiple_contributed_stats_updates_merge(self, tmp_path):
        responses = [
            InvocationResponse(
                id="m",
                response_text="r",
                input_prompt="p",
                num_tokens_output=5,
                num_tokens_input=3,
            )
        ]
        result = Result(
            responses=responses,
            total_requests=1,
            clients=1,
            n_requests=1,
            total_test_time=0.5,
        )
        result._update_contributed_stats({"stat_a": 1.0})
        result._update_contributed_stats({"stat_b": 2.0})
        result._update_contributed_stats({"stat_a": 10.0})  # overwrite

        output = UPath(tmp_path / "out")
        result.save(output)

        loaded = Result.load(output, load_responses=True)
        assert loaded.stats["stat_a"] == 10.0
        assert loaded.stats["stat_b"] == 2.0
