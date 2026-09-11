# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests that load *static, synthetic* result snapshots from disk.

Unlike the other load tests (which build data in-memory at test time), these load committed
fixture files under ``tests/unit/fixtures/result_snapshots/``. They pin the on-disk de/serialization
contract across the whole loading pipeline - ``Result.load``, ``_RunConfig.load``,
``LoadTestResult.load`` and the endpoint / tokenizer / callback restoration underneath - so format
drift can't silently break loading of previously-saved runs.

The fixtures are deliberately synthetic (fake model IDs like ``synthetic.test-model-v1``,
lorem-ipsum responses, timestamps in the past) so nobody mistakes them for real benchmark data.
They cover a base "happy path" modern-format snapshot plus variations: legacy formats, an
interrupted run, error/annotation edge cases, and a multi-concurrency load test.

To regenerate the fixtures (e.g. after an intentional format change), run:
    uv run python tests/unit/fixtures/_generate_fixtures.py
"""

from datetime import datetime, timezone
import json
import warnings

import httpx
import pytest

from llmeter.callbacks.cost.dimensions import InputTokens, OutputTokens
from llmeter.callbacks.cost.model import CostModel
from llmeter.callbacks.mlflow import MlflowCallback
from llmeter.endpoints.bedrock import BedrockConverse, BedrockConverseStream
from llmeter.endpoints.openai import OpenAICompletionStreamEndpoint
from llmeter.experiments import LoadTestResult
from llmeter.results import Result
from llmeter.runner import _RunConfig
from llmeter.serialization import load_object
from llmeter.tokenizers import DummyTokenizer
from llmeter.warnings import LegacyResultFormatWarning


class TestBaseLoad:
    """The modern-format baseline: OpenAI endpoint + CostModel + MlflowCallback."""

    @pytest.fixture
    def base_dir(self, snapshots_dir):
        return snapshots_dir / "base"

    def test_result_loads(self, base_dir):
        result = Result.load(base_dir)
        assert len(result.responses) == 5
        assert result.total_requests == 5
        assert result.clients == 2
        assert result.n_requests == 3
        assert result.model_id == "synthetic.test-model-v1"
        assert result.run_name == "synthetic-base-test"

    def test_null_output_path_filled_from_load_path(self, base_dir):
        """A null output_path in summary.json must be filled with the real load path.

        This is what makes a relocated/copied result directory reloadable - the on-disk
        path won't match wherever the data now lives.
        """
        result = Result.load(base_dir)
        assert result.output_path is not None
        assert str(base_dir) in str(result.output_path)

    def test_timestamps_parsed_as_datetime(self, base_dir):
        result = Result.load(base_dir)
        assert result.start_time == datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert result.end_time == datetime(2025, 1, 15, 10, 0, 4, tzinfo=timezone.utc)
        assert isinstance(result.responses[0].request_time, datetime)

    def test_stats_match_frozen_values(self, base_dir):
        """Stats loaded from stats.json (no responses) should match the frozen values."""
        result = Result.load(base_dir, load_responses=False)
        stats = result.stats
        assert stats["total_requests"] == 5
        assert stats["failed_requests"] == 0
        assert stats["total_input_tokens"] == 185
        assert stats["total_output_tokens"] == 386
        assert stats["time_to_first_token-p50"] == pytest.approx(0.415)
        assert stats["requests_per_minute"] == pytest.approx(66.35700066357)

    def test_response_optional_token_fields(self, base_dir):
        result = Result.load(base_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-base-002"].num_tokens_input_cached == 12
        assert by_id["synth-base-003"].num_tokens_output_reasoning == 8

    # --- run_config.json / endpoint / tokenizer / callback restoration ---

    @pytest.fixture
    def base_config(self, base_dir):
        return _RunConfig.load(base_dir)

    def test_endpoint_restored(self, base_config):
        ep = base_config._endpoint
        assert isinstance(ep, OpenAICompletionStreamEndpoint)
        assert ep.model_id == "synthetic.test-model-v1"
        assert ep.endpoint_name == "synthetic-openai-stream"
        assert ep.provider == "openai"

    def test_endpoint_client_config_restored(self, base_config):
        """The OpenAI client's config surface (base_url, retries, timeout) must round-trip."""
        client = base_config._endpoint._client
        assert str(client.base_url) == "https://api.synthetic-provider.example.com/v1/"
        assert client.max_retries == 3
        assert isinstance(client.timeout, httpx.Timeout)
        assert client.timeout.connect == 5.0
        assert client.timeout.read == 60.0
        assert client.timeout.write == 5.0
        assert client.timeout.pool == 5.0

    def test_tokenizer_restored(self, base_config):
        assert isinstance(base_config._tokenizer, DummyTokenizer)

    def test_cost_model_callback_restored(self, base_config):
        cost_model = base_config.callbacks[0]
        assert isinstance(cost_model, CostModel)

        input_dim = cost_model.request_dims["InputTokens"]
        output_dim = cost_model.request_dims["OutputTokens"]
        assert isinstance(input_dim, InputTokens)
        assert isinstance(output_dim, OutputTokens)
        assert input_dim.price_per_million == 3.0
        assert output_dim.price_per_million == 15.0
        assert cost_model.run_dims == {}

    def test_mlflow_callback_restored(self, base_config):
        mlflow_cb = base_config.callbacks[1]
        assert isinstance(mlflow_cb, MlflowCallback)
        assert mlflow_cb.step is None
        assert mlflow_cb.nested is True


class TestLegacyV01EndpointTypeLoad:
    """Legacy v0.1 format: ``endpoint_type`` dispatch + flat cost annotations on responses."""

    @pytest.fixture
    def legacy_dir(self, snapshots_dir):
        return snapshots_dir / "legacy" / "v0_1_endpoint_type"

    def test_result_loads(self, legacy_dir):
        result = Result.load(legacy_dir)
        assert len(result.responses) == 5
        assert result.model_id == "synthetic.legacy-model-v1"
        assert result.run_name == "synthetic-legacy-test"

    def test_legacy_cost_keys_collected_into_annotations(self, legacy_dir):
        """Flat ``cost_*`` keys that current InvocationResponse doesn't declare go to annotations."""
        result = Result.load(legacy_dir)
        first = result.responses[0]
        assert first.annotations["cost_InputTokens"] == pytest.approx(0.000084)
        assert first.annotations["cost_OutputTokens"] == pytest.approx(0.00108)
        assert first.annotations["cost_total"] == pytest.approx(0.001164)

    def test_endpoint_restored_via_endpoint_type(self, legacy_dir):
        cfg = _RunConfig.load(legacy_dir)
        ep = cfg._endpoint
        assert isinstance(ep, BedrockConverse)
        assert ep.model_id == "synthetic.legacy-model-v1"
        assert ep.region == "us-west-2"
        # provider is derived by the constructor, not read from the (legacy) config dict
        assert ep.provider == "bedrock"

    def test_tokenizer_restored_via_tokenizer_module(self, legacy_dir):
        cfg = _RunConfig.load(legacy_dir)
        assert isinstance(cfg._tokenizer, DummyTokenizer)


class TestLegacyV01StrCallbacksLoad:
    """Legacy v0.1 data where callbacks were serialized as Python repr strings."""

    @pytest.fixture
    def strcb_dir(self, snapshots_dir):
        return snapshots_dir / "legacy" / "v0_1_str_callbacks"

    def test_result_loads(self, strcb_dir):
        result = Result.load(strcb_dir)
        assert len(result.responses) == 5
        assert result.model_id == "synthetic.strcb-model-v1"

    def test_config_loads_without_raising_on_str_callbacks(self, strcb_dir):
        """Unrestorable string callbacks must not break run_config loading."""
        cfg = _RunConfig.load(strcb_dir)
        assert isinstance(cfg._endpoint, BedrockConverseStream)
        # String callbacks are left untouched (not restored to objects, but not fatal either)
        assert cfg.callbacks is not None
        assert len(cfg.callbacks) == 2
        assert all(isinstance(cb, str) for cb in cfg.callbacks)


class TestLegacyV01InterruptedRunLoad:
    """A legacy-format interrupted run: no summary.json + top-level endpoint fields.

    Exercises the legacy branch of ``Result._recover_metadata``: older data kept endpoint
    fields (including the derived ``provider``) at the top level rather than nested in a
    ``__llmeter_state__`` envelope. Recovery must handle it - consistent with the legacy
    handling in ``_RunConfig.load`` / ``Endpoint.load`` - and warn that the format is
    deprecated.
    """

    @pytest.fixture
    def legacy_interrupted_dir(self, snapshots_dir):
        return snapshots_dir / "legacy" / "v0_1_interrupted"

    def test_no_summary_file_present(self, legacy_interrupted_dir):
        assert not (legacy_interrupted_dir / "summary.json").exists()

    def test_recovers_top_level_endpoint_fields(self, legacy_interrupted_dir):
        result = Result.load(legacy_interrupted_dir)
        assert result.model_id == "synthetic.legacy-interrupted-v1"
        assert result.endpoint_name == "synthetic-legacy-intr-endpoint"
        # Legacy configs persist the derived provider at the top level, so it recovers
        assert result.provider == "bedrock"
        assert result.clients == 2
        assert result.n_requests == 4

    def test_recovery_warns_legacy_format(self, legacy_interrupted_dir, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="llmeter.results"):
            Result.load(legacy_interrupted_dir)
        assert any("legacy data format" in r.message for r in caplog.records)


class TestInterruptedRunLoad:
    """A snapshot with no summary.json - exercises metadata recovery."""

    @pytest.fixture
    def interrupted_dir(self, snapshots_dir):
        return snapshots_dir / "interrupted_run"

    def test_no_summary_file_present(self, interrupted_dir):
        assert not (interrupted_dir / "summary.json").exists()

    def test_recovers_responses(self, interrupted_dir):
        result = Result.load(interrupted_dir)
        assert len(result.responses) == 5
        assert result.total_requests == 5

    def test_recovers_metadata_from_run_config(self, interrupted_dir):
        result = Result.load(interrupted_dir)
        assert result.clients == 4
        assert result.n_requests == 20
        assert result.run_name == "synthetic-interrupted-test"

    def test_recovers_endpoint_fields_from_modern_config(self, interrupted_dir):
        """Endpoint fields must be read from the modern ``__llmeter_state__`` envelope.

        A *current* interrupted run writes run_config.json via ``dump_object`` (nested
        state), so recovery must look inside ``__llmeter_state__`` and not just the top
        level. ``provider`` is derived by the endpoint constructor and not persisted, so
        it stays ``None`` on this lightweight recovery path (no endpoint instantiation).
        """
        result = Result.load(interrupted_dir)
        assert result.model_id == "synthetic.interrupted-model-v1"
        assert result.endpoint_name == "synthetic-interrupted-endpoint"
        assert result.provider is None

    def test_derives_timestamps_from_responses(self, interrupted_dir):
        result = Result.load(interrupted_dir)
        # request_time runs 16:30:02 .. 16:30:10 (i*2 for i in 1..5)
        assert result.start_time == datetime(
            2025, 1, 20, 16, 30, 2, tzinfo=timezone.utc
        )
        assert result.end_time == datetime(2025, 1, 20, 16, 30, 10, tzinfo=timezone.utc)

    def test_modern_config_recovery_does_not_warn_legacy(self, interrupted_dir, caplog):
        """The modern-format recovery path must not emit the legacy-format warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="llmeter.results"):
            Result.load(interrupted_dir)
        assert not any("legacy data format" in r.message for r in caplog.records)

    def test_computes_stats_from_responses(self, interrupted_dir):
        result = Result.load(interrupted_dir)
        stats = result.stats
        assert stats["failed_requests"] == 0
        assert "time_to_first_token-p50" in stats

    def test_endpoint_config_with_legacy_ttft_flag_is_constructible(
        self, interrupted_dir
    ):
        """This snapshot's endpoint config retains the retired ``ttft_visible_tokens_only`` key.

        ``Result.load`` reads endpoint metadata straight out of the JSON without instantiating the
        endpoint, so it never reaches the constructor and never sees the deprecated argument. That
        makes this the test that actually pins backward compatibility: reconstructing the endpoint
        from the saved config must still succeed, and must warn rather than raise ``TypeError``.
        """
        config = json.loads((interrupted_dir / "run_config.json").read_text())
        endpoint_state = config["endpoint"]["__llmeter_state__"]
        assert "ttft_visible_tokens_only" in endpoint_state, (
            "Fixture should retain the retired key; see _generate_fixtures.py"
        )

        with pytest.warns(DeprecationWarning, match="ttft_visible_tokens_only"):
            endpoint = load_object(config["endpoint"])

        assert isinstance(endpoint, BedrockConverseStream)
        assert endpoint.model_id == "synthetic.interrupted-model-v1"
        # The retired flag is accepted and discarded, not persisted onto the instance
        assert not hasattr(endpoint, "ttft_visible_tokens_only")

    def test_loading_result_does_not_warn_deprecated(self, interrupted_dir):
        """Loading the *Result* should stay quiet, since it never builds the endpoint.

        Guards against a future change that starts instantiating endpoints during recovery and
        surfaces a confusing deprecation warning to users just for opening an old run.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Result.load(interrupted_dir)
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


class TestErrorsAndAnnotationsLoad:
    """Responses carrying errors, retries, cached tokens, and custom annotations."""

    @pytest.fixture
    def errors_dir(self, snapshots_dir):
        return snapshots_dir / "errors_and_annotations"

    def test_result_loads(self, errors_dir):
        result = Result.load(errors_dir)
        assert len(result.responses) == 5

    def test_errors_preserved(self, errors_dir):
        result = Result.load(errors_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-err-002"].error == "ThrottlingException: Rate exceeded"
        assert by_id["synth-err-004"].error == "ReadTimeoutError: timed out after 60s"
        assert by_id["synth-err-001"].error is None

    def test_failed_requests_counted(self, errors_dir):
        result = Result.load(errors_dir)
        assert result.stats["failed_requests"] == 2

    def test_retries_preserved(self, errors_dir):
        result = Result.load(errors_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-err-002"].retries == 2
        assert by_id["synth-err-005"].retries == 1
        assert by_id["synth-err-001"].retries == 0

    def test_custom_annotations_preserved(self, errors_dir):
        result = Result.load(errors_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-err-001"].annotations == {
            "custom_metric": 42.5,
            "experiment_tag": "baseline",
        }
        assert by_id["synth-err-005"].annotations["cache_hit_ratio"] == pytest.approx(
            0.667
        )

    def test_cached_input_tokens_preserved(self, errors_dir):
        result = Result.load(errors_dir)
        by_id = {r.id: r for r in result.responses}
        assert by_id["synth-err-005"].num_tokens_input_cached == 30


class TestLoadTestResultLoad:
    """A multi-concurrency snapshot loaded via LoadTestResult.load."""

    @pytest.fixture
    def load_test_dir(self, snapshots_dir):
        return snapshots_dir / "load_test"

    def test_loads_all_concurrency_levels(self, load_test_dir):
        lt = LoadTestResult.load(load_test_dir)
        assert set(lt.results.keys()) == {1, 3}

    def test_sub_results_have_correct_data(self, load_test_dir):
        lt = LoadTestResult.load(load_test_dir)
        assert len(lt.results[1].responses) == 5
        assert len(lt.results[3].responses) == 5
        assert lt.results[1].clients == 1
        assert lt.results[3].clients == 3

    def test_sub_result_stats_populated(self, load_test_dir):
        lt = LoadTestResult.load(load_test_dir)
        for clients in (1, 3):
            stats = lt.results[clients].stats
            assert stats["total_requests"] == 5
            assert "time_to_first_token-p50" in stats

    def test_load_without_responses(self, load_test_dir):
        lt = LoadTestResult.load(load_test_dir, load_responses=False)
        assert lt.results[1].responses == []
        assert lt.results[1].stats["total_requests"] == 5


class TestLegacyV02VisibleOnlyTtftWarning:
    """Loading a v0.2.0 file must flag that `time_to_first_token` meant something else.

    See ``llmeter/warnings.py``. LLMeter warns rather than migrating, because the old
    ``ttft_visible_tokens_only=True`` default applied whether or not a model reasoned, and Bedrock
    Converse / Anthropic files carry no reasoning-token count to detect it from.
    """

    @pytest.fixture
    def legacy_dir(self, snapshots_dir):
        return snapshots_dir / "legacy" / "v0_2_visible_only_ttft"

    def test_fixture_really_lacks_the_field(self, legacy_dir):
        """Guards the fixture itself: if it gained the key, the warning test would be vacuous."""
        rows = [
            json.loads(line)
            for line in (legacy_dir / "responses.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert rows, "fixture should contain responses"
        assert all("time_to_first_content_token" not in r for r in rows)
        assert all(r["time_to_first_token"] is not None for r in rows)

    def test_load_warns(self, legacy_dir):
        with pytest.warns(LegacyResultFormatWarning) as record:
            Result.load(legacy_dir)

        assert len(record) == 1, "should warn once per load, not once per response"
        message = str(record[0].message)
        assert "time_to_first_content_token" in message
        assert "v0_2_visible_only_ttft" in message, "should identify which file"

    def test_values_are_not_rewritten(self, legacy_dir):
        """The warning is informational: loaded metrics must be exactly what was saved."""
        rows = [
            json.loads(line)
            for line in (legacy_dir / "responses.jsonl").read_text().splitlines()
            if line.strip()
        ]
        with pytest.warns(LegacyResultFormatWarning):
            result = Result.load(legacy_dir)

        for saved, loaded in zip(rows, result.responses):
            assert loaded.time_to_first_token == saved["time_to_first_token"]
            assert loaded.time_per_output_token == saved["time_per_output_token"]
            # Not back-filled by guesswork
            assert loaded.time_to_first_content_token is None
            assert loaded.num_tokens_output_reasoning is None

    def test_warning_is_filterable_by_category(self, legacy_dir):
        """Users must be able to silence just this, without hiding other warnings."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=LegacyResultFormatWarning)
            Result.load(legacy_dir)
        assert [
            w for w in caught if issubclass(w.category, LegacyResultFormatWarning)
        ] == []

    def test_no_warning_when_responses_not_loaded(self, legacy_dir):
        """Known limitation, pinned deliberately.

        With ``load_responses=False`` there are no records to inspect, so the legacy format cannot
        be detected -- and the stats read from ``stats.json`` were computed under the old
        definition. Documented rather than silently papered over.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Result.load(legacy_dir, load_responses=False)
        assert [
            w for w in caught if issubclass(w.category, LegacyResultFormatWarning)
        ] == []

    @pytest.mark.parametrize(
        "snapshot", ["base", "interrupted_run", "errors_and_annotations"]
    )
    def test_modern_snapshots_do_not_warn(self, snapshots_dir, snapshot):
        """Current-format fixtures must be silent, or the warning would be noise."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Result.load(snapshots_dir / snapshot)
        assert [
            w for w in caught if issubclass(w.category, LegacyResultFormatWarning)
        ] == []

    def test_non_streaming_responses_do_not_warn(self, tmp_path):
        """A non-streaming run legitimately has neither metric, and must not be mistaken for old.

        This is why detection requires a non-null ``time_to_first_token``, not merely a missing
        ``time_to_first_content_token``.
        """
        (tmp_path / "responses.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "response_text": "hi",
                        "time_to_first_token": None,
                        "time_to_last_token": 1.0,
                        "num_tokens_output": 5,
                    }
                )
                for _ in range(3)
            )
            + "\n"
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Result.load(tmp_path)
        assert [
            w for w in caught if issubclass(w.category, LegacyResultFormatWarning)
        ] == []

    def test_fixture_keeps_the_rest_of_the_v0_2_field_set(self, legacy_dir):
        """This is a v0.2.0 fixture, not a v0.1 one.

        ``ttft_visible_tokens_only`` was introduced by the release that became v0.2.0, so a v0.1
        file could never carry it. The fixture must therefore retain everything v0.2.0 shipped and
        omit only ``time_to_first_content_token`` -- otherwise it would be testing the wrong
        migration boundary.
        """
        rows = [
            json.loads(line)
            for line in (legacy_dir / "responses.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for row in rows:
            assert "time_to_first_content_token" not in row
            for field in (
                "num_tokens_input_cached",
                "num_tokens_output_reasoning",
                "retries",
                "annotations",
            ):
                assert field in row, (
                    f"v0.2.0 shipped {field!r}, so it should be present (possibly null)"
                )

    def test_run_config_records_the_retired_flag(self, legacy_dir):
        """Documents where the old semantics came from, and that detection cannot rely on it."""
        config = json.loads((legacy_dir / "run_config.json").read_text())
        state = config["endpoint"]["__llmeter_state__"]
        assert state["ttft_visible_tokens_only"] is True


class TestLegacyV01FieldSets:
    """Every ``legacy/v0_1_*`` snapshot must carry only fields that existed in v0.1.

    A v0.1 fixture that contains a later field isn't testing v0.1 compatibility. This guards all of
    them at once so a newly-added response field can't silently leak into the legacy snapshots.
    """

    POST_V0_1_FIELDS = (
        "num_tokens_input_cached",
        "num_tokens_output_reasoning",
        "retries",
        "annotations",
        "time_to_first_content_token",
    )

    @pytest.mark.parametrize(
        "snapshot", ["v0_1_endpoint_type", "v0_1_str_callbacks", "v0_1_interrupted"]
    )
    def test_no_post_v0_1_fields(self, snapshots_dir, snapshot):
        rows = [
            json.loads(line)
            for line in (snapshots_dir / "legacy" / snapshot / "responses.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        assert rows, f"{snapshot} should contain responses"
        for row in rows:
            leaked = [f for f in self.POST_V0_1_FIELDS if f in row]
            assert not leaked, f"{snapshot} leaked post-v0.1 field(s): {leaked}"


class TestFixturePathHygiene:
    """Committed fixtures must not embed real filesystem paths.

    Snapshots are checked into git, so a path interpolated at generation time would leak the
    generating developer's filesystem layout and would differ between contributors. Every
    persisted path should be an obviously-fake placeholder (the loader overrides ``output_path``
    from the directory it actually reads from, so nothing depends on it being real).
    """

    #: Substrings that indicate a real, machine-specific path rather than a placeholder.
    REAL_PATH_MARKERS = (
        "/Users/",
        "/home/",
        "/tmp/",
        "/var/folders/",
        ":\\",
        "file://",
    )

    def _fixture_json_files(self, snapshots_dir):
        return sorted(snapshots_dir.rglob("*.json"))

    def test_fixtures_exist(self, snapshots_dir):
        assert self._fixture_json_files(snapshots_dir), (
            "expected committed fixture JSON files"
        )

    def test_no_absolute_or_machine_specific_paths(self, snapshots_dir):
        offenders = []
        for path in self._fixture_json_files(snapshots_dir):
            text = path.read_text()
            for marker in self.REAL_PATH_MARKERS:
                if marker in text:
                    offenders.append(
                        f"{path.relative_to(snapshots_dir)} contains {marker!r}"
                    )
        assert not offenders, "Fixtures embed real paths:\n  " + "\n  ".join(offenders)

    def test_persisted_output_paths_are_placeholders(self, snapshots_dir):
        """`output_path` must be null or an obviously-stale placeholder, never a resolved path."""
        offenders = []
        for path in self._fixture_json_files(snapshots_dir):
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                continue
            value = data.get("output_path")
            if value is None:
                continue
            if not str(value).startswith(("stale/", "tests/")):
                offenders.append(f"{path.relative_to(snapshots_dir)}: {value!r}")
        assert not offenders, (
            "output_path should be null or a 'stale/...' placeholder:\n  "
            + "\n  ".join(offenders)
        )


class TestLegacyStatsJsonFidelity:
    """The `stats.json` in a `v0_*` fixture must match what that release actually wrote.

    These fixtures exist to prove backward compatibility, so their contents have to be what a user
    upgrading really has on disk -- which is *not* what current code computes from the same
    responses. `stats.json` is written by `Result.save()` from `Result.stats`, and which code
    produced that changed mid-v0.1-line:

    * **v0.1.0 - v0.1.9**: no `RunningStats` yet, so stats came from `Result._compute_stats`.
    * **v0.1.10 onwards** (`RunningStats` added in commit `4d9ce3e`): stats came from
      `RunningStats.to_stats`, whose metric list included `time_per_output_token` but *not*
      `num_tokens_input_cached` -- the opposite of the same release's `_compute_stats` -- and which
      also emits `output_tps`.

    So each fixture has to name an era, and its `responses.jsonl` field set has to belong to the same
    one. The `v0_1_*` fixtures omit `retries` (added v0.1.5) and the cached/reasoning token counts
    (added v0.1.10), so they are early-v0.1 and get the `_compute_stats` shape;
    `v0_2_visible_only_ttft` carries the full v0.2.0 field set and gets the `RunningStats` shape.

    `_generate_fixtures._as_legacy_stats` applies this, and these tests keep it honest -- without
    them, any change to `llmeter.results._STANDARD_AGGREGATION_METRICS` silently rewrites the legacy
    fixtures and they quietly stop testing the format they are named after.
    """

    AGGREGATIONS = ("average", "p50", "p90", "p99")
    """Aggregation suffixes, for telling `{metric}-{aggregation}` keys from run-level ones"""

    COMMON_RUN_KEYS = {
        "failed_requests",
        "failed_requests_rate",
        "total_input_tokens",
        "total_output_tokens",
        "requests_per_minute",
        "average_input_tokens_per_minute",
        "average_output_tokens_per_minute",
    }
    """Run-level keys every era's save path produced"""

    RECOMPUTE_ONLY_RUN_KEYS = {
        "total_cached_input_tokens",
        "total_reasoning_output_tokens",
    }
    """
    Run-level keys only `results._get_run_stats` produces

    These never reached a saved file in any of these eras - they arrived with v0.1.10, on the
    *recompute* path only.
    """

    ERAS = [
        # Early v0.1: `_compute_stats`-written, before `RunningStats` existed.
        (
            "v0_1_endpoint_type",
            {
                "time_to_last_token",
                "time_to_first_token",
                "num_tokens_output",
                "num_tokens_input",
            },
            False,
        ),
        (
            "v0_1_str_callbacks",
            {
                "time_to_last_token",
                "time_to_first_token",
                "num_tokens_output",
                "num_tokens_input",
            },
            False,
        ),
        # v0.2.0: `RunningStats`-written.
        (
            "v0_2_visible_only_ttft",
            {
                "time_to_last_token",
                "time_to_first_token",
                "time_per_output_token",
                "num_tokens_output",
                "num_tokens_input",
            },
            True,
        ),
    ]
    """
    `(fixture, expected aggregated metrics, expects output_tps)` per era

    Deliberately duplicated from `_generate_fixtures` rather than imported: a test that imported the
    same constant it checks would pass no matter what that constant said.
    """

    @classmethod
    def _split_keys(cls, stats: dict) -> tuple[set[str], set[str]]:
        """Partition stat keys into (aggregated metric names, run-level keys)."""
        metrics, run_level = set(), set()
        for key in stats:
            base, _, aggregation = key.rpartition("-")
            if aggregation in cls.AGGREGATIONS and base:
                metrics.add(base)
            else:
                run_level.add(key)
        return metrics, run_level

    @staticmethod
    def _load(snapshots_dir, fixture: str) -> dict:
        return json.loads(
            (snapshots_dir / "legacy" / fixture / "stats.json").read_text()
        )

    @pytest.mark.parametrize("fixture,expected_metrics,_tps", ERAS)
    def test_aggregated_metrics_match_the_era(
        self, snapshots_dir, fixture, expected_metrics, _tps
    ):
        metrics, _ = self._split_keys(self._load(snapshots_dir, fixture))

        assert metrics == expected_metrics

    @pytest.mark.parametrize("fixture,_metrics,expects_output_tps", ERAS)
    def test_output_tps_presence_matches_the_era(
        self, snapshots_dir, fixture, _metrics, expects_output_tps
    ):
        """`output_tps` is a `RunningStats.to_stats` key, so it only exists from v0.1.10."""
        _, run_level = self._split_keys(self._load(snapshots_dir, fixture))

        assert ("output_tps" in run_level) is expects_output_tps

    @pytest.mark.parametrize("fixture,_metrics,_tps", ERAS)
    def test_common_run_level_keys_present(
        self, snapshots_dir, fixture, _metrics, _tps
    ):
        _, run_level = self._split_keys(self._load(snapshots_dir, fixture))

        assert self.COMMON_RUN_KEYS <= run_level, (
            f"missing keys a real save wrote: {sorted(self.COMMON_RUN_KEYS - run_level)}"
        )

    @pytest.mark.parametrize("fixture,_metrics,_tps", ERAS)
    def test_no_recompute_only_run_keys(self, snapshots_dir, fixture, _metrics, _tps):
        _, run_level = self._split_keys(self._load(snapshots_dir, fixture))

        assert not (run_level & self.RECOMPUTE_ONLY_RUN_KEYS), (
            "these come from `_get_run_stats`, which never ran on the save path"
        )

    @pytest.mark.parametrize("fixture,_metrics,_tps", ERAS)
    def test_no_aggregations_for_later_metrics(
        self, snapshots_dir, fixture, _metrics, _tps
    ):
        """`time_to_first_content_token` post-dates every one of these files."""
        metrics, _ = self._split_keys(self._load(snapshots_dir, fixture))

        assert "time_to_first_content_token" not in metrics

    def test_early_v0_1_response_fields_match_that_era(self, snapshots_dir):
        """Guard the coherence the era profiles depend on.

        If these fixtures gained `retries` (v0.1.5) or the cached/reasoning token counts (v0.1.10),
        they would no longer be early-v0.1 and the `_compute_stats`-shaped stats above would be
        emulating a combination that never shipped.
        """
        for fixture in ("v0_1_endpoint_type", "v0_1_str_callbacks"):
            path = snapshots_dir / "legacy" / fixture / "responses.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            assert rows, fixture
            for field in (
                "retries",
                "num_tokens_input_cached",
                "num_tokens_output_reasoning",
            ):
                assert all(field not in r for r in rows), f"{fixture}: {field}"


class TestLegacyStatsLoadBehaviour:
    """What a caller actually gets when loading these legacy stats, and the traps in it."""

    @pytest.fixture
    def legacy_dir(self, snapshots_dir):
        return snapshots_dir / "legacy" / "v0_2_visible_only_ttft"

    def test_content_ttft_aggregates_absent_either_way(self, legacy_dir):
        """No `time_to_first_content_token-*` keys appear, with or without responses.

        Loading *with* responses recomputes stats under the current metric list, which does include
        the metric -- but every legacy response has it unset, so it aggregates to nothing. Code
        reading TTFCT aggregates must therefore tolerate their absence rather than assume the
        current schema.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LegacyResultFormatWarning)
            with_responses = Result.load(legacy_dir)
            without_responses = Result.load(legacy_dir, load_responses=False)

        for label, result in (
            ("with responses", with_responses),
            ("without responses", without_responses),
        ):
            assert not [
                k for k in result.stats if k.startswith("time_to_first_content_token-")
            ], label

    def test_saved_tpot_is_preserved_but_uses_the_old_definition(self, legacy_dir):
        """The subtle trap: TPOT is present and looks current, but was computed differently.

        These files pair a *visible*-token TTFT with a reasoning-inclusive output token count, so
        their `time_per_output_token` is understated for reasoning models. It is loaded unchanged --
        LLMeter cannot recompute it without the missing TTFT -- which is precisely what
        `LegacyResultFormatWarning` exists to flag.
        """
        saved = json.loads((legacy_dir / "stats.json").read_text())
        assert "time_per_output_token-p50" in saved, (
            "guard: fixture must carry saved TPOT"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LegacyResultFormatWarning)
            result = Result.load(legacy_dir, load_responses=False)

        assert (
            result.stats["time_per_output_token-p50"]
            == saved["time_per_output_token-p50"]
        )

    def test_saved_only_keys_survive_loading_with_responses(self, legacy_dir):
        """`output_tps` is in the saved file but not recomputable, so it must survive the merge.

        `Result._resolve_stats` recomputes from responses and then merges back any `stats.json` key
        recomputation did not produce. Without that, upgrading would silently drop stats from
        previously-saved runs.
        """
        saved = json.loads((legacy_dir / "stats.json").read_text())
        assert "output_tps" in saved, "guard: fixture must carry a recompute-only key"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LegacyResultFormatWarning)
            result = Result.load(legacy_dir)

        assert result.responses, (
            "should have loaded responses, or the merge path is untested"
        )
        assert result.stats["output_tps"] == saved["output_tps"]
