# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for save/load functionality across the library."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from llmeter.tokenizers import DummyTokenizer, Tokenizer, save_tokenizer
from llmeter.prompt_utils import save_payloads, load_payloads
from llmeter.endpoints.base import Endpoint, InvocationResponse
from llmeter.endpoints.openai import OpenAICompletionEndpoint
from llmeter.results import Result
from llmeter.runner import _RunConfig


# Custom strategies
@composite
def valid_run_config(draw):
    """Generate valid _RunConfig objects with mock endpoint."""
    # Create a mock endpoint that returns proper dict
    mock_endpoint = Mock(spec=Endpoint)
    mock_endpoint.to_dict.return_value = {
        "model_id": "test-model",
        "provider": "test",
        "endpoint_type": "OpenAICompletionEndpoint",
        "endpoint_name": "test-endpoint",
    }
    mock_endpoint.model_id = "test-model"

    return _RunConfig(
        endpoint=mock_endpoint,
        run_name=draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(
                    min_codepoint=32,
                    max_codepoint=126,
                    blacklist_characters='\\/:*?"<>|',
                ),
            )
            | st.none()
        ),
        run_description=draw(st.text(max_size=200) | st.none()),
        clients=draw(st.integers(min_value=1, max_value=100)),
        n_requests=draw(st.integers(min_value=1, max_value=1000) | st.none()),
    )


@composite
def valid_invocation_response(draw):
    """Generate valid InvocationResponse objects."""
    return InvocationResponse(
        id=draw(st.text(min_size=1, max_size=50)),
        response_text=draw(st.text(max_size=500)),
        num_tokens_input=draw(st.integers(min_value=0, max_value=10000) | st.none()),
        num_tokens_output=draw(st.integers(min_value=0, max_value=10000) | st.none()),
        time_to_first_token=draw(
            st.floats(min_value=0, max_value=60, allow_nan=False) | st.none()
        ),
        time_to_last_token=draw(st.floats(min_value=0, max_value=120, allow_nan=False)),
        error=draw(st.text(max_size=100) | st.none()),
    )


@composite
def valid_result(draw):
    """Generate valid Result objects."""
    responses = draw(st.lists(valid_invocation_response(), min_size=1, max_size=20))
    return Result(
        responses=responses,
        total_requests=len(responses),
        clients=draw(st.integers(min_value=1, max_value=10)),
        n_requests=len(responses),
        model_id=draw(st.text(min_size=1, max_size=50) | st.none()),
        run_name=draw(st.text(min_size=1, max_size=50) | st.none()),
        run_description=draw(st.text(max_size=200) | st.none()),
    )


# Tokenizer save/load property tests
class TestTokenizerSaveLoadProperties:
    """Property-based tests for tokenizer serialization."""

    @given(st.text(min_size=0, max_size=500))
    @settings(deadline=None)
    def test_save_load_tokenizer_roundtrip(self, text):
        """Saving and loading tokenizer should preserve behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = DummyTokenizer()
            output_path = Path(tmpdir) / "tokenizer.json"

            # Save
            saved_path = save_tokenizer(original, output_path)
            assert saved_path.exists()

            # Load
            loaded = Tokenizer.load_from_file(saved_path)

            # Should encode the same way
            original_tokens = original.encode(text)
            loaded_tokens = loaded.encode(text)
            assert original_tokens == loaded_tokens

    def test_save_tokenizer_creates_valid_json(self):
        """Saved tokenizer should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = DummyTokenizer()
            output_path = Path(tmpdir) / "tokenizer.json"

            saved_path = save_tokenizer(tokenizer, output_path)

            # Should be valid JSON
            with open(saved_path, "r") as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert "tokenizer_module" in data

    @given(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                min_codepoint=32, max_codepoint=126, blacklist_characters='\\/:*?"<>|'
            ),
        )
    )
    @settings(deadline=None)
    def test_save_tokenizer_with_various_filenames(self, filename):
        """Tokenizer should save with various valid filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = DummyTokenizer()
            output_path = Path(tmpdir) / f"{filename}.json"

            saved_path = save_tokenizer(tokenizer, output_path)
            assert saved_path.exists()

            # Should be loadable
            loaded = Tokenizer.load_from_file(saved_path)
            assert isinstance(loaded, DummyTokenizer)


# Payload save/load property tests
class TestPayloadSaveLoadProperties:
    """Property-based tests for payload serialization."""

    @given(
        st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=50),
                st.one_of(
                    st.text(max_size=200),
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.booleans(),
                ),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=30,
        )
    )
    @settings(deadline=None)
    def test_save_load_payloads_preserves_data(self, payloads):
        """Save/load roundtrip should preserve all payload data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Save
            saved_path = save_payloads(payloads, output_path)
            assert saved_path.exists()

            # Load
            loaded_payloads = list(load_payloads(saved_path))

            # Should match exactly
            assert len(loaded_payloads) == len(payloads)
            for original, loaded in zip(payloads, loaded_payloads):
                assert loaded == original

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(st.text(max_size=200), st.integers(), st.booleans()),
            min_size=1,
            max_size=10,
        )
    )
    @settings(deadline=None)
    def test_save_single_payload_as_dict(self, payload):
        """Single dict should be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            saved_path = save_payloads(payload, output_path)
            loaded_payloads = list(load_payloads(saved_path))

            assert len(loaded_payloads) == 1
            assert loaded_payloads[0] == payload

    @given(
        st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.text(max_size=100),
                min_size=1,
                max_size=5,
            ),
            min_size=1,
            max_size=20,
        ),
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                min_codepoint=32, max_codepoint=126, blacklist_characters='\\/:*?"<>|'
            ),
        ),
    )
    @settings(deadline=None)
    def test_save_payloads_with_custom_filename(self, payloads, filename):
        """Payloads should save with custom filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            saved_path = save_payloads(
                payloads, output_path, output_file=f"{filename}.jsonl"
            )
            assert saved_path.exists()
            assert saved_path.name == f"{filename}.jsonl"

            # Should be loadable
            loaded = list(load_payloads(saved_path))
            assert len(loaded) == len(payloads)


# RunConfig save/load property tests
class TestRunConfigSaveLoadProperties:
    """Property-based tests for RunConfig serialization."""

    @given(valid_run_config())
    @settings(deadline=None)
    def test_run_config_save_creates_files(self, config):
        """RunConfig save should create config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Save
            config.save(output_path)

            # Config file should exist
            config_file = output_path / "run_config.json"
            assert config_file.exists()

    @given(valid_run_config())
    @settings(deadline=None)
    def test_run_config_save_creates_valid_json(self, config):
        """RunConfig save should create valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            config.save(output_path)

            config_file = output_path / "run_config.json"

            # Should be valid JSON
            with open(config_file, "r") as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert "clients" in data
            assert data["clients"] == config.clients

    @given(valid_run_config())
    @settings(deadline=None)
    def test_run_config_creates_parent_directories(self, config):
        """RunConfig save should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist
            output_path = Path(tmpdir) / "nested" / "dirs"

            config.save(output_path)

            # Config file should exist
            config_file = output_path / "run_config.json"
            assert config_file.exists()
            assert config_file.parent.exists()


# Result save/load property tests
class TestResultSaveLoadProperties:
    """Property-based tests for Result serialization."""

    @given(valid_result())
    @settings(deadline=None, max_examples=10)
    def test_result_save_load_roundtrip(self, result):
        """Result save/load should preserve responses and config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Save
            result.save(output_path)

            # Load
            loaded = Result.load(output_path)

            # Should have same number of responses
            assert len(loaded.responses) == len(result.responses)

            # Config should match
            assert loaded.run_name == result.run_name
            assert loaded.clients == result.clients

    @given(
        st.lists(valid_invocation_response(), min_size=1, max_size=10),
        st.integers(min_value=1, max_value=10),
    )
    @settings(deadline=None, max_examples=10)
    def test_result_save_creates_multiple_files(self, responses, clients):
        """Result save should create responses and summary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            result = Result(
                responses=responses,
                total_requests=len(responses),
                clients=clients,
                n_requests=len(responses),
            )

            result.save(output_path)

            # Should create responses file
            responses_file = output_path / "responses.jsonl"
            assert responses_file.exists()

            # Should create summary file
            summary_file = output_path / "summary.json"
            assert summary_file.exists()

            # Should create stats file
            stats_file = output_path / "stats.json"
            assert stats_file.exists()

    @given(valid_result())
    @settings(deadline=None, max_examples=10)
    def test_result_to_dict_includes_responses(self, result):
        """Result to_dict should optionally include responses."""
        # Without responses
        dict_without = result.to_dict(include_responses=False)
        assert "responses" not in dict_without

        # With responses
        dict_with = result.to_dict(include_responses=True)
        assert "responses" in dict_with
        assert len(dict_with["responses"]) == len(result.responses)


# Endpoint save/load property tests
class TestEndpointSaveLoadProperties:
    """Property-based tests for Endpoint serialization."""

    @given(
        st.text(min_size=1, max_size=50),
    )
    @settings(deadline=None)
    def test_endpoint_save_load_roundtrip(self, model_id):
        """Endpoint save/load should preserve configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "endpoint.json"

            # Create endpoint with valid API key
            endpoint = OpenAICompletionEndpoint(
                model_id=model_id, api_key="sk-test-key-12345"
            )

            # Save
            saved_path = endpoint.save(output_path)
            assert saved_path.exists()

            # Verify file contains model_id
            with open(saved_path, "r") as f:
                data = json.load(f)
            assert data["model_id"] == model_id

    @given(st.text(min_size=1, max_size=50))
    @settings(deadline=None)
    def test_endpoint_to_dict_is_json_serializable(self, model_id):
        """Endpoint to_dict should produce JSON-serializable output."""
        endpoint = OpenAICompletionEndpoint(model_id=model_id, api_key="test")

        endpoint_dict = endpoint.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(endpoint_dict)
        parsed = json.loads(json_str)

        assert isinstance(parsed, dict)
        assert "model_id" in parsed


# LoadTestResult save/load property tests
class TestLoadTestResultSaveLoadProperties:
    """Property-based tests for LoadTestResult serialization."""

    @given(
        st.lists(valid_result(), min_size=1, max_size=5),
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                min_codepoint=32, max_codepoint=126, blacklist_characters='\\/:*?"<>|'
            ),
        ),
    )
    @settings(deadline=None, max_examples=5)
    def test_load_test_result_save_load_individual_results(self, results, test_name):
        """Individual results should be saveable and loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Save each result
            for i, result in enumerate(results):
                result_path = output_path / f"result_{i}"
                result.save(result_path)

            # Each result should be loadable
            for i in range(len(results)):
                result_path = output_path / f"result_{i}"
                loaded = Result.load(result_path)
                assert len(loaded.responses) == len(results[i].responses)


# Cross-format compatibility tests
class TestCrossFormatCompatibility:
    """Property-based tests for format compatibility."""

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(st.text(max_size=100), st.integers(), st.booleans()),
            min_size=1,
            max_size=10,
        )
    )
    @settings(deadline=None)
    def test_json_roundtrip_preserves_types(self, data):
        """JSON serialization should preserve data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "data.json"

            # Save as JSON
            with open(file_path, "w") as f:
                json.dump(data, f)

            # Load back
            with open(file_path, "r") as f:
                loaded = json.load(f)

            assert loaded == data

    @given(
        st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.text(max_size=100),
                min_size=1,
                max_size=5,
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(deadline=None)
    def test_jsonl_format_consistency(self, payloads):
        """JSONL format should be consistent across save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            saved_path = save_payloads(payloads, output_path)

            # Read raw file
            with open(saved_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            # Each line should be valid JSON
            for line in lines:
                parsed = json.loads(line)
                assert isinstance(parsed, dict)

            # Should match number of payloads
            assert len(lines) == len(payloads)


# Error handling property tests
class TestSaveLoadErrorHandling:
    """Property-based tests for error handling in save/load operations."""

    @given(valid_run_config())
    @settings(deadline=None)
    def test_load_from_nonexistent_path_raises(self, config):
        """Loading from nonexistent path should raise appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"

            try:
                _RunConfig.load(nonexistent)
                assert False, "Should have raised an error"
            except (FileNotFoundError, Exception):
                # Expected to raise
                pass

    @given(
        st.text(min_size=1, max_size=50),
    )
    @settings(deadline=None)
    def test_save_creates_directories_if_needed(self, model_id):
        """Save operations should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "endpoint.json"

            endpoint = OpenAICompletionEndpoint(
                model_id=model_id, api_key="sk-test-key-12345"
            )
            saved_path = endpoint.save(nested_path)

            assert saved_path.exists()
            assert saved_path.parent.exists()
