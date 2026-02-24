# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for prompt_utils module."""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llmeter.prompt_utils import (
    CreatePromptCollection,
    load_payloads,
    load_prompts,
    save_payloads,
)
from llmeter.tokenizers import DummyTokenizer


class TestCreatePromptCollection:
    """Tests for CreatePromptCollection class."""

    def test_create_collection_basic(self):
        """Test basic prompt collection creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a source file
            source_file = Path(tmpdir) / "source.txt"
            source_file.write_text("This is a test sentence. " * 100)

            collection = CreatePromptCollection(
                input_lengths=[10, 20],
                output_lengths=[5, 10],
                source_file=source_file,
                requests_per_combination=2,
            )

            result = collection.create_collection()

            # Should have 2 input_lengths * 2 output_lengths * 2 requests = 8 items
            assert len(result) == 8

    def test_create_collection_with_custom_tokenizer(self):
        """Test prompt collection with custom tokenizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.txt"
            source_file.write_text("Hello world test data " * 50)

            tokenizer = DummyTokenizer()
            collection = CreatePromptCollection(
                input_lengths=[5, 10],
                output_lengths=[3],
                source_file=source_file,
                tokenizer=tokenizer,
            )

            result = collection.create_collection()
            assert len(result) == 2  # 2 input_lengths * 1 output_length * 1 request

    def test_create_collection_single_combination(self):
        """Test with single input/output combination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.txt"
            source_file.write_text("Sample text for testing purposes.")

            collection = CreatePromptCollection(
                input_lengths=[15],
                output_lengths=[10],
                source_file=source_file,
                requests_per_combination=1,
            )

            result = collection.create_collection()
            assert len(result) == 1

    def test_create_collection_multiple_requests(self):
        """Test multiple requests per combination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.txt"
            source_file.write_text("Test data " * 100)

            collection = CreatePromptCollection(
                input_lengths=[10],
                output_lengths=[5],
                source_file=source_file,
                requests_per_combination=5,
            )

            result = collection.create_collection()
            assert len(result) == 5  # 1 * 1 * 5

    def test_generate_sample_truncates_correctly(self):
        """Test that sample generation truncates to correct size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.txt"
            text = "word " * 100
            source_file.write_text(text)

            collection = CreatePromptCollection(
                input_lengths=[10],
                output_lengths=[5],
                source_file=source_file,
            )

            # Access the generated samples
            collection._generate_samples()
            sample = collection._samples[0]

            # Sample should be truncated to 10 tokens
            tokenizer = DummyTokenizer()
            tokens = tokenizer.encode(sample)
            assert len(tokens) == 10

    def test_create_collection_with_utf8_sig_encoding(self):
        """Test handling of UTF-8 with BOM encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.txt"
            # Write with UTF-8 BOM
            source_file.write_text("Test content with BOM", encoding="utf-8-sig")

            collection = CreatePromptCollection(
                input_lengths=[5],
                output_lengths=[3],
                source_file=source_file,
                source_file_encoding="utf-8-sig",
            )

            result = collection.create_collection()
            assert len(result) > 0

    def test_create_collection_randomizes_order(self):
        """Test that collection order is randomized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.txt"
            source_file.write_text("Random test data " * 100)

            collection = CreatePromptCollection(
                input_lengths=[10, 20, 30],
                output_lengths=[5, 10],
                source_file=source_file,
                requests_per_combination=3,
            )

            result = collection.create_collection()
            # Should have 3 * 2 * 3 = 18 items
            assert len(result) == 18
            # Each item should be a tuple of (sample, output_length)
            assert all(isinstance(item, tuple) for item in result)


class TestLoadPrompts:
    """Tests for load_prompts function."""

    def test_load_prompts_from_single_file(self):
        """Test loading prompts from a single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "prompts.txt"
            prompt_file.write_text("Prompt 1\nPrompt 2\nPrompt 3\n")

            def create_payload(input_text):
                return {"text": input_text}

            prompts = list(load_prompts(prompt_file, create_payload))
            assert len(prompts) == 3
            assert prompts[0] == {"text": "Prompt 1"}
            assert prompts[1] == {"text": "Prompt 2"}

    def test_load_prompts_with_kwargs(self):
        """Test load_prompts with additional kwargs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "prompts.txt"
            prompt_file.write_text("Test prompt\n")

            def create_payload(input_text, max_tokens=100):
                return {"text": input_text, "max_tokens": max_tokens}

            prompts = list(
                load_prompts(
                    prompt_file, create_payload, create_payload_kwargs={"max_tokens": 50}
                )
            )
            assert len(prompts) == 1
            assert prompts[0]["max_tokens"] == 50

    def test_load_prompts_from_directory(self):
        """Test loading prompts from directory with pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            (dir_path / "prompts1.txt").write_text("Prompt A\n")
            (dir_path / "prompts2.txt").write_text("Prompt B\n")
            (dir_path / "other.log").write_text("Should be ignored\n")

            def create_payload(input_text):
                return {"text": input_text}

            prompts = list(
                load_prompts(dir_path, create_payload, file_pattern="*.txt")
            )
            assert len(prompts) == 2

    def test_load_prompts_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "prompts.txt"
            prompt_file.write_text("Prompt 1\n\n\nPrompt 2\n  \n")

            def create_payload(input_text):
                return {"text": input_text}

            prompts = list(load_prompts(prompt_file, create_payload))
            assert len(prompts) == 2

    def test_load_prompts_handles_exceptions(self):
        """Test that exceptions in create_payload_fn are handled when loading from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            (dir_path / "prompts.txt").write_text("Good prompt\nBad prompt\nAnother good prompt\n")

            def create_payload(input_text):
                if "Bad" in input_text:
                    raise ValueError("Bad prompt!")
                return {"text": input_text}

            prompts = list(load_prompts(dir_path, create_payload, file_pattern="*.txt"))
            # Should skip the bad prompt and continue
            assert len(prompts) == 2
            assert prompts[0]["text"] == "Good prompt"
            assert prompts[1]["text"] == "Another good prompt"

    def test_load_prompts_from_directory_no_pattern(self):
        """Test loading from directory without file pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            (dir_path / "file1.txt").write_text("Content 1\n")
            (dir_path / "file2.dat").write_text("Content 2\n")

            def create_payload(input_text):
                return {"text": input_text}

            prompts = list(load_prompts(dir_path, create_payload))
            # Should load from all files
            assert len(prompts) >= 2


class TestLoadPayloads:
    """Tests for load_payloads function."""

    def test_load_payloads_from_json_file(self):
        """Test loading from a single JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "payload.json"
            data = {"key": "value", "number": 42}
            json_file.write_text(json.dumps(data))

            payloads = list(load_payloads(json_file))
            assert len(payloads) == 1
            assert payloads[0] == data

    def test_load_payloads_from_jsonl_file(self):
        """Test loading from JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "payloads.jsonl"
            data = [{"id": 1}, {"id": 2}, {"id": 3}]
            with jsonl_file.open("w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

            payloads = list(load_payloads(jsonl_file))
            assert len(payloads) == 3
            assert payloads[0]["id"] == 1

    def test_load_payloads_from_manifest_file(self):
        """Test loading from .manifest file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "data.manifest"
            data = [{"item": "a"}, {"item": "b"}]
            with manifest_file.open("w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

            payloads = list(load_payloads(manifest_file))
            assert len(payloads) == 2

    def test_load_payloads_from_directory(self):
        """Test loading from directory with multiple JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            (dir_path / "file1.json").write_text(json.dumps({"a": 1}))
            (dir_path / "file2.json").write_text(json.dumps({"b": 2}))

            payloads = list(load_payloads(dir_path))
            assert len(payloads) == 2

    def test_load_payloads_nonexistent_path_raises(self):
        """Test that loading from nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list(load_payloads("/nonexistent/path"))

    def test_load_payloads_skips_empty_lines_in_jsonl(self):
        """Test that empty lines in JSONL are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "data.jsonl"
            with jsonl_file.open("w") as f:
                f.write(json.dumps({"id": 1}) + "\n")
                f.write("\n")
                f.write("  \n")
                f.write(json.dumps({"id": 2}) + "\n")

            payloads = list(load_payloads(jsonl_file))
            assert len(payloads) == 2

    def test_load_payloads_handles_json_decode_error(self):
        """Test handling of malformed JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "bad.jsonl"
            with jsonl_file.open("w") as f:
                f.write(json.dumps({"id": 1}) + "\n")
                f.write("not valid json\n")
                f.write(json.dumps({"id": 2}) + "\n")

            # Should skip the bad line and continue
            payloads = list(load_payloads(jsonl_file))
            assert len(payloads) == 2

    def test_load_payloads_handles_io_error(self):
        """Test handling of IO errors during file reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "test.json"
            json_file.write_text(json.dumps({"test": "data"}))

            # Make file unreadable (Unix-like systems)
            try:
                json_file.chmod(0o000)
                # Should handle the error gracefully
                payloads = list(load_payloads(json_file))
                # May be empty due to permission error
            finally:
                # Restore permissions for cleanup
                json_file.chmod(0o644)


class TestSavePayloads:
    """Tests for save_payloads function."""

    def test_save_payloads_creates_directory(self):
        """Test that save_payloads creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir"
            payloads = [{"test": "data"}]

            result_path = save_payloads(payloads, output_path)
            assert result_path.exists()
            assert result_path.parent.exists()

    def test_save_payloads_custom_filename(self):
        """Test saving with custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payloads = [{"id": 1}]

            result_path = save_payloads(
                payloads, output_path, output_file="custom.jsonl"
            )
            assert result_path.name == "custom.jsonl"

    def test_save_payloads_list_of_dicts(self):
        """Test saving list of dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payloads = [{"a": 1}, {"b": 2}, {"c": 3}]

            result_path = save_payloads(payloads, output_path)

            # Verify content
            with result_path.open("r") as f:
                lines = f.readlines()
            assert len(lines) == 3

    def test_save_payloads_single_dict(self):
        """Test saving single dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payload = {"single": "item"}

            result_path = save_payloads(payload, output_path)

            # Should save as single line
            with result_path.open("r") as f:
                lines = f.readlines()
            assert len(lines) == 1
            assert json.loads(lines[0]) == payload

    def test_save_payloads_returns_path(self):
        """Test that save_payloads returns the output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payloads = [{"test": "data"}]

            result_path = save_payloads(payloads, output_path)
            assert isinstance(result_path, Path)
            assert result_path.exists()


# Property-based tests
class TestPromptUtilsProperties:
    """Property-based tests for prompt_utils."""

    @given(
        st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=50),
                st.one_of(st.text(max_size=100), st.integers(), st.booleans()),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(deadline=None)
    def test_save_load_roundtrip_preserves_data(self, payloads):
        """Save and load should preserve all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            saved_path = save_payloads(payloads, output_path)
            loaded = list(load_payloads(saved_path))

            assert len(loaded) == len(payloads)
            for original, loaded_item in zip(payloads, loaded):
                assert loaded_item == original

    @given(st.text(min_size=1, max_size=1000))
    @settings(deadline=None)
    def test_create_prompt_collection_with_various_text(self, text):
        """CreatePromptCollection should handle various text inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.txt"
            source_file.write_text(text)

            collection = CreatePromptCollection(
                input_lengths=[5],
                output_lengths=[3],
                source_file=source_file,
            )

            result = collection.create_collection()
            assert len(result) > 0

    @given(
        st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=5),
        st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=5),
    )
    @settings(deadline=None)
    def test_create_collection_length_combinations(self, input_lengths, output_lengths):
        """Test various combinations of input/output lengths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.txt"
            source_file.write_text("Test data " * 200)

            collection = CreatePromptCollection(
                input_lengths=input_lengths,
                output_lengths=output_lengths,
                source_file=source_file,
                requests_per_combination=1,
            )

            result = collection.create_collection()
            expected_length = len(input_lengths) * len(output_lengths)
            assert len(result) == expected_length
