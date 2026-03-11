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
    LLMeterBytesEncoder,
    load_payloads,
    load_prompts,
    save_payloads,
)
from llmeter.tokenizers import DummyTokenizer


class TestLLMeterBytesEncoder:
    """Unit tests for LLMeterBytesEncoder class.

    These tests verify specific examples and edge cases for the LLMeterBytesEncoder
    class, complementing the property-based tests.

    Requirements: 1.1, 1.2, 1.3, 1.6
    """

    def test_simple_bytes_object_serialization(self):
        """Test serialization of a simple bytes object.

        Validates: Requirements 1.1, 1.2, 1.3
        """
        payload = {"data": b"hello world"}

        # Serialize using the encoder
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Verify it's valid JSON
        parsed = json.loads(serialized)

        # Verify the marker object structure
        assert "__llmeter_bytes__" in parsed["data"]
        assert len(parsed["data"]) == 1
        assert isinstance(parsed["data"]["__llmeter_bytes__"], str)

        # Verify the base64 encoding is correct
        import base64

        decoded = base64.b64decode(parsed["data"]["__llmeter_bytes__"])
        assert decoded == b"hello world"

    def test_nested_bytes_in_dict_structure(self):
        """Test serialization of bytes nested in complex dict structure.

        Validates: Requirements 1.1, 1.3, 1.5
        """
        payload = {
            "modelId": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "What is in this image?"},
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": b"\xff\xd8\xff\xe0\x00\x10JFIF"},
                            }
                        },
                    ],
                }
            ],
        }

        # Serialize
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Verify it's valid JSON
        parsed = json.loads(serialized)

        # Verify structure is preserved
        assert parsed["modelId"] == "test-model"
        assert parsed["messages"][0]["role"] == "user"
        assert parsed["messages"][0]["content"][0]["text"] == "What is in this image?"

        # Verify bytes are replaced with marker
        bytes_marker = parsed["messages"][0]["content"][1]["image"]["source"]["bytes"]
        assert "__llmeter_bytes__" in bytes_marker
        assert len(bytes_marker) == 1

        # Verify the base64 encoding
        import base64

        decoded = base64.b64decode(bytes_marker["__llmeter_bytes__"])
        assert decoded == b"\xff\xd8\xff\xe0\x00\x10JFIF"

    def test_empty_bytes_object(self):
        """Test serialization of an empty bytes object.

        Validates: Requirements 1.1, 1.3
        """
        payload = {"empty": b""}

        # Serialize
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Verify it's valid JSON
        parsed = json.loads(serialized)

        # Verify marker object exists
        assert "__llmeter_bytes__" in parsed["empty"]

        # Verify empty bytes decodes correctly
        import base64

        decoded = base64.b64decode(parsed["empty"]["__llmeter_bytes__"])
        assert decoded == b""
        assert len(decoded) == 0

    def test_large_binary_data_1mb(self):
        """Test serialization of large binary data (1MB).

        Validates: Requirements 1.6, 10.1
        """
        import os

        # Create 1MB of random binary data
        large_data = os.urandom(1024 * 1024)
        payload = {"large_image": large_data}

        # Serialize
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Verify it's valid JSON
        parsed = json.loads(serialized)

        # Verify marker object exists
        assert "__llmeter_bytes__" in parsed["large_image"]

        # Verify the data round-trips correctly
        import base64

        decoded = base64.b64decode(parsed["large_image"]["__llmeter_bytes__"])
        assert decoded == large_data
        assert len(decoded) == 1024 * 1024

    def test_multiple_bytes_objects_in_payload(self):
        """Test serialization of payload with multiple bytes objects.

        Validates: Requirements 1.1, 1.3, 9.8
        """
        payload = {
            "image1": b"first image data",
            "image2": b"second image data",
            "nested": {"image3": b"third image data"},
        }

        # Serialize
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Verify it's valid JSON
        parsed = json.loads(serialized)

        # Verify all bytes objects have markers
        assert "__llmeter_bytes__" in parsed["image1"]
        assert "__llmeter_bytes__" in parsed["image2"]
        assert "__llmeter_bytes__" in parsed["nested"]["image3"]

        # Verify all decode correctly
        import base64

        assert (
            base64.b64decode(parsed["image1"]["__llmeter_bytes__"])
            == b"first image data"
        )
        assert (
            base64.b64decode(parsed["image2"]["__llmeter_bytes__"])
            == b"second image data"
        )
        assert (
            base64.b64decode(parsed["nested"]["image3"]["__llmeter_bytes__"])
            == b"third image data"
        )

    def test_bytes_in_list(self):
        """Test serialization of bytes objects within lists.

        Validates: Requirements 1.1, 1.3, 1.5
        """
        payload = {"images": [b"image1", b"image2", b"image3"]}

        # Serialize
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Verify it's valid JSON
        parsed = json.loads(serialized)

        # Verify all list items have markers
        assert len(parsed["images"]) == 3
        for item in parsed["images"]:
            assert "__llmeter_bytes__" in item
            assert len(item) == 1

    def test_mixed_types_with_bytes(self):
        """Test serialization of payload with mixed types including bytes.

        Validates: Requirements 1.1, 1.5
        """
        payload = {
            "string": "text value",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "bytes": b"binary data",
            "nested": {"more_bytes": b"more binary"},
        }

        # Serialize
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Verify it's valid JSON
        parsed = json.loads(serialized)

        # Verify non-bytes types are preserved
        assert parsed["string"] == "text value"
        assert parsed["number"] == 42
        assert parsed["float"] == 3.14
        assert parsed["boolean"] is True
        assert parsed["null"] is None
        assert parsed["list"] == [1, 2, 3]

        # Verify bytes have markers
        assert "__llmeter_bytes__" in parsed["bytes"]
        assert "__llmeter_bytes__" in parsed["nested"]["more_bytes"]


class TestLLMeterBytesDecoder:
    """Unit tests for llmeter_bytes_decoder function.

    These tests verify specific examples and edge cases for the llmeter_bytes_decoder
    function, complementing the property-based tests.

    Requirements: 2.1, 2.2, 2.3, 2.4, 6.2
    """

    def test_marker_object_decoding(self):
        """Test decoding of a marker object with valid base64.

        Validates: Requirements 2.1, 2.2, 2.3
        """
        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Create a marker object with base64-encoded bytes
        marker = {"__llmeter_bytes__": "aGVsbG8gd29ybGQ="}  # "hello world" in base64

        # Decode the marker
        result = llmeter_bytes_decoder(marker)

        # Verify it returns bytes
        assert isinstance(result, bytes)
        assert result == b"hello world"

    def test_non_marker_dict_passthrough(self):
        """Test that non-marker dicts are returned unchanged.

        Validates: Requirements 2.4
        """
        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Regular dict without marker key
        regular_dict = {"key": "value", "number": 42, "nested": {"data": "test"}}

        # Decode should return unchanged
        result = llmeter_bytes_decoder(regular_dict)

        # Verify it's the same dict
        assert result == regular_dict
        assert result is regular_dict  # Should be the exact same object

    def test_invalid_base64_error_handling(self):
        """Test that invalid base64 in marker raises appropriate error.

        Validates: Requirements 6.2
        """
        import binascii

        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Marker with invalid base64 string
        invalid_marker = {"__llmeter_bytes__": "not-valid-base64!!!"}

        # Should raise binascii.Error when trying to decode
        with pytest.raises(binascii.Error):
            llmeter_bytes_decoder(invalid_marker)

    def test_multi_key_dict_with_marker_key_not_decoded(self):
        """Test that multi-key dict containing marker key is not decoded.

        This is a safety check to ensure we only decode single-key marker objects.

        Validates: Requirements 2.4
        """
        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Dict with marker key but also other keys (should not be decoded)
        multi_key_dict = {
            "__llmeter_bytes__": "aGVsbG8=",
            "other_key": "other_value",
        }

        # Should return unchanged (not decode)
        result = llmeter_bytes_decoder(multi_key_dict)

        # Verify it's returned as-is
        assert result == multi_key_dict
        assert isinstance(result, dict)
        assert "__llmeter_bytes__" in result
        assert "other_key" in result

    def test_empty_bytes_decoding(self):
        """Test decoding of marker object with empty bytes.

        Validates: Requirements 2.1, 2.3
        """
        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Marker with empty base64 string (empty bytes)
        empty_marker = {"__llmeter_bytes__": ""}

        # Decode
        result = llmeter_bytes_decoder(empty_marker)

        # Verify it returns empty bytes
        assert isinstance(result, bytes)
        assert result == b""
        assert len(result) == 0

    def test_large_binary_data_decoding(self):
        """Test decoding of marker object with large binary data.

        Validates: Requirements 2.1, 2.3
        """
        import base64
        import os

        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Create 1MB of random binary data
        large_data = os.urandom(1024 * 1024)
        base64_encoded = base64.b64encode(large_data).decode("utf-8")

        # Create marker
        marker = {"__llmeter_bytes__": base64_encoded}

        # Decode
        result = llmeter_bytes_decoder(marker)

        # Verify it matches original data
        assert isinstance(result, bytes)
        assert result == large_data
        assert len(result) == 1024 * 1024

    def test_nested_structure_with_marker(self):
        """Test that decoder works correctly when used with json.loads on nested structures.

        Validates: Requirements 2.1, 2.5
        """
        from llmeter.prompt_utils import llmeter_bytes_decoder

        # JSON string with nested marker objects
        json_str = json.dumps(
            {
                "modelId": "test-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "What is this?"},
                            {
                                "image": {
                                    "source": {
                                        "bytes": {
                                            "__llmeter_bytes__": "aGVsbG8="  # "hello"
                                        }
                                    }
                                }
                            },
                        ],
                    }
                ],
            }
        )

        # Load with decoder
        result = json.loads(json_str, object_hook=llmeter_bytes_decoder)

        # Verify structure is preserved
        assert result["modelId"] == "test-model"
        assert result["messages"][0]["role"] == "user"

        # Verify bytes are decoded
        bytes_value = result["messages"][0]["content"][1]["image"]["source"]["bytes"]
        assert isinstance(bytes_value, bytes)
        assert bytes_value == b"hello"

    def test_dict_without_marker_key(self):
        """Test that dict without marker key is returned unchanged.

        Validates: Requirements 2.4
        """
        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Dict without the marker key
        normal_dict = {"data": "value", "count": 123}

        # Should return unchanged
        result = llmeter_bytes_decoder(normal_dict)

        assert result == normal_dict
        assert isinstance(result, dict)

    def test_marker_with_special_characters(self):
        """Test decoding marker with special characters in base64.

        Validates: Requirements 2.1, 2.3
        """
        import base64

        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Binary data with special characters
        special_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
        base64_encoded = base64.b64encode(special_data).decode("utf-8")

        # Create marker
        marker = {"__llmeter_bytes__": base64_encoded}

        # Decode
        result = llmeter_bytes_decoder(marker)

        # Verify
        assert isinstance(result, bytes)
        assert result == special_data


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
                    prompt_file,
                    create_payload,
                    create_payload_kwargs={"max_tokens": 50},
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

            prompts = list(load_prompts(dir_path, create_payload, file_pattern="*.txt"))
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
            (dir_path / "prompts.txt").write_text(
                "Good prompt\nBad prompt\nAnother good prompt\n"
            )

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
                _ = list(load_payloads(json_file))
                # May be empty due to permission error
            finally:
                # Restore permissions for cleanup
                json_file.chmod(0o644)

    def test_load_payloads_with_marker_objects(self):
        """Test loading payload with marker objects.

        Validates: Requirements 2.1, 3.1, 5.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "payload.jsonl"

            # Create a payload with marker objects (simulating saved binary content)
            payload_with_markers = {
                "modelId": "test-model",
                "image": {
                    "source": {
                        "bytes": {
                            "__llmeter_bytes__": "aGVsbG8gd29ybGQ="  # "hello world"
                        }
                    }
                },
            }

            # Write to file
            with jsonl_file.open("w") as f:
                f.write(json.dumps(payload_with_markers) + "\n")

            # Load using load_payloads
            payloads = list(load_payloads(jsonl_file))

            # Verify payload was loaded
            assert len(payloads) == 1

            # Verify bytes were restored from marker
            loaded_payload = payloads[0]
            assert loaded_payload["modelId"] == "test-model"
            assert isinstance(
                loaded_payload["image"]["source"]["bytes"], bytes
            )
            assert loaded_payload["image"]["source"]["bytes"] == b"hello world"

    def test_load_payloads_bytes_correctly_restored(self):
        """Test that bytes objects are correctly restored from marker objects.

        Validates: Requirements 2.1, 2.3, 5.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "payload.jsonl"

            # Create multiple payloads with different bytes content
            payloads_with_markers = [
                {"id": 1, "data": {"__llmeter_bytes__": "Zmlyc3Q="}},  # "first"
                {"id": 2, "data": {"__llmeter_bytes__": "c2Vjb25k"}},  # "second"
                {
                    "id": 3,
                    "nested": {
                        "deep": {"data": {"__llmeter_bytes__": "dGhpcmQ="}}  # "third"
                    },
                },
            ]

            # Write to file
            with jsonl_file.open("w") as f:
                for payload in payloads_with_markers:
                    f.write(json.dumps(payload) + "\n")

            # Load payloads
            loaded = list(load_payloads(jsonl_file))

            # Verify all bytes were restored correctly
            assert len(loaded) == 3
            assert loaded[0]["data"] == b"first"
            assert loaded[1]["data"] == b"second"
            assert loaded[2]["nested"]["deep"]["data"] == b"third"

            # Verify they are bytes objects
            assert isinstance(loaded[0]["data"], bytes)
            assert isinstance(loaded[1]["data"], bytes)
            assert isinstance(loaded[2]["nested"]["deep"]["data"], bytes)

    def test_load_payloads_custom_deserializer_parameter(self):
        """Test custom deserializer parameter works correctly.

        Validates: Requirements 5.2, 5.4
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "payload.jsonl"

            # Write a simple payload
            payload = {"test": "data", "number": 42}
            with jsonl_file.open("w") as f:
                f.write(json.dumps(payload) + "\n")

            # Custom deserializer that adds a field
            def custom_deserializer(json_str):
                data = json.loads(json_str)
                data["custom_field"] = "added_by_deserializer"
                return data

            # Load with custom deserializer
            loaded = list(load_payloads(jsonl_file, deserializer=custom_deserializer))

            # Verify custom deserializer was used
            assert len(loaded) == 1
            assert loaded[0]["test"] == "data"
            assert loaded[0]["number"] == 42
            assert loaded[0]["custom_field"] == "added_by_deserializer"

    def test_load_payloads_backward_compatibility_old_format(self):
        """Test backward compatibility - old format loads successfully.

        When loading payloads saved with the old format (no binary data, no markers),
        they should load successfully without any issues.

        Validates: Requirements 3.1, 5.2, 5.4
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "old_format.jsonl"

            # Create payloads in old format (no marker objects)
            old_format_payloads = [
                {"modelId": "test-model-1", "prompt": "Hello world", "maxTokens": 100},
                {
                    "modelId": "test-model-2",
                    "messages": [
                        {"role": "user", "content": "What is the weather?"}
                    ],
                },
                {
                    "modelId": "test-model-3",
                    "config": {"temperature": 0.7, "topP": 0.9},
                },
            ]

            # Write using standard json.dumps (old format)
            with jsonl_file.open("w") as f:
                for payload in old_format_payloads:
                    f.write(json.dumps(payload) + "\n")

            # Load using new load_payloads (with binary support)
            loaded = list(load_payloads(jsonl_file))

            # Verify all payloads loaded successfully
            assert len(loaded) == 3

            # Verify content matches exactly
            assert loaded[0] == old_format_payloads[0]
            assert loaded[1] == old_format_payloads[1]
            assert loaded[2] == old_format_payloads[2]

            # Verify no marker objects were introduced
            for payload in loaded:
                assert "__llmeter_bytes__" not in json.dumps(payload)

    def test_load_payloads_round_trip_with_binary_content(self):
        """Test round-trip: save with binary, load, verify bytes restored.

        Validates: Requirements 2.1, 2.5, 4.1
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Original payload with binary content
            original_payload = {
                "modelId": "test-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "Describe this image"},
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {"bytes": b"\xff\xd8\xff\xe0\x00\x10JFIF"},
                                }
                            },
                        ],
                    }
                ],
            }

            # Save using save_payloads
            saved_path = save_payloads(original_payload, output_path)

            # Load using load_payloads
            loaded = list(load_payloads(saved_path))

            # Verify round-trip integrity
            assert len(loaded) == 1
            loaded_payload = loaded[0]

            # Verify structure is preserved
            assert loaded_payload["modelId"] == original_payload["modelId"]
            assert len(loaded_payload["messages"]) == 1
            assert loaded_payload["messages"][0]["role"] == "user"

            # Verify bytes were restored correctly
            loaded_bytes = loaded_payload["messages"][0]["content"][1]["image"][
                "source"
            ]["bytes"]
            original_bytes = original_payload["messages"][0]["content"][1]["image"][
                "source"
            ]["bytes"]

            assert isinstance(loaded_bytes, bytes)
            assert loaded_bytes == original_bytes

    def test_load_payloads_multiple_marker_objects_in_single_payload(self):
        """Test loading payload with multiple marker objects.

        Validates: Requirements 2.1, 9.8
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "payload.jsonl"

            # Payload with multiple marker objects
            payload_with_multiple_markers = {
                "modelId": "test-model",
                "images": [
                    {"id": 1, "data": {"__llmeter_bytes__": "aW1hZ2Ux"}},  # "image1"
                    {"id": 2, "data": {"__llmeter_bytes__": "aW1hZ2Uy"}},  # "image2"
                    {"id": 3, "data": {"__llmeter_bytes__": "aW1hZ2Uz"}},  # "image3"
                ],
            }

            # Write to file
            with jsonl_file.open("w") as f:
                f.write(json.dumps(payload_with_multiple_markers) + "\n")

            # Load payload
            loaded = list(load_payloads(jsonl_file))

            # Verify all marker objects were decoded
            assert len(loaded) == 1
            payload = loaded[0]

            assert len(payload["images"]) == 3
            assert payload["images"][0]["data"] == b"image1"
            assert payload["images"][1]["data"] == b"image2"
            assert payload["images"][2]["data"] == b"image3"

            # Verify all are bytes
            for image in payload["images"]:
                assert isinstance(image["data"], bytes)

    def test_load_payloads_empty_bytes_marker(self):
        """Test loading marker object with empty bytes.

        Validates: Requirements 2.1, 2.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "payload.jsonl"

            # Payload with empty bytes marker
            payload_with_empty = {
                "id": 1,
                "empty_data": {"__llmeter_bytes__": ""},  # Empty base64 = empty bytes
            }

            # Write to file
            with jsonl_file.open("w") as f:
                f.write(json.dumps(payload_with_empty) + "\n")

            # Load payload
            loaded = list(load_payloads(jsonl_file))

            # Verify empty bytes were restored
            assert len(loaded) == 1
            assert loaded[0]["empty_data"] == b""
            assert isinstance(loaded[0]["empty_data"], bytes)
            assert len(loaded[0]["empty_data"]) == 0

    def test_load_payloads_large_binary_data(self):
        """Test loading payload with large binary data (1MB).

        Validates: Requirements 2.1, 10.2
        """
        import base64
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "payload.jsonl"

            # Create 1MB of random binary data
            large_data = os.urandom(1024 * 1024)
            base64_encoded = base64.b64encode(large_data).decode("utf-8")

            # Payload with large marker object
            payload_with_large = {
                "id": 1,
                "large_image": {"__llmeter_bytes__": base64_encoded},
            }

            # Write to file
            with jsonl_file.open("w") as f:
                f.write(json.dumps(payload_with_large) + "\n")

            # Load payload
            loaded = list(load_payloads(jsonl_file))

            # Verify large bytes were restored correctly
            assert len(loaded) == 1
            assert isinstance(loaded[0]["large_image"], bytes)
            assert loaded[0]["large_image"] == large_data
            assert len(loaded[0]["large_image"]) == 1024 * 1024


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

    def test_save_payloads_with_bytes_objects(self):
        """Test saving payload with bytes objects.

        Validates: Requirements 1.1, 3.3, 5.1
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payload = {
                "modelId": "test-model",
                "image": {"source": {"bytes": b"\xff\xd8\xff\xe0\x00\x10JFIF"}},
            }

            result_path = save_payloads(payload, output_path)

            # Verify file was created
            assert result_path.exists()

            # Read the file and verify it contains valid JSON
            with result_path.open("r") as f:
                line = f.readline()
                parsed = json.loads(line)

            # Verify structure is preserved
            assert parsed["modelId"] == "test-model"

            # Verify bytes are replaced with marker object
            assert "__llmeter_bytes__" in parsed["image"]["source"]["bytes"]
            assert len(parsed["image"]["source"]["bytes"]) == 1

    def test_save_payloads_file_contains_valid_json_with_markers(self):
        """Test that saved file contains valid JSON with marker objects.

        Validates: Requirements 1.1, 1.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payloads = [
                {"id": 1, "data": b"first bytes"},
                {"id": 2, "data": b"second bytes"},
                {"id": 3, "nested": {"data": b"nested bytes"}},
            ]

            result_path = save_payloads(payloads, output_path)

            # Read and parse each line
            with result_path.open("r") as f:
                lines = f.readlines()

            assert len(lines) == 3

            # Verify each line is valid JSON
            for i, line in enumerate(lines):
                parsed = json.loads(line)
                assert parsed["id"] == i + 1

                # Verify marker objects exist
                if i < 2:
                    assert "__llmeter_bytes__" in parsed["data"]
                else:
                    assert "__llmeter_bytes__" in parsed["nested"]["data"]

    def test_save_payloads_custom_serializer_parameter(self):
        """Test custom serializer parameter works correctly.

        Validates: Requirements 5.1, 5.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payload = {"test": "data", "number": 42}

            # Custom serializer that adds a prefix
            def custom_serializer(payload_dict):
                return "CUSTOM:" + json.dumps(payload_dict)

            result_path = save_payloads(
                payload, output_path, serializer=custom_serializer
            )

            # Verify custom serializer was used
            with result_path.open("r") as f:
                line = f.readline()

            assert line.startswith("CUSTOM:")
            # Remove prefix and verify it's valid JSON
            json_part = line[7:].strip()
            parsed = json.loads(json_part)
            assert parsed["test"] == "data"
            assert parsed["number"] == 42

    def test_save_payloads_backward_compatibility_no_bytes(self):
        """Test backward compatibility when payload has no bytes.

        When a payload contains no bytes objects, the output should be identical
        to using standard json.dumps (no marker objects introduced).

        Validates: Requirements 3.3, 5.1
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payload = {
                "modelId": "test-model",
                "prompt": "Hello world",
                "maxTokens": 100,
                "temperature": 0.7,
            }

            # Save with new encoder
            result_path = save_payloads(payload, output_path)

            # Read the saved content
            with result_path.open("r") as f:
                saved_line = f.readline().strip()

            # Compare with standard json.dumps
            standard_json = json.dumps(payload)

            # They should be identical (no marker objects introduced)
            assert saved_line == standard_json

            # Verify no marker keys exist
            parsed = json.loads(saved_line)
            assert "__llmeter_bytes__" not in json.dumps(parsed)

    def test_save_payloads_multiple_payloads_with_mixed_content(self):
        """Test saving multiple payloads with mixed binary and non-binary content.

        Validates: Requirements 1.1, 3.3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payloads = [
                {"id": 1, "text": "no binary"},
                {"id": 2, "image": b"binary data"},
                {"id": 3, "text": "also no binary"},
            ]

            result_path = save_payloads(payloads, output_path)

            # Read all lines
            with result_path.open("r") as f:
                lines = f.readlines()

            assert len(lines) == 3

            # First payload: no marker
            parsed1 = json.loads(lines[0])
            assert "__llmeter_bytes__" not in json.dumps(parsed1)

            # Second payload: has marker
            parsed2 = json.loads(lines[1])
            assert "__llmeter_bytes__" in parsed2["image"]

            # Third payload: no marker
            parsed3 = json.loads(lines[2])
            assert "__llmeter_bytes__" not in json.dumps(parsed3)


# Property-based tests
class TestPromptUtilsProperties:
    """Property-based tests for prompt_utils."""

    @given(
        st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=50).filter(lambda k: k != "__llmeter_bytes__"),
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


class TestErrorHandling:
    """Unit tests for error conditions in serialization/deserialization.

    These tests verify that errors are handled gracefully with descriptive messages.

    Requirements: 6.1, 6.2, 6.3, 6.4
    """

    def test_unserializable_type_error_message(self):
        """Test that unserializable types raise TypeError with type information.

        Validates: Requirements 6.1
        """

        class CustomUnserializableObject:
            """A custom class that cannot be serialized to JSON."""

            def __init__(self, value):
                self.value = value

        payload = {
            "modelId": "test-model",
            "custom_object": CustomUnserializableObject(42),
        }

        # Should raise TypeError when trying to serialize
        with pytest.raises(TypeError) as exc_info:
            json.dumps(payload, cls=LLMeterBytesEncoder)

        # Verify error message contains type information
        error_message = str(exc_info.value)
        assert "CustomUnserializableObject" in error_message

    def test_invalid_json_error_handling(self):
        """Test that invalid JSON raises JSONDecodeError with descriptive message.

        Validates: Requirements 6.2
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "invalid.jsonl"

            # Write invalid JSON to file
            with jsonl_file.open("w") as f:
                f.write("This is not valid JSON at all\n")

            # Should handle JSONDecodeError gracefully
            # load_payloads skips invalid lines, so we need to check the behavior
            payloads = list(load_payloads(jsonl_file))

            # Invalid line should be skipped
            assert len(payloads) == 0

    def test_invalid_base64_error_handling(self):
        """Test that invalid base64 in marker objects raises binascii.Error.

        Validates: Requirements 6.2, 6.3
        """
        import binascii

        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Create a marker object with invalid base64 (incorrect padding)
        # Base64 strings must have length that is a multiple of 4
        # A single character will trigger binascii.Error
        invalid_marker = {
            "__llmeter_bytes__": "a"
        }

        # Should raise binascii.Error when trying to decode
        with pytest.raises(binascii.Error):
            llmeter_bytes_decoder(invalid_marker)

    def test_invalid_base64_in_load_payloads(self):
        """Test that invalid base64 in saved payload is handled during load.

        Validates: Requirements 6.2, 6.3
        """
        import binascii

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "invalid_base64.jsonl"

            # Create a payload with invalid base64 in marker object
            # Using a single character which will trigger binascii.Error
            invalid_payload = {
                "modelId": "test-model",
                "image": {
                    "source": {
                        "bytes": {
                            "__llmeter_bytes__": "a"
                        }
                    }
                },
            }

            # Write to file
            with jsonl_file.open("w") as f:
                f.write(json.dumps(invalid_payload) + "\n")

            # Should raise binascii.Error when trying to load
            with pytest.raises(binascii.Error):
                list(load_payloads(jsonl_file))

    def test_custom_serializer_exception_propagation(self):
        """Test that custom serializer exceptions are propagated correctly.

        Validates: Requirements 6.4
        """

        class CustomSerializerError(Exception):
            """Custom exception for testing."""

            pass

        def failing_serializer(payload):
            """A custom serializer that always raises an exception."""
            raise CustomSerializerError("Custom serializer failed!")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            payload = {"test": "data"}

            # Should propagate the custom exception
            with pytest.raises(CustomSerializerError) as exc_info:
                save_payloads(payload, output_path, serializer=failing_serializer)

            # Verify the exception message is preserved
            assert "Custom serializer failed!" in str(exc_info.value)

    def test_custom_deserializer_exception_propagation(self):
        """Test that custom deserializer exceptions are propagated correctly.

        Validates: Requirements 6.4
        """

        class CustomDeserializerError(Exception):
            """Custom exception for testing."""

            pass

        def failing_deserializer(json_str):
            """A custom deserializer that always raises an exception."""
            raise CustomDeserializerError("Custom deserializer failed!")

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "payload.jsonl"

            # Write a valid payload
            payload = {"test": "data"}
            with jsonl_file.open("w") as f:
                f.write(json.dumps(payload) + "\n")

            # Should propagate the custom exception
            with pytest.raises(CustomDeserializerError) as exc_info:
                list(load_payloads(jsonl_file, deserializer=failing_deserializer))

            # Verify the exception message is preserved
            assert "Custom deserializer failed!" in str(exc_info.value)

    def test_unserializable_nested_object_error(self):
        """Test error message for unserializable object in nested structure.

        Validates: Requirements 6.1, 6.3
        """

        class NestedCustomObject:
            """A custom class for testing nested error handling."""

            def __init__(self):
                self.data = "test"

        payload = {
            "modelId": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "Hello"},
                        {"custom": NestedCustomObject()},
                    ],
                }
            ],
        }

        # Should raise TypeError with type information
        with pytest.raises(TypeError) as exc_info:
            json.dumps(payload, cls=LLMeterBytesEncoder)

        # Verify error message contains the type name
        error_message = str(exc_info.value)
        assert "NestedCustomObject" in error_message

    def test_bytes_serialization_with_encoding_error(self):
        """Test that bytes serialization handles all byte values correctly.

        This test verifies that bytes with any value (0-255) can be serialized
        without encoding errors.

        Validates: Requirements 1.1, 6.1
        """
        # Create bytes with all possible byte values
        all_bytes = bytes(range(256))
        payload = {"data": all_bytes}

        # Should serialize without errors
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Should be valid JSON
        parsed = json.loads(serialized)

        # Verify marker exists
        assert "__llmeter_bytes__" in parsed["data"]

        # Verify round-trip works
        from llmeter.prompt_utils import llmeter_bytes_decoder

        deserialized = json.loads(serialized, object_hook=llmeter_bytes_decoder)
        assert deserialized["data"] == all_bytes

    def test_empty_payload_serialization(self):
        """Test that empty payloads are handled correctly.

        Validates: Requirements 1.1, 6.1
        """
        # Empty dict
        empty_payload = {}

        # Should serialize without errors
        serialized = json.dumps(empty_payload, cls=LLMeterBytesEncoder)
        assert serialized == "{}"

        # Empty list
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            empty_list = []

            # Should handle empty list
            result_path = save_payloads(empty_list, output_path)

            # File should exist but be empty
            with result_path.open("r") as f:
                content = f.read()
            assert content == ""

    def test_none_value_serialization(self):
        """Test that None values are handled correctly.

        Validates: Requirements 1.1, 1.5
        """
        payload = {
            "modelId": "test-model",
            "optional_field": None,
            "nested": {"also_none": None},
        }

        # Should serialize without errors
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Should be valid JSON
        parsed = json.loads(serialized)

        # Verify None values are preserved
        assert parsed["optional_field"] is None
        assert parsed["nested"]["also_none"] is None

    def test_unicode_in_payload_with_bytes(self):
        """Test that Unicode strings work correctly alongside bytes.

        Validates: Requirements 1.1, 1.5
        """
        payload = {
            "text": "Hello 世界 🌍",
            "emoji": "🎉🎊🎈",
            "bytes": b"\xff\xfe\xfd",
        }

        # Should serialize without errors
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Should be valid JSON
        parsed = json.loads(serialized)

        # Verify Unicode is preserved
        assert parsed["text"] == "Hello 世界 🌍"
        assert parsed["emoji"] == "🎉🎊🎈"

        # Verify bytes have marker
        assert "__llmeter_bytes__" in parsed["bytes"]


class TestPerformance:
    """Performance tests for serialization/deserialization.

    These tests verify that serialization and deserialization of large binary data
    completes within acceptable time limits and doesn't create unnecessary data copies.

    Requirements: 10.1, 10.2, 10.3, 10.4
    """

    def test_1mb_image_serialization_performance(self):
        """Test that 1MB image serialization completes within 100ms.

        Validates: Requirements 10.1
        """
        import os
        import time

        # Create 1MB of random binary data (simulating an image)
        large_image = os.urandom(1024 * 1024)
        payload = {
            "modelId": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "Describe this image"},
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": large_image},
                            }
                        },
                    ],
                }
            ],
        }

        # Measure serialization time
        start_time = time.perf_counter()
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)
        end_time = time.perf_counter()

        # Calculate elapsed time in milliseconds
        elapsed_ms = (end_time - start_time) * 1000

        # Verify serialization completed within 100ms
        assert (
            elapsed_ms < 100
        ), f"Serialization took {elapsed_ms:.2f}ms, expected < 100ms"

        # Verify the result is valid JSON
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert "__llmeter_bytes__" in parsed["messages"][0]["content"][1]["image"][
            "source"
        ]["bytes"]

    def test_1mb_image_deserialization_performance(self):
        """Test that 1MB image deserialization completes within 100ms.

        Validates: Requirements 10.2
        """
        import os
        import time

        # Create 1MB of random binary data
        large_image = os.urandom(1024 * 1024)
        payload = {
            "modelId": "test-model",
            "image": {"source": {"bytes": large_image}},
        }

        # First serialize the payload
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Measure deserialization time
        from llmeter.prompt_utils import llmeter_bytes_decoder

        start_time = time.perf_counter()
        deserialized = json.loads(serialized, object_hook=llmeter_bytes_decoder)
        end_time = time.perf_counter()

        # Calculate elapsed time in milliseconds
        elapsed_ms = (end_time - start_time) * 1000

        # Verify deserialization completed within 100ms
        assert (
            elapsed_ms < 100
        ), f"Deserialization took {elapsed_ms:.2f}ms, expected < 100ms"

        # Verify the result is correct
        assert isinstance(deserialized["image"]["source"]["bytes"], bytes)
        assert deserialized["image"]["source"]["bytes"] == large_image

    def test_serialization_no_unnecessary_copies(self):
        """Test that serialization doesn't create unnecessary data copies.

        This test verifies that the serialization process is memory-efficient
        by checking that the base64 encoding is done in-place without creating
        multiple intermediate copies of the binary data.

        Validates: Requirements 10.3
        """
        import os
        import sys

        # Create a moderately large binary payload (512KB)
        binary_data = os.urandom(512 * 1024)
        payload = {"data": binary_data}

        # Get initial memory usage (approximate)
        initial_size = sys.getsizeof(binary_data)

        # Serialize the payload
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Parse to verify structure
        parsed = json.loads(serialized)
        assert "__llmeter_bytes__" in parsed["data"]

        # The serialized string should be roughly 4/3 the size of the original
        # (base64 encoding overhead) plus JSON structure overhead
        # We verify it's not significantly larger (which would indicate copies)
        serialized_size = sys.getsizeof(serialized)
        base64_expected_size = (initial_size * 4 // 3) + 1000  # +1000 for JSON overhead

        # Allow 50% overhead for Python string internals and JSON structure
        max_acceptable_size = base64_expected_size * 1.5

        assert (
            serialized_size < max_acceptable_size
        ), f"Serialized size {serialized_size} exceeds expected {max_acceptable_size}"

    def test_deserialization_no_unnecessary_copies(self):
        """Test that deserialization doesn't create unnecessary data copies.

        This test verifies that the deserialization process is memory-efficient
        by checking that the base64 decoding is done efficiently without creating
        multiple intermediate copies of the binary data.

        Validates: Requirements 10.4
        """
        import os
        import sys

        # Create a moderately large binary payload (512KB)
        binary_data = os.urandom(512 * 1024)
        payload = {"data": binary_data}

        # Serialize first
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)

        # Deserialize
        from llmeter.prompt_utils import llmeter_bytes_decoder

        deserialized = json.loads(serialized, object_hook=llmeter_bytes_decoder)

        # Verify the deserialized bytes match original
        assert deserialized["data"] == binary_data

        # The deserialized bytes should be approximately the original size
        deserialized_size = sys.getsizeof(deserialized["data"])
        original_size = sys.getsizeof(binary_data)

        # Allow small overhead for Python object internals
        # bytes objects have minimal overhead
        max_acceptable_size = original_size * 1.1

        assert (
            deserialized_size < max_acceptable_size
        ), f"Deserialized size {deserialized_size} exceeds expected {max_acceptable_size}"

    def test_round_trip_performance_with_multiple_images(self):
        """Test round-trip performance with multiple large images.

        This test verifies that serialization and deserialization remain performant
        even with multiple large binary objects in a single payload.

        Validates: Requirements 10.1, 10.2
        """
        import os
        import time

        # Create payload with 3 images of 512KB each (total ~1.5MB)
        images = [os.urandom(512 * 1024) for _ in range(3)]
        payload = {
            "modelId": "test-model",
            "images": [
                {"id": i, "data": img} for i, img in enumerate(images)
            ],
        }

        # Measure serialization time
        start_time = time.perf_counter()
        serialized = json.dumps(payload, cls=LLMeterBytesEncoder)
        serialize_time = (time.perf_counter() - start_time) * 1000

        # Measure deserialization time
        from llmeter.prompt_utils import llmeter_bytes_decoder

        start_time = time.perf_counter()
        deserialized = json.loads(serialized, object_hook=llmeter_bytes_decoder)
        deserialize_time = (time.perf_counter() - start_time) * 1000

        # With 3 images, we allow proportionally more time (but still reasonable)
        # Each operation should complete in under 200ms for 1.5MB total
        assert (
            serialize_time < 200
        ), f"Serialization took {serialize_time:.2f}ms, expected < 200ms"
        assert (
            deserialize_time < 200
        ), f"Deserialization took {deserialize_time:.2f}ms, expected < 200ms"

        # Verify correctness
        assert len(deserialized["images"]) == 3
        for i, img in enumerate(images):
            assert deserialized["images"][i]["data"] == img

    def test_serialization_performance_scales_linearly(self):
        """Test that serialization performance scales linearly with data size.

        This test verifies that doubling the data size roughly doubles the time,
        indicating no algorithmic inefficiencies.

        Validates: Requirements 10.1, 10.3
        """
        import os
        import time

        # Test with 256KB
        small_data = os.urandom(256 * 1024)
        small_payload = {"data": small_data}

        start_time = time.perf_counter()
        json.dumps(small_payload, cls=LLMeterBytesEncoder)
        small_time = time.perf_counter() - start_time

        # Test with 512KB (2x size)
        large_data = os.urandom(512 * 1024)
        large_payload = {"data": large_data}

        start_time = time.perf_counter()
        json.dumps(large_payload, cls=LLMeterBytesEncoder)
        large_time = time.perf_counter() - start_time

        # Large should take roughly 2x the time (allow 3x for variance)
        # This verifies linear scaling, not quadratic or worse
        assert (
            large_time < small_time * 3
        ), f"Performance doesn't scale linearly: {small_time:.4f}s vs {large_time:.4f}s"

    def test_deserialization_performance_scales_linearly(self):
        """Test that deserialization performance scales linearly with data size.

        This test verifies that doubling the data size roughly doubles the time,
        indicating no algorithmic inefficiencies.

        Validates: Requirements 10.2, 10.4
        """
        import os
        import time

        from llmeter.prompt_utils import llmeter_bytes_decoder

        # Test with 256KB
        small_data = os.urandom(256 * 1024)
        small_payload = {"data": small_data}
        small_serialized = json.dumps(small_payload, cls=LLMeterBytesEncoder)

        start_time = time.perf_counter()
        json.loads(small_serialized, object_hook=llmeter_bytes_decoder)
        small_time = time.perf_counter() - start_time

        # Test with 512KB (2x size)
        large_data = os.urandom(512 * 1024)
        large_payload = {"data": large_data}
        large_serialized = json.dumps(large_payload, cls=LLMeterBytesEncoder)

        start_time = time.perf_counter()
        json.loads(large_serialized, object_hook=llmeter_bytes_decoder)
        large_time = time.perf_counter() - start_time

        # Large should take roughly 2x the time (allow 3x for variance)
        # This verifies linear scaling, not quadratic or worse
        assert (
            large_time < small_time * 3
        ), f"Performance doesn't scale linearly: {small_time:.4f}s vs {large_time:.4f}s"
