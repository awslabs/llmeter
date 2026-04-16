# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from llmeter.endpoints.bedrock import BedrockBase
from llmeter.prompt_utils import DocumentContent, ImageContent


class TestBedrockMultiModal:
    """Test multi-modal functionality for Bedrock endpoints using ContentItem API."""

    def test_create_payload_single_image_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            temp_path = f.name

        try:
            payload = BedrockBase.create_payload(
                [ImageContent.from_path(temp_path), "What's in this image?"],
                max_tokens=256,
            )

            assert "messages" in payload
            assert len(payload["messages"]) == 1
            assert payload["messages"][0]["role"] == "user"

            content = payload["messages"][0]["content"]
            assert len(content) == 2  # image + text
            assert "image" in content[0]
            assert content[0]["image"]["format"] == "jpeg"
            assert "bytes" in content[0]["image"]["source"]
            assert content[1] == {"text": "What's in this image?"}
        finally:
            Path(temp_path).unlink()

    def test_create_payload_multiple_images(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            f1.write(b"\xff\xd8\xff\xe0")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            f2.write(b"\x89PNG\r\n\x1a\n")
            path2 = f2.name

        try:
            payload = BedrockBase.create_payload(
                [
                    "Compare these images:",
                    ImageContent.from_path(path1),
                    ImageContent.from_path(path2),
                ],
                max_tokens=256,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 3
            assert content[0] == {"text": "Compare these images:"}
            assert content[1]["image"]["format"] == "jpeg"
            assert content[2]["image"]["format"] == "png"
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    def test_create_payload_mixed_content(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
            img_file.write(b"\xff\xd8\xff\xe0")
            img_path = img_file.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as doc_file:
            doc_file.write(b"%PDF-1.4")
            doc_path = doc_file.name

        try:
            payload = BedrockBase.create_payload(
                [
                    ImageContent.from_path(img_path),
                    "Analyze this chart with the report:",
                    DocumentContent.from_path(doc_path),
                ],
                max_tokens=1024,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 3
            assert "image" in content[0]
            assert content[0]["image"]["format"] == "jpeg"
            assert content[1] == {"text": "Analyze this chart with the report:"}
            assert "document" in content[2]
            assert content[2]["document"]["format"] == "pdf"
        finally:
            Path(img_path).unlink()
            Path(doc_path).unlink()

    def test_create_payload_text_only(self):
        payload = BedrockBase.create_payload("Hello, world!", max_tokens=256)
        assert "messages" in payload
        content = payload["messages"][0]["content"]
        assert len(content) == 1
        assert content[0] == {"text": "Hello, world!"}

    def test_create_payload_text_list(self):
        payload = BedrockBase.create_payload(["Hello", "World"], max_tokens=256)
        content = payload["messages"][0]["content"]
        assert len(content) == 2
        assert content[0] == {"text": "Hello"}
        assert content[1] == {"text": "World"}

    def test_create_payload_ordering_preserved(self):
        """Content blocks appear in the exact order provided."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            img_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            doc_path = f.name

        try:
            payload = BedrockBase.create_payload(
                [
                    "First text",
                    DocumentContent.from_path(doc_path),
                    "Middle text",
                    ImageContent.from_path(img_path),
                    "Last text",
                ],
            )
            content = payload["messages"][0]["content"]
            assert len(content) == 5
            assert content[0] == {"text": "First text"}
            assert "document" in content[1]
            assert content[2] == {"text": "Middle text"}
            assert "image" in content[3]
            assert content[4] == {"text": "Last text"}
        finally:
            Path(img_path).unlink()
            Path(doc_path).unlink()

    def test_create_payload_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            BedrockBase.create_payload([])

    def test_create_payload_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be str or MediaContent"):
            BedrockBase.create_payload([123])  # type: ignore

    def test_create_payload_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            BedrockBase.create_payload(
                [ImageContent.from_path("/nonexistent/file.jpg")]
            )
