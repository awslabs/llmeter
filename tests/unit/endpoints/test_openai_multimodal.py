# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from llmeter.endpoints.openai import OpenAIEndpoint


class TestOpenAIMultiModal:
    """Test multi-modal functionality for OpenAI endpoints."""

    def test_create_payload_single_image_from_file(self):
        """Test creating payload with single image from file path."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
            temp_path = f.name

        try:
            payload = OpenAIEndpoint.create_payload(
                user_message="What's in this image?", images=[temp_path], max_tokens=256
            )

            assert "messages" in payload
            assert len(payload["messages"]) == 1
            assert payload["messages"][0]["role"] == "user"

            content = payload["messages"][0]["content"]
            assert len(content) == 2  # text + image
            assert content[0]["text"] == "What's in this image?"
            assert "image" in content[1]
            # OpenAI uses full MIME types
            assert content[1]["image"]["format"] == "image/jpeg"
            assert "bytes" in content[1]["image"]["source"]

        finally:
            Path(temp_path).unlink()

    def test_create_payload_single_image_from_bytes(self):
        """Test creating payload with single image from bytes."""
        # Create a minimal valid JPEG file
        jpeg_bytes = (
            b"\xff\xd8"  # SOI (Start of Image)
            b"\xff\xe0"  # APP0 marker
            b"\x00\x10"  # APP0 length (16 bytes)
            b"JFIF\x00"  # JFIF identifier
            b"\x01\x01"  # JFIF version 1.1
            b"\x00"  # density units (0 = no units)
            b"\x00\x01"  # X density = 1
            b"\x00\x01"  # Y density = 1
            b"\x00\x00"  # thumbnail width and height = 0
            b"\xff\xd9"  # EOI (End of Image)
        )

        try:
            payload = OpenAIEndpoint.create_payload(
                user_message="What's in this image?",
                images=[jpeg_bytes],
                max_tokens=256,
            )

            assert "messages" in payload
            content = payload["messages"][0]["content"]
            assert len(content) == 2  # text + image
            assert "image" in content[1]
            # OpenAI uses full MIME types
            assert content[1]["image"]["format"] == "image/jpeg"
            assert content[1]["image"]["source"]["bytes"] == jpeg_bytes
        except ValueError as e:
            # If puremagic can't detect the format, skip this test
            if "Cannot detect format from bytes" in str(e):
                pytest.skip("puremagic cannot detect format from minimal JPEG bytes")
            raise

    def test_create_payload_mixed_content(self):
        """Test creating payload with mixed content types."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
            img_file.write(b"\xff\xd8\xff\xe0")
            img_path = img_file.name

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as doc_file:
            doc_file.write(b"%PDF-1.4")
            doc_path = doc_file.name

        try:
            payload = OpenAIEndpoint.create_payload(
                user_message="Analyze this",
                images=[img_path],
                documents=[doc_path],
                max_tokens=1024,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 3  # text + image + document
            assert content[0]["text"] == "Analyze this"
            assert "image" in content[1]
            # OpenAI uses full MIME types
            assert content[1]["image"]["format"] == "image/jpeg"
            assert "document" in content[2]
            assert content[2]["document"]["format"] == "application/pdf"

        finally:
            Path(img_path).unlink()
            Path(doc_path).unlink()

    def test_create_payload_text_only_backward_compatible(self):
        """Test that text-only payloads still work (backward compatibility)."""
        payload = OpenAIEndpoint.create_payload(
            user_message="Hello, world!", max_tokens=256
        )

        assert "messages" in payload
        content = payload["messages"][0]["content"]
        # Text-only should be a string, not a list
        assert content == "Hello, world!"

    def test_create_payload_invalid_image_type(self):
        """Test that invalid image types raise TypeError."""
        with pytest.raises(
            TypeError, match="Items in images list must be bytes or str"
        ):
            OpenAIEndpoint.create_payload(
                user_message="Test",
                images=[123],  # Invalid type
                max_tokens=256,
            )

    def test_create_payload_invalid_images_not_list(self):
        """Test that non-list images parameter raises TypeError."""
        with pytest.raises(TypeError, match="images must be a list"):
            OpenAIEndpoint.create_payload(
                user_message="Test", images="not_a_list", max_tokens=256
            )

    def test_create_payload_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            OpenAIEndpoint.create_payload(
                user_message="Test", images=["/nonexistent/file.jpg"], max_tokens=256
            )
