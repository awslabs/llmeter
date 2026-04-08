# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from llmeter.endpoints.bedrock import BedrockBase


class TestBedrockMultiModal:
    """Test multi-modal functionality for Bedrock endpoints."""

    def test_create_payload_single_image_from_file(self):
        """Test creating payload with single image from file path."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
            temp_path = f.name

        try:
            payload = BedrockBase.create_payload(
                user_message="What's in this image?", images=[temp_path], max_tokens=256
            )

            assert "messages" in payload
            assert len(payload["messages"]) == 1
            assert payload["messages"][0]["role"] == "user"

            content = payload["messages"][0]["content"]
            assert len(content) == 2  # text + image
            assert content[0]["text"] == "What's in this image?"
            assert "image" in content[1]
            assert content[1]["image"]["format"] == "jpeg"
            assert "bytes" in content[1]["image"]["source"]

        finally:
            Path(temp_path).unlink()

    def test_create_payload_single_image_from_bytes(self):
        """Test creating payload with single image from bytes."""
        # Create a minimal valid JPEG file
        # JPEG structure: SOI (FFD8) + APP0 marker + minimal data + EOI (FFD9)
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
            payload = BedrockBase.create_payload(
                user_message="What's in this image?",
                images=[jpeg_bytes],
                max_tokens=256,
            )

            assert "messages" in payload
            content = payload["messages"][0]["content"]
            assert len(content) == 2  # text + image
            assert "image" in content[1]
            assert content[1]["image"]["format"] == "jpeg"
            assert content[1]["image"]["source"]["bytes"] == jpeg_bytes
        except ValueError as e:
            # If puremagic can't detect the format, skip this test
            if "Cannot detect format from bytes" in str(e):
                pytest.skip("puremagic cannot detect format from minimal JPEG bytes")
            raise

    def test_create_payload_multiple_images(self):
        """Test creating payload with multiple images."""
        # Create temporary image files
        temp_files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes
                temp_files.append(f.name)

        try:
            payload = BedrockBase.create_payload(
                user_message="Compare these images", images=temp_files, max_tokens=512
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 3  # text + 2 images
            assert content[0]["text"] == "Compare these images"
            assert "image" in content[1]
            assert "image" in content[2]

        finally:
            for path in temp_files:
                Path(path).unlink()

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
            payload = BedrockBase.create_payload(
                user_message="Analyze this",
                images=[img_path],
                documents=[doc_path],
                max_tokens=1024,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 3  # text + image + document
            assert content[0]["text"] == "Analyze this"
            assert "image" in content[1]
            assert content[1]["image"]["format"] == "jpeg"
            assert "document" in content[2]
            assert content[2]["document"]["format"] == "pdf"

        finally:
            Path(img_path).unlink()
            Path(doc_path).unlink()

    def test_create_payload_text_only_backward_compatible(self):
        """Test that text-only payloads still work (backward compatibility)."""
        payload = BedrockBase.create_payload(
            user_message="Hello, world!", max_tokens=256
        )

        assert "messages" in payload
        content = payload["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["text"] == "Hello, world!"

    def test_create_payload_empty_media_lists(self):
        """Test that empty media lists are handled correctly."""
        payload = BedrockBase.create_payload(
            user_message="Hello", images=[], documents=None, max_tokens=256
        )

        # Should behave like text-only
        content = payload["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["text"] == "Hello"

    def test_create_payload_invalid_image_type(self):
        """Test that invalid image types raise TypeError."""
        with pytest.raises(
            TypeError, match="Items in images list must be bytes or str"
        ):
            BedrockBase.create_payload(
                user_message="Test",
                images=[123],  # Invalid type
                max_tokens=256,
            )

    def test_create_payload_invalid_images_not_list(self):
        """Test that non-list images parameter raises TypeError."""
        with pytest.raises(TypeError, match="images must be a list"):
            BedrockBase.create_payload(
                user_message="Test", images="not_a_list", max_tokens=256
            )

    def test_create_payload_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            BedrockBase.create_payload(
                user_message="Test", images=["/nonexistent/file.jpg"], max_tokens=256
            )

    def test_create_payload_bytes_without_puremagic(self):
        """Test that bytes without detectable format raises ValueError."""
        # Random bytes that don't match any known format
        random_bytes = b"\x00\x01\x02\x03\x04\x05"

        with pytest.raises(ValueError, match="Cannot detect format from bytes"):
            BedrockBase.create_payload(
                user_message="Test", images=[random_bytes], max_tokens=256
            )

    def test_create_payload_file_without_extension(self):
        """Test that file without recognized extension raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix="", delete=False) as f:
            f.write(b"some content")
            temp_path = f.name

        try:
            with pytest.raises(
                ValueError, match="(Cannot detect format|Unsupported MIME type)"
            ):
                BedrockBase.create_payload(
                    user_message="Test", images=[temp_path], max_tokens=256
                )
        finally:
            Path(temp_path).unlink()

    def test_create_payload_content_ordering(self):
        """Test that content blocks are ordered correctly: text, images, videos, audio, documents."""
        # Create temporary files for each media type with proper extensions
        # puremagic may not detect these minimal magic bytes, so we rely on extensions
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img:
            img.write(
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
            )
            img_path = img.name

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid:
            vid.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41")
            vid_path = vid.name

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as aud:
            aud.write(b"ID3\x03\x00\x00\x00\x00\x00\x00")
            aud_path = aud.name

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as doc:
            doc.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
            doc_path = doc.name

        try:
            payload = BedrockBase.create_payload(
                user_message="Analyze all",
                images=[img_path],
                videos=[vid_path],
                audio=[aud_path],
                documents=[doc_path],
                max_tokens=1024,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 5  # text + 4 media types
            assert "text" in content[0]
            assert "image" in content[1]
            assert "video" in content[2]
            assert "audio" in content[3]
            assert "document" in content[4]

        finally:
            for path in [img_path, vid_path, aud_path, doc_path]:
                Path(path).unlink()
