# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import tempfile
from pathlib import Path

import pytest

from llmeter.endpoints.openai import OpenAIEndpoint


class TestOpenAIMultiModal:
    """Test multi-modal functionality for OpenAI endpoints."""

    def test_create_payload_single_image_from_file(self):
        """Test creating payload with single image from file path."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
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
            assert content[0] == {"type": "text", "text": "What's in this image?"}

            img_block = content[1]
            assert img_block["type"] == "image_url"
            url = img_block["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")
            # Verify round-trip of the bytes
            b64_data = url.split(",", 1)[1]
            assert base64.b64decode(b64_data) == b"\xff\xd8\xff\xe0"
        finally:
            Path(temp_path).unlink()

    def test_create_payload_single_image_from_bytes(self):
        """Test creating payload with single image from bytes."""
        jpeg_bytes = (
            b"\xff\xd8"  # SOI
            b"\xff\xe0"  # APP0 marker
            b"\x00\x10"  # APP0 length
            b"JFIF\x00"  # JFIF identifier
            b"\x01\x01"  # version
            b"\x00"  # density units
            b"\x00\x01"  # X density
            b"\x00\x01"  # Y density
            b"\x00\x00"  # thumbnail
            b"\xff\xd9"  # EOI
        )

        try:
            payload = OpenAIEndpoint.create_payload(
                user_message="What's in this image?",
                images=[jpeg_bytes],
                max_tokens=256,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 2
            img_block = content[1]
            assert img_block["type"] == "image_url"
            url = img_block["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")
            b64_data = url.split(",", 1)[1]
            assert base64.b64decode(b64_data) == jpeg_bytes
        except ValueError as e:
            if "Cannot detect format from bytes" in str(e):
                pytest.skip("puremagic cannot detect format from minimal JPEG bytes")
            raise

    def test_create_payload_mixed_content(self):
        """Test creating payload with image + PDF document."""
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
            assert content[0] == {"type": "text", "text": "Analyze this"}

            # Image block
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

            # Document block (PDF → file content part)
            assert content[2]["type"] == "file"
            assert content[2]["file"]["file_data"].startswith(
                "data:application/pdf;base64,"
            )
            assert "filename" in content[2]["file"]
        finally:
            Path(img_path).unlink()
            Path(doc_path).unlink()

    def test_create_payload_audio_from_file(self):
        """Test creating payload with audio from file path."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            temp_path = f.name

        try:
            payload = OpenAIEndpoint.create_payload(
                user_message="Transcribe this",
                audio=[temp_path],
                max_tokens=256,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 2
            audio_block = content[1]
            assert audio_block["type"] == "input_audio"
            assert audio_block["input_audio"]["format"] == "wav"
            assert isinstance(audio_block["input_audio"]["data"], str)
            # Verify it's valid base64
            base64.b64decode(audio_block["input_audio"]["data"])
        finally:
            Path(temp_path).unlink()

    def test_create_payload_audio_mp3_from_file(self):
        """Test creating payload with MP3 audio."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\xff\xfb\x90\x00")  # MP3 frame header
            temp_path = f.name

        try:
            payload = OpenAIEndpoint.create_payload(
                user_message="What is said?",
                audio=[temp_path],
                max_tokens=256,
            )

            content = payload["messages"][0]["content"]
            audio_block = content[1]
            assert audio_block["type"] == "input_audio"
            assert audio_block["input_audio"]["format"] == "mp3"
        finally:
            Path(temp_path).unlink()

    def test_create_payload_video_raises(self):
        """Test that video content raises ValueError (not supported by OpenAI inline)."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x18ftypmp42")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="does not support inline video"):
                OpenAIEndpoint.create_payload(
                    user_message="Describe this",
                    videos=[temp_path],
                    max_tokens=256,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_payload_text_only_backward_compatible(self):
        """Test that text-only payloads still work (backward compatibility)."""
        payload = OpenAIEndpoint.create_payload(
            user_message="Hello, world!", max_tokens=256
        )

        assert "messages" in payload
        content = payload["messages"][0]["content"]
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
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            OpenAIEndpoint.create_payload(
                user_message="Test", images=["/nonexistent/file.jpg"], max_tokens=256
            )
