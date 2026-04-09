# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import tempfile
from pathlib import Path

import pytest

from llmeter.endpoints.openai import OpenAIEndpoint
from llmeter.prompt_utils import (
    AudioContent,
    DocumentContent,
    ImageContent,
    VideoContent,
)


class TestOpenAIMultiModal:
    """Test multi-modal functionality for OpenAI endpoints using ContentItem API."""

    def test_create_payload_single_image_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            temp_path = f.name

        try:
            payload = OpenAIEndpoint.create_payload(
                [ImageContent.from_path(temp_path), "What's in this image?"],
                max_tokens=256,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "image_url"
            assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            b64 = content[0]["image_url"]["url"].split(",", 1)[1]
            assert base64.b64decode(b64) == b"\xff\xd8\xff\xe0"
            assert content[1] == {"type": "text", "text": "What's in this image?"}
        finally:
            Path(temp_path).unlink()

    def test_create_payload_mixed_content(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
            img_file.write(b"\xff\xd8\xff\xe0")
            img_path = img_file.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as doc_file:
            doc_file.write(b"%PDF-1.4")
            doc_path = doc_file.name

        try:
            payload = OpenAIEndpoint.create_payload(
                [
                    ImageContent.from_path(img_path),
                    "Analyze this",
                    DocumentContent.from_path(doc_path),
                ],
                max_tokens=1024,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 3
            assert content[0]["type"] == "image_url"
            assert content[1] == {"type": "text", "text": "Analyze this"}
            assert content[2]["type"] == "file"
            assert content[2]["file"]["file_data"].startswith(
                "data:application/pdf;base64,"
            )
        finally:
            Path(img_path).unlink()
            Path(doc_path).unlink()

    def test_create_payload_audio_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            temp_path = f.name

        try:
            payload = OpenAIEndpoint.create_payload(
                ["Transcribe this", AudioContent.from_path(temp_path)],
                max_tokens=256,
            )

            content = payload["messages"][0]["content"]
            assert len(content) == 2
            assert content[1]["type"] == "input_audio"
            assert content[1]["input_audio"]["format"] == "wav"
            base64.b64decode(content[1]["input_audio"]["data"])  # valid base64
        finally:
            Path(temp_path).unlink()

    def test_create_payload_video_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x18ftypmp42")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="does not support inline video"):
                OpenAIEndpoint.create_payload(
                    [VideoContent.from_path(temp_path), "Describe this"],
                    max_tokens=256,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_payload_text_only(self):
        payload = OpenAIEndpoint.create_payload("Hello, world!", max_tokens=256)
        assert payload["messages"][0]["content"] == "Hello, world!"

    def test_create_payload_ordering_preserved(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            img_path = f.name

        try:
            payload = OpenAIEndpoint.create_payload(
                [
                    "Before",
                    ImageContent.from_path(img_path),
                    "After",
                ],
            )
            content = payload["messages"][0]["content"]
            assert len(content) == 3
            assert content[0] == {"type": "text", "text": "Before"}
            assert content[1]["type"] == "image_url"
            assert content[2] == {"type": "text", "text": "After"}
        finally:
            Path(img_path).unlink()

    def test_create_payload_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            OpenAIEndpoint.create_payload([])

    def test_create_payload_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be str or MediaContent"):
            OpenAIEndpoint.create_payload([123])  # type: ignore

    def test_create_payload_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            OpenAIEndpoint.create_payload(
                [ImageContent.from_path("/nonexistent/file.jpg")]
            )
