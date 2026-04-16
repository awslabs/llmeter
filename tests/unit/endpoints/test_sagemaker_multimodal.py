# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from llmeter.endpoints.sagemaker import SageMakerBase
from llmeter.prompt_utils import DocumentContent, ImageContent


class TestSageMakerMultiModal:
    """Test multi-modal functionality for SageMaker endpoints using ContentItem API."""

    def test_create_payload_single_image_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            temp_path = f.name

        try:
            payload = SageMakerBase.create_payload(
                [ImageContent.from_path(temp_path), "What's in this image?"],
                max_tokens=256,
            )

            assert "inputs" in payload
            content = payload["inputs"]
            assert len(content) == 2
            assert "image" in content[0]
            assert content[0]["image"]["format"] == "jpeg"
            assert content[1] == {"text": "What's in this image?"}
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
            payload = SageMakerBase.create_payload(
                [
                    ImageContent.from_path(img_path),
                    "Analyze this",
                    DocumentContent.from_path(doc_path),
                ],
                max_tokens=1024,
            )

            content = payload["inputs"]
            assert len(content) == 3
            assert "image" in content[0]
            assert content[1] == {"text": "Analyze this"}
            assert "document" in content[2]
        finally:
            Path(img_path).unlink()
            Path(doc_path).unlink()

    def test_create_payload_text_only(self):
        payload = SageMakerBase.create_payload("Hello, world!", max_tokens=256)
        assert payload["inputs"] == "Hello, world!"

    def test_create_payload_ordering_preserved(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            img_path = f.name

        try:
            payload = SageMakerBase.create_payload(
                [
                    "Before",
                    ImageContent.from_path(img_path),
                    "After",
                ],
            )
            content = payload["inputs"]
            assert len(content) == 3
            assert content[0] == {"text": "Before"}
            assert "image" in content[1]
            assert content[2] == {"text": "After"}
        finally:
            Path(img_path).unlink()

    def test_create_payload_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SageMakerBase.create_payload([])

    def test_create_payload_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be str or MediaContent"):
            SageMakerBase.create_payload([123])  # type: ignore
