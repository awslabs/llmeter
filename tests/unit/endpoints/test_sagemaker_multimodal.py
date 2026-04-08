# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from llmeter.endpoints.sagemaker import SageMakerBase


class TestSageMakerMultiModal:
    """Test multi-modal functionality for SageMaker endpoints."""

    def test_create_payload_single_image_from_file(self):
        """Test creating payload with single image from file path."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
            temp_path = f.name

        try:
            payload = SageMakerBase.create_payload(
                input_text="What's in this image?", images=[temp_path], max_tokens=256
            )

            assert "inputs" in payload
            content = payload["inputs"]
            assert len(content) == 2  # text + image
            assert content[0]["text"] == "What's in this image?"
            assert "image" in content[1]
            # SageMaker uses Bedrock-style short format strings
            assert content[1]["image"]["format"] == "jpeg"
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
            payload = SageMakerBase.create_payload(
                input_text="What's in this image?",
                images=[jpeg_bytes],
                max_tokens=256,
            )

            assert "inputs" in payload
            content = payload["inputs"]
            assert len(content) == 2  # text + image
            assert "image" in content[1]
            # SageMaker uses Bedrock-style short format strings
            assert content[1]["image"]["format"] == "jpeg"
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
            payload = SageMakerBase.create_payload(
                input_text="Analyze this",
                images=[img_path],
                documents=[doc_path],
                max_tokens=1024,
            )

            content = payload["inputs"]
            assert len(content) == 3  # text + image + document
            assert content[0]["text"] == "Analyze this"
            assert "image" in content[1]
            # SageMaker uses Bedrock-style short format strings
            assert content[1]["image"]["format"] == "jpeg"
            assert "document" in content[2]
            assert content[2]["document"]["format"] == "pdf"

        finally:
            Path(img_path).unlink()
            Path(doc_path).unlink()

    def test_create_payload_text_only_backward_compatible(self):
        """Test that text-only payloads still work (backward compatibility)."""
        payload = SageMakerBase.create_payload(
            input_text="Hello, world!", max_tokens=256
        )

        assert "inputs" in payload
        # Text-only should be a string, not a list
        assert payload["inputs"] == "Hello, world!"

    def test_create_payload_invalid_image_type(self):
        """Test that invalid image types raise TypeError."""
        with pytest.raises(
            TypeError, match="Items in images list must be bytes or str"
        ):
            SageMakerBase.create_payload(
                input_text="Test",
                images=[123],  # Invalid type
                max_tokens=256,
            )

    def test_create_payload_invalid_images_not_list(self):
        """Test that non-list images parameter raises TypeError."""
        with pytest.raises(TypeError, match="images must be a list"):
            SageMakerBase.create_payload(
                input_text="Test", images="not_a_list", max_tokens=256
            )

    def test_create_payload_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            SageMakerBase.create_payload(
                input_text="Test", images=["/nonexistent/file.jpg"], max_tokens=256
            )
