# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

from llmeter.endpoints.bedrock import BedrockBase
from llmeter.endpoints.openai import OpenAIEndpoint
from llmeter.endpoints.sagemaker import SageMakerEndpoint
from llmeter.prompt_utils import (
    DocumentContent,
    ImageContent,
    load_payloads,
    load_prompts,
    save_payloads,
)


class TestMultiModalSerialization:
    """Test serialization and deserialization of multi-modal payloads."""

    def test_save_and_load_single_image_payload(self):
        """Test saving and loading a payload with a single image."""
        # Create a payload with image bytes
        image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_image_path = f.name

        try:
            payload = BedrockBase.create_payload(
                [ImageContent.from_path(temp_image_path), "What's in this image?"],
                max_tokens=256,
            )

            # Save payload
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = save_payloads(payload, temp_dir, "test_payload.jsonl")
                assert output_path.exists()

                # Load payload
                loaded_payloads = list(load_payloads(output_path))
                assert len(loaded_payloads) == 1

                loaded_payload = loaded_payloads[0]

                # Verify structure
                assert "messages" in loaded_payload
                content = loaded_payload["messages"][0]["content"]
                assert len(content) == 2  # image + text

                # Verify binary content is preserved
                loaded_image_bytes = content[0]["image"]["source"]["bytes"]
                assert isinstance(loaded_image_bytes, bytes)
                assert loaded_image_bytes == image_bytes

        finally:
            Path(temp_image_path).unlink()

    def test_save_and_load_multiple_content_types(self):
        """Test saving and loading a payload with multiple content types."""
        # Create test files
        image_bytes = b"\xff\xd8\xff\xe0"
        pdf_bytes = b"%PDF-1.4\n"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
            img_file.write(image_bytes)
            img_path = img_file.name

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as doc_file:
            doc_file.write(pdf_bytes)
            doc_path = doc_file.name

        try:
            payload = BedrockBase.create_payload(
                [
                    ImageContent.from_path(img_path),
                    "Analyze this",
                    DocumentContent.from_path(doc_path),
                ],
                max_tokens=1024,
            )

            # Save and load
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = save_payloads(payload, temp_dir)
                loaded_payloads = list(load_payloads(output_path))

                assert len(loaded_payloads) == 1
                loaded_payload = loaded_payloads[0]

                # Verify all content is preserved
                content = loaded_payload["messages"][0]["content"]
                assert len(content) == 3  # image + text + document

                # Verify binary content
                loaded_image = content[0]["image"]["source"]["bytes"]
                loaded_doc = content[2]["document"]["source"]["bytes"]

                assert isinstance(loaded_image, bytes)
                assert isinstance(loaded_doc, bytes)
                assert loaded_image == image_bytes
                assert loaded_doc == pdf_bytes

        finally:
            Path(img_path).unlink()
            Path(doc_path).unlink()

    def test_round_trip_preservation(self):
        """Test that binary content is preserved byte-for-byte in round-trip."""
        # Create payload with bytes directly
        image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            original_payload = BedrockBase.create_payload(
                [ImageContent.from_path(temp_path), "Test"], max_tokens=256
            )

            # Save and load
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = save_payloads(original_payload, temp_dir)
                loaded_payload = list(load_payloads(output_path))[0]

                # Verify exact equality
                assert original_payload == loaded_payload

        finally:
            Path(temp_path).unlink()

    def test_save_multiple_payloads(self):
        """Test saving and loading multiple payloads."""
        image_bytes = b"\xff\xd8\xff\xe0"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            payloads = [
                BedrockBase.create_payload(
                    [ImageContent.from_path(temp_path), f"Image {i}"], max_tokens=256
                )
                for i in range(3)
            ]

            # Save and load
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = save_payloads(payloads, temp_dir)
                loaded_payloads = list(load_payloads(output_path))

                assert len(loaded_payloads) == 3

                # Verify each payload
                for i, loaded in enumerate(loaded_payloads):
                    content = loaded["messages"][0]["content"]
                    assert content[1]["text"] == f"Image {i}"
                    assert isinstance(content[0]["image"]["source"]["bytes"], bytes)

        finally:
            Path(temp_path).unlink()

    def test_load_prompts_with_multimodal_create_payload(self):
        """Test load_prompts integration with multi-modal create_payload."""
        # Create a prompts file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("What is this?\n")
            f.write("Describe the image\n")
            prompts_path = f.name

        # Create an image file
        image_bytes = b"\xff\xd8\xff\xe0"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_file:
            img_file.write(image_bytes)
            img_path = img_file.name

        try:
            # Define a create_payload function that includes an image
            def create_multimodal_payload(input_text, **kwargs):
                return BedrockBase.create_payload(
                    [ImageContent.from_path(img_path), input_text],
                    max_tokens=256,
                    **kwargs,
                )

            # Load prompts with multi-modal payload creation
            payloads = list(load_prompts(prompts_path, create_multimodal_payload))

            assert len(payloads) == 2

            # Verify each payload has the image
            for payload in payloads:
                content = payload["messages"][0]["content"]
                assert len(content) == 2  # image + text
                assert "image" in content[0]
                assert isinstance(content[0]["image"]["source"]["bytes"], bytes)

        finally:
            Path(prompts_path).unlink()
            Path(img_path).unlink()

    def test_openai_payload_serialization(self):
        """Test serialization of OpenAI multi-modal payloads."""
        image_bytes = b"\xff\xd8\xff\xe0"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            payload = OpenAIEndpoint.create_payload(
                [ImageContent.from_path(temp_path), "Test"], max_tokens=256
            )

            # Save and load
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = save_payloads(payload, temp_dir)
                loaded_payload = list(load_payloads(output_path))[0]

                # Verify exact equality
                assert payload == loaded_payload

                # Verify OpenAI-specific format (image_url content part)
                content = loaded_payload["messages"][0]["content"]
                assert content[0]["type"] == "image_url"
                assert content[0]["image_url"]["url"].startswith(
                    "data:image/jpeg;base64,"
                )

        finally:
            Path(temp_path).unlink()

    def test_sagemaker_payload_serialization(self):
        """Test serialization of SageMaker multi-modal payloads."""
        image_bytes = b"\xff\xd8\xff\xe0"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            payload = SageMakerEndpoint.create_payload(
                [ImageContent.from_path(temp_path), "Test"], max_tokens=256
            )

            # Save and load
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = save_payloads(payload, temp_dir)
                loaded_payload = list(load_payloads(output_path))[0]

                # Verify exact equality
                assert payload == loaded_payload

                # Verify SageMaker-specific format
                content = loaded_payload["inputs"]
                assert content[0]["image"]["format"] == "jpeg"

        finally:
            Path(temp_path).unlink()
