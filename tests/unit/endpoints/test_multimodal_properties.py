# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for multi-modal payload functionality.

This module contains property-based tests using Hypothesis to validate
the correctness properties defined in the multi-modal payload design document.
Each test validates universal properties that should hold across all valid inputs.
"""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from llmeter.endpoints.bedrock import BedrockBase
from llmeter.prompt_utils import (
    LLMeterBytesEncoder,
    llmeter_bytes_decoder,
    load_payloads,
    save_payloads,
)


# Test file content generators
@st.composite
def image_bytes(draw):
    """Generate valid JPEG image bytes with magic bytes."""
    # JPEG magic bytes: FF D8 FF E0
    jpeg_header = b"\xff\xd8\xff\xe0"
    # Add some random content
    content_size = draw(st.integers(min_value=10, max_value=100))
    content = draw(st.binary(min_size=content_size, max_size=content_size))
    # JPEG end marker: FF D9
    jpeg_footer = b"\xff\xd9"
    return jpeg_header + content + jpeg_footer


@st.composite
def png_bytes(draw):
    """Generate valid PNG image bytes with magic bytes."""
    # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
    png_header = b"\x89PNG\r\n\x1a\n"
    content_size = draw(st.integers(min_value=10, max_value=100))
    content = draw(st.binary(min_size=content_size, max_size=content_size))
    return png_header + content


@st.composite
def pdf_bytes(draw):
    """Generate valid PDF bytes with magic bytes."""
    # PDF magic bytes: %PDF-
    pdf_header = b"%PDF-1.4\n"
    content_size = draw(st.integers(min_value=10, max_value=100))
    content = draw(st.binary(min_size=content_size, max_size=content_size))
    pdf_footer = b"\n%%EOF"
    return pdf_header + content + pdf_footer


@st.composite
def mp4_bytes(draw):
    """Generate valid MP4 video bytes with magic bytes."""
    # MP4 magic bytes typically start with ftyp
    mp4_header = b"\x00\x00\x00\x20ftypisom"
    content_size = draw(st.integers(min_value=10, max_value=100))
    content = draw(st.binary(min_size=content_size, max_size=content_size))
    return mp4_header + content


@st.composite
def mp3_bytes(draw):
    """Generate valid MP3 audio bytes with magic bytes."""
    # MP3 magic bytes: ID3 or FF FB
    mp3_header = b"ID3\x03\x00\x00"
    content_size = draw(st.integers(min_value=10, max_value=100))
    content = draw(st.binary(min_size=content_size, max_size=content_size))
    return mp3_header + content


# Strategy for generating media bytes
media_bytes_strategy = st.one_of(
    image_bytes(),
    png_bytes(),
    pdf_bytes(),
    mp4_bytes(),
    mp3_bytes(),
)


# Strategy for generating file extensions
image_extensions = st.sampled_from([".jpg", ".jpeg", ".png", ".gif", ".webp"])
video_extensions = st.sampled_from([".mp4", ".mov", ".avi"])
audio_extensions = st.sampled_from([".mp3", ".wav", ".ogg"])
document_extensions = st.sampled_from([".pdf"])


def create_temp_file(tmp_path: Path, content: bytes, extension: str) -> str:
    """Create a temporary file with given content and extension."""
    file_path = tmp_path / f"test_file_{id(content)}{extension}"
    file_path.write_bytes(content)
    return str(file_path)


# Property-based tests


@given(
    file_content=image_bytes(),
    extension=image_extensions,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_1_file_path_content_inclusion(file_content, extension):
    """Property 1: File path content inclusion.

    **Validates: Requirements 1.2, 2.2, 3.2, 4.2**

    For any media type (image, video, audio, document) and valid file path,
    when creating a payload with that file path in the corresponding parameter list,
    the resulting payload SHALL contain a content block with the file's bytes and
    the format detected from the file extension.
    """
    # Create temporary file and use it within the same context
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        file_path = create_temp_file(tmp_path, file_content, extension)

        # Create payload with the file
        payload = BedrockBase.create_payload(
            user_message="Test message", images=[file_path], max_tokens=256
        )

        # Verify payload structure
        assert "messages" in payload
        assert len(payload["messages"]) == 1
        assert "content" in payload["messages"][0]

        content_blocks = payload["messages"][0]["content"]

        # Find the image content block
        image_blocks = [block for block in content_blocks if "image" in block]
        assert len(image_blocks) == 1

        # Verify the image block contains the file's bytes
        image_block = image_blocks[0]
        assert "image" in image_block
        assert "source" in image_block["image"]
        assert "bytes" in image_block["image"]["source"]
        assert image_block["image"]["source"]["bytes"] == file_content

        # Verify format is detected
        assert "format" in image_block["image"]
        assert image_block["image"]["format"] in ["jpeg", "png", "gif", "webp"]


@given(
    num_images=st.integers(min_value=1, max_value=5),
    file_content=image_bytes(),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_2_multiple_items_preservation(num_images, file_content):
    """Property 2: Multiple items preservation.

    **Validates: Requirements 1.3, 2.3, 3.3, 4.3**

    For any media type and list of file paths, when creating a payload,
    the resulting payload SHALL contain exactly as many content blocks of that
    media type as there are items in the list, preserving their count.
    """
    # Create multiple temporary files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        file_paths = []
        for i in range(num_images):
            file_path = create_temp_file(tmp_path, file_content, ".jpg")
            file_paths.append(file_path)

        # Create payload with multiple images
        payload = BedrockBase.create_payload(
            user_message="Test message", images=file_paths, max_tokens=256
        )

        # Verify payload structure
        content_blocks = payload["messages"][0]["content"]

        # Count image blocks
        image_blocks = [block for block in content_blocks if "image" in block]

        # Verify count matches input
        assert len(image_blocks) == num_images


@given(
    num_texts=st.integers(min_value=1, max_value=3),
    num_images=st.integers(min_value=1, max_value=3),
    num_videos=st.integers(min_value=0, max_value=2),
    num_audio=st.integers(min_value=0, max_value=2),
    num_documents=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_3_content_ordering_preservation(
    num_texts, num_images, num_videos, num_audio, num_documents
):
    """Property 3: Content ordering preservation.

    **Validates: Requirements 1.4**

    For any combination of text messages and media items, when creating a payload,
    the resulting content array SHALL preserve the order: text blocks first (in order),
    then media blocks (in the order: images, videos, audio, documents).
    """
    # Create text messages
    text_messages = [f"Text {i}" for i in range(num_texts)]

    # Create media files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        images = [
            create_temp_file(tmp_path, b"\xff\xd8\xff\xe0test\xff\xd9", ".jpg")
            for _ in range(num_images)
        ]
        videos = [
            create_temp_file(tmp_path, b"\x00\x00\x00\x20ftypisomtest", ".mp4")
            for _ in range(num_videos)
        ]
        audio = [
            create_temp_file(tmp_path, b"ID3\x03\x00\x00test", ".mp3")
            for _ in range(num_audio)
        ]
        documents = [
            create_temp_file(tmp_path, b"%PDF-1.4\ntest\n%%EOF", ".pdf")
            for _ in range(num_documents)
        ]

        # Create payload
        payload = BedrockBase.create_payload(
            user_message=text_messages if len(text_messages) > 1 else text_messages[0],
            images=images if images else None,
            videos=videos if videos else None,
            audio=audio if audio else None,
            documents=documents if documents else None,
            max_tokens=256,
        )

        # Verify ordering
        content_blocks = payload["messages"][0]["content"]

        # Extract block types in order
        block_types = []
        for block in content_blocks:
            if "text" in block:
                block_types.append("text")
            elif "image" in block:
                block_types.append("image")
            elif "video" in block:
                block_types.append("video")
            elif "audio" in block:
                block_types.append("audio")
            elif "document" in block:
                block_types.append("document")

        # Verify text blocks come first
        text_count = block_types.count("text")
        assert text_count == num_texts
        assert block_types[:text_count] == ["text"] * text_count

        # Verify media blocks follow in order: images, videos, audio, documents
        media_blocks = block_types[text_count:]
        expected_order = (
            ["image"] * num_images
            + ["video"] * num_videos
            + ["audio"] * num_audio
            + ["document"] * num_documents
        )
        assert media_blocks == expected_order


def valid_file_path_string(s: str) -> bool:
    """Check if a string is a valid file path (no null bytes, not too long)."""
    if not s or len(s) > 255:
        return False
    if "\x00" in s:
        return False
    # Check if it's a valid path that doesn't exist
    try:
        path = Path(s)
        return not path.exists()
    except (ValueError, OSError):
        return False


@given(
    non_existent_path=st.text(min_size=1, max_size=50).filter(valid_file_path_string)
)
@settings(max_examples=100)
def test_property_4_missing_file_error_handling(non_existent_path):
    """Property 4: Missing file error handling.

    **Validates: Requirements 5.2**

    For any non-existent file path provided in any media parameter,
    attempting to create a payload SHALL raise a FileNotFoundError
    with a message containing the file path.
    """
    with pytest.raises(FileNotFoundError) as exc_info:
        BedrockBase.create_payload(
            user_message="Test message", images=[non_existent_path], max_tokens=256
        )

    # Verify error message contains the file path
    assert non_existent_path in str(exc_info.value)


@given(
    invalid_item=st.one_of(
        st.integers(),
        st.floats(),
        st.dictionaries(st.text(), st.text()),
        st.lists(st.integers()),
    )
)
@settings(max_examples=100)
def test_property_5_invalid_type_rejection(invalid_item):
    """Property 5: Invalid type rejection.

    **Validates: Requirements 5.4**

    For any media parameter list containing items that are neither bytes nor strings,
    attempting to create a payload SHALL raise a TypeError with a descriptive message.
    """
    with pytest.raises(TypeError) as exc_info:
        BedrockBase.create_payload(
            user_message="Test message", images=[invalid_item], max_tokens=256
        )

    # Verify error message is descriptive
    error_msg = str(exc_info.value)
    assert "images" in error_msg
    assert "bytes" in error_msg or "str" in error_msg


@given(
    user_message=st.one_of(
        st.text(min_size=1, max_size=100),
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=3),
    ),
    max_tokens=st.integers(min_value=1, max_value=4096),
)
@settings(max_examples=100)
def test_property_6_backward_compatibility(user_message, max_tokens):
    """Property 6: Backward compatibility for text-only payloads.

    **Validates: Requirements 6.1, 6.2, 6.4**

    For any text input provided using the user_message parameter (without media parameters),
    the new create_payload implementation SHALL produce output identical to the current
    implementation, maintaining the same structure and field values.
    """
    # Create payload with text only
    payload = BedrockBase.create_payload(
        user_message=user_message, max_tokens=max_tokens
    )

    # Verify expected structure
    assert "messages" in payload
    assert "inferenceConfig" in payload
    assert payload["inferenceConfig"]["maxTokens"] == max_tokens

    # Verify messages structure
    messages = payload["messages"]
    if isinstance(user_message, str):
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 1
        assert messages[0]["content"][0]["text"] == user_message
    else:
        assert len(messages) == len(user_message)
        for i, msg in enumerate(messages):
            assert msg["role"] == "user"
            assert len(msg["content"]) == 1
            assert msg["content"][0]["text"] == user_message[i]


@given(
    max_tokens=st.integers(min_value=1, max_value=4096),
    extra_kwargs=st.dictionaries(
        st.text(min_size=1, max_size=20).filter(
            lambda x: (
                x
                not in [
                    "messages",
                    "inferenceConfig",
                    "images",
                    "documents",
                    "videos",
                    "audio",
                    "user_message",
                    "max_tokens",
                ]
            )
        ),
        st.one_of(st.text(), st.integers(), st.booleans()),
        min_size=0,
        max_size=3,
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_7_existing_parameters_preservation(max_tokens, extra_kwargs):
    """Property 7: Existing parameters preservation.

    **Validates: Requirements 6.3**

    For any values of max_tokens and additional kwargs, when creating a payload
    (text-only or multi-modal), these parameters SHALL be preserved in the
    resulting payload structure exactly as they are in the current implementation.
    """
    # Create a test image file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_path = create_temp_file(tmp_path, b"\xff\xd8\xff\xe0test\xff\xd9", ".jpg")

        # Create payload with extra kwargs
        payload = BedrockBase.create_payload(
            user_message="Test message",
            images=[image_path],
            max_tokens=max_tokens,
            **extra_kwargs,
        )

        # Verify max_tokens is preserved
        assert payload["inferenceConfig"]["maxTokens"] == max_tokens

        # Verify extra kwargs are preserved
        for key, value in extra_kwargs.items():
            assert key in payload
            assert payload[key] == value


@given(
    image_content=image_bytes(),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_8_serialization_round_trip(image_content):
    """Property 8: Multi-modal payload serialization round-trip.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 9.2, 9.3**

    For any valid multi-modal payload containing binary content, the round-trip
    property SHALL hold: load_payloads(save_payloads(payload)) == payload,
    preserving byte-for-byte equality of all binary content and exact equality
    of all other values.
    """
    # Create a test image file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_path = create_temp_file(tmp_path, image_content, ".jpg")

        # Create payload with binary content
        original_payload = BedrockBase.create_payload(
            user_message="Test message", images=[image_path], max_tokens=256
        )

        # Save and load the payload
        save_path = save_payloads(original_payload, tmp_dir, "test_payload.jsonl")
        loaded_payloads = list(load_payloads(save_path))

        # Verify round-trip preservation
        assert len(loaded_payloads) == 1
        loaded_payload = loaded_payloads[0]

        # Verify structure equality
        assert loaded_payload == original_payload

        # Verify binary content is preserved byte-for-byte
        original_bytes = original_payload["messages"][0]["content"][1]["image"][
            "source"
        ]["bytes"]
        loaded_bytes = loaded_payload["messages"][0]["content"][1]["image"]["source"][
            "bytes"
        ]
        assert original_bytes == loaded_bytes
        assert isinstance(loaded_bytes, bytes)


@given(
    extension=st.sampled_from([".jpg", ".png", ".pdf", ".mp4", ".mp3"]),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_9_format_detection(extension):
    """Property 9: Format detection from file content.

    **Validates: Requirements 8.5**

    For any file with recognizable content (JPEG, PNG, PDF, MP4, etc.),
    when creating a payload, the format SHALL be correctly detected from
    the file's magic bytes using puremagic (or from extension as fallback)
    and included in the content block.
    """
    # Generate appropriate content for the extension
    if extension in [".jpg", ".jpeg"]:
        file_content = b"\xff\xd8\xff\xe0test\xff\xd9"
        media_param = "images"
        media_key = "image"
    elif extension == ".png":
        file_content = b"\x89PNG\r\n\x1a\ntest"
        media_param = "images"
        media_key = "image"
    elif extension == ".pdf":
        file_content = b"%PDF-1.4\ntest\n%%EOF"
        media_param = "documents"
        media_key = "document"
    elif extension == ".mp4":
        file_content = b"\x00\x00\x00\x20ftypisomtest"
        media_param = "videos"
        media_key = "video"
    elif extension == ".mp3":
        file_content = b"ID3\x03\x00\x00test"
        media_param = "audio"
        media_key = "audio"

    # Create temporary file and use it within the same context
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        file_path = create_temp_file(tmp_path, file_content, extension)

        # Create payload
        kwargs = {media_param: [file_path]}
        payload = BedrockBase.create_payload(
            user_message="Test message", max_tokens=256, **kwargs
        )

        # Find the media content block
        content_blocks = payload["messages"][0]["content"]
        media_blocks = [block for block in content_blocks if media_key in block]
        assert len(media_blocks) == 1

        # Verify format is detected and included
        media_block = media_blocks[0]
        assert "format" in media_block[media_key]
        assert isinstance(media_block[media_key]["format"], str)
        assert len(media_block[media_key]["format"]) > 0


@given(
    num_prompts=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_10_load_prompts_integration(num_prompts):
    """Property 10: load_prompts integration with multi-modal payloads.

    **Validates: Requirements 9.1**

    For any create_payload call that produces multi-modal payloads,
    when used with load_prompts, the function SHALL yield valid multi-modal
    payloads that can be serialized and used with endpoint invoke methods.
    """
    # Create a test image file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_path = create_temp_file(tmp_path, b"\xff\xd8\xff\xe0test\xff\xd9", ".jpg")

        # Create a prompts file
        prompts_file = tmp_path / "prompts.txt"
        prompts = [f"Prompt {i}" for i in range(num_prompts)]
        prompts_file.write_text("\n".join(prompts))

        # Define create_payload function that produces multi-modal payloads
        def create_multimodal_payload(input_text, **kwargs):
            return BedrockBase.create_payload(
                user_message=input_text, images=[image_path], max_tokens=256, **kwargs
            )

        # Use load_prompts with the multi-modal create_payload function
        from llmeter.prompt_utils import load_prompts

        payloads = list(
            load_prompts(
                prompts_file,
                create_payload_fn=create_multimodal_payload,
                create_payload_kwargs={},
            )
        )

        # Verify correct number of payloads
        assert len(payloads) == num_prompts

        # Verify each payload is valid and contains multi-modal content
        for i, payload in enumerate(payloads):
            assert "messages" in payload
            content_blocks = payload["messages"][0]["content"]

            # Should have text and image blocks
            text_blocks = [block for block in content_blocks if "text" in block]
            image_blocks = [block for block in content_blocks if "image" in block]

            assert len(text_blocks) >= 1
            assert len(image_blocks) == 1

            # Verify the payload can be serialized
            json_str = json.dumps(payload, cls=LLMeterBytesEncoder)
            assert len(json_str) > 0

            # Verify it can be deserialized
            deserialized = json.loads(json_str, object_hook=llmeter_bytes_decoder)
            assert deserialized == payload


@given(
    extra_kwargs=st.dictionaries(
        st.text(min_size=1, max_size=20).filter(
            lambda x: (
                x
                not in [
                    "messages",
                    "inferenceConfig",
                    "user_message",
                    "max_tokens",
                    "images",
                    "documents",
                    "videos",
                    "audio",
                ]
            )
        ),
        st.one_of(st.text(min_size=1), st.integers(), st.booleans()),
        min_size=1,
        max_size=3,
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_11_create_payload_kwargs_compatibility(extra_kwargs):
    """Property 11: create_payload_kwargs pattern compatibility.

    **Validates: Requirements 9.4**

    For any dictionary of additional parameters passed via create_payload_kwargs,
    when used with load_prompts or directly with create_payload, these parameters
    SHALL be passed through and included in the resulting payload.
    """
    # Create a test image file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_path = create_temp_file(tmp_path, b"\xff\xd8\xff\xe0test\xff\xd9", ".jpg")

        # Create payload with extra kwargs
        payload = BedrockBase.create_payload(
            user_message="Test message",
            images=[image_path],
            max_tokens=256,
            **extra_kwargs,
        )

        # Verify all extra kwargs are present in the payload
        for key, value in extra_kwargs.items():
            assert key in payload
            assert payload[key] == value

        # Test with load_prompts pattern
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("Test prompt")

        def create_multimodal_payload(input_text, **kwargs):
            return BedrockBase.create_payload(
                user_message=input_text, images=[image_path], max_tokens=256, **kwargs
            )

        from llmeter.prompt_utils import load_prompts

        payloads = list(
            load_prompts(
                prompts_file,
                create_payload_fn=create_multimodal_payload,
                create_payload_kwargs=extra_kwargs,
            )
        )

        # Verify kwargs are passed through
        assert len(payloads) == 1
        for key, value in extra_kwargs.items():
            assert key in payloads[0]
            assert payloads[0][key] == value
