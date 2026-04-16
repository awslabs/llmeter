# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Bedrock Converse API endpoints.

This module contains integration tests that verify BedrockConverse and
BedrockConverseStream endpoints work correctly with actual AWS Bedrock services.

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest -m integ

Required AWS Permissions:
    - bedrock:InvokeModel
    - bedrock:InvokeModelWithResponseStream

Estimated Cost:
    - ~$0.0001 per text-only test run
    - ~$0.0002 per image test run
    - ~$0.0006 total for all tests in this module
"""

import pytest

from llmeter.endpoints.bedrock import BedrockConverse


@pytest.mark.integ
def test_bedrock_converse_non_streaming(
    aws_credentials, aws_region, bedrock_test_model, test_payload
):
    """
    Test BedrockConverse endpoint with actual Bedrock service.

    This test validates that the BedrockConverse endpoint can successfully:
    - Create an endpoint instance with a valid model
    - Invoke the endpoint with a test payload
    - Receive a non-streaming response
    - Extract response text, token counts, and timing information
    - Complete without errors

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
        test_payload: Simple text test payload (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0001 per test run (using Claude Haiku 4.5 with minimal tokens)
    """
    # Create BedrockConverse endpoint instance
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    # Invoke the endpoint with test payload
    response = endpoint.invoke(test_payload)

    # Verify response contains non-empty text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str), "Response text should be a string"

    # Verify token counts are present and positive
    assert response.num_tokens_input is not None, "Input token count should not be None"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert response.num_tokens_output is not None, (
        "Output token count should not be None"
    )
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify response time is measured and positive
    assert response.time_to_last_token is not None, "Response time should not be None"
    assert response.time_to_last_token > 0, "Response time should be positive"

    # Verify no errors in response
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response has an ID from the Bedrock API (AWS RequestId format)
    assert response.id is not None, "Response should have an ID"
    assert "-" in response.id, (
        f"Response ID should be an AWS RequestId (UUID with hyphens), got: {response.id}"
    )


@pytest.mark.integ
def test_bedrock_converse_streaming(
    aws_credentials, aws_region, bedrock_test_model, test_payload
):
    """
    Test BedrockConverseStream endpoint with actual Bedrock service.

    This test validates that the BedrockConverseStream endpoint can successfully:
    - Create a streaming endpoint instance with a valid model
    - Invoke the endpoint with a test payload
    - Receive a streaming response
    - Extract response text, token counts, and timing information
    - Measure time to first token (TTFT) and time to last token (TTLT)
    - Verify TTLT > TTFT
    - Complete without errors

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
        test_payload: Simple text test payload (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModelWithResponseStream

    Estimated Cost:
        ~$0.0001 per test run (using Claude Haiku 4.5 with minimal tokens)
    """
    from llmeter.endpoints.bedrock import BedrockConverseStream

    # Create BedrockConverseStream endpoint instance
    endpoint = BedrockConverseStream(model_id=bedrock_test_model, region=aws_region)

    # Invoke the endpoint with test payload
    response = endpoint.invoke(test_payload)

    # Verify response contains non-empty text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str), "Response text should be a string"

    # Verify token counts are present and positive
    assert response.num_tokens_input is not None, "Input token count should not be None"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert response.num_tokens_output is not None, (
        "Output token count should not be None"
    )
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify time to first token is measured and positive
    assert response.time_to_first_token is not None, (
        "Time to first token should not be None"
    )
    assert response.time_to_first_token > 0, "Time to first token should be positive"

    # Verify time to last token is measured and positive
    assert response.time_to_last_token is not None, (
        "Time to last token should not be None"
    )
    assert response.time_to_last_token > 0, "Time to last token should be positive"

    # Verify TTLT > TTFT (streaming should take time to complete)
    assert response.time_to_last_token > response.time_to_first_token, (
        "Time to last token should be greater than time to first token"
    )

    # Verify no errors in response
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response has an ID from the Bedrock API (AWS RequestId format)
    assert response.id is not None, "Response should have an ID"
    assert "-" in response.id, (
        f"Response ID should be an AWS RequestId (UUID with hyphens), got: {response.id}"
    )


@pytest.mark.integ
def test_bedrock_converse_with_image(
    aws_credentials, aws_region, bedrock_test_model, test_payload_with_image
):
    """
    Test BedrockConverse endpoint with image payload for multimodal capabilities.

    This test validates that the BedrockConverse endpoint can successfully:
    - Create an endpoint instance with a valid model
    - Invoke the endpoint with an image payload
    - Receive a non-streaming response
    - Extract response text describing the image
    - Verify token counts include image tokens
    - Measure response timing
    - Complete without errors

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5 (with multimodal content)**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
        test_payload_with_image: Test payload with image content (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0002 per test run (higher due to image processing)
    """
    # Create BedrockConverse endpoint instance
    endpoint = BedrockConverse(model_id=bedrock_test_model, region=aws_region)

    # Invoke the endpoint with image payload
    response = endpoint.invoke(test_payload_with_image)

    # Verify response contains non-empty text
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert isinstance(response.response_text, str), "Response text should be a string"

    # Verify the model actually sees the image content — it must identify at least
    # one of the two colors in the split image (red/blue)
    response_lower = response.response_text.lower()
    assert "red" in response_lower or "blue" in response_lower, (
        "Model should identify at least one color (red/blue) from the split image, got: "
        f"{response.response_text[:200]}"
    )

    # Verify token counts are present and positive
    assert response.num_tokens_input is not None, "Input token count should not be None"
    assert response.num_tokens_input > 0, "Input token count should be positive"
    assert response.num_tokens_input > 30, (
        "Input token count should include image tokens (expected > 30)"
    )

    assert response.num_tokens_output is not None, (
        "Output token count should not be None"
    )
    assert response.num_tokens_output > 0, "Output token count should be positive"

    # Verify response time is measured and positive
    assert response.time_to_last_token is not None, "Response time should not be None"
    assert response.time_to_last_token > 0, "Response time should be positive"

    # Verify no errors in response
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response has an ID from the Bedrock API (AWS RequestId format)
    assert response.id is not None, "Response should have an ID"
    assert "-" in response.id, (
        f"Response ID should be an AWS RequestId (UUID with hyphens), got: {response.id}"
    )


@pytest.mark.integ
def test_bedrock_converse_streaming_with_image(
    aws_credentials, aws_region, bedrock_test_model, test_payload_with_image
):
    """
    Test BedrockConverseStream endpoint with image payload.

    Verifies the model actually processes the image by checking it identifies
    the dominant color of the test image (a red square).

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
        test_payload_with_image: Test payload with image content (from fixture).
    """
    from llmeter.endpoints.bedrock import BedrockConverseStream

    endpoint = BedrockConverseStream(model_id=bedrock_test_model, region=aws_region)
    response = endpoint.invoke(test_payload_with_image)

    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"

    # Verify the model actually sees the image content
    response_lower = response.response_text.lower()
    assert "red" in response_lower or "blue" in response_lower, (
        "Model should identify at least one color (red/blue) from the split image, got: "
        f"{response.response_text[:200]}"
    )

    # Verify token counts include image tokens
    assert response.num_tokens_input is not None, "Input token count should not be None"
    assert response.num_tokens_input > 30, (
        "Input token count should include image tokens (expected > 30)"
    )

    # Verify streaming timing
    assert response.time_to_first_token is not None
    assert response.time_to_first_token > 0
    assert response.time_to_last_token is not None
    assert response.time_to_last_token > response.time_to_first_token

    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )

    # Verify response has an ID from the Bedrock API (AWS RequestId format)
    assert response.id is not None, "Response should have an ID"
    assert "-" in response.id, (
        f"Response ID should be an AWS RequestId (UUID with hyphens), got: {response.id}"
    )


def test_save_load_payload_with_image(test_payload_with_image, tmp_path):
    """
    Test saving and loading payload with image content using serialization.

    This test validates that:
    - Payloads with image bytes can be saved to disk
    - Saved payloads contain valid JSON with marker objects
    - Loaded payloads restore bytes objects correctly
    - Round-trip preserves byte-for-byte equality

    **Validates: Requirements 1.6, 9.3**

    Args:
        test_payload_with_image: Test payload with image content (from fixture).
        tmp_path: Temporary directory for test files (from pytest).
    """
    from llmeter.prompt_utils import load_payloads, save_payloads

    # Save payload with image
    output_file = save_payloads(test_payload_with_image, tmp_path, "test_image.jsonl")
    assert output_file.exists(), "Output file should be created"

    # Verify file contains valid JSON with marker objects
    with output_file.open("r") as f:
        content = f.read()
        assert "__llmeter_bytes__" in content, "File should contain marker objects"

    # Load payload and verify bytes are restored
    loaded_payloads = list(load_payloads(output_file))
    assert len(loaded_payloads) == 1, "Should load exactly one payload"

    loaded = loaded_payloads[0]
    original_bytes = test_payload_with_image["messages"][0]["content"][0]["image"][
        "source"
    ]["bytes"]
    loaded_bytes = loaded["messages"][0]["content"][0]["image"]["source"]["bytes"]

    assert isinstance(loaded_bytes, bytes), "Loaded bytes should be bytes type"
    assert loaded_bytes == original_bytes, "Bytes should match after round-trip"
    assert loaded == test_payload_with_image, (
        "Full payload should match after round-trip"
    )


def test_save_load_payload_with_video(tmp_path):
    """
    Test saving and loading payload with video content using serialization.

    This test validates that:
    - Payloads with video bytes can be saved to disk
    - Saved payloads contain valid JSON with marker objects
    - Loaded payloads restore bytes objects correctly
    - Video content path (messages[].content[].video.source.bytes) is handled

    **Validates: Requirements 1.6, 9.4**

    Args:
        tmp_path: Temporary directory for test files (from pytest).
    """
    from llmeter.prompt_utils import load_payloads, save_payloads

    # Create a test payload with video content (simulated video bytes)
    video_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "video": {
                            "format": "mp4",
                            "source": {
                                "bytes": b"\x00\x00\x00\x18ftypmp42"
                            },  # MP4 header
                        }
                    },
                    {"text": "What is happening in this video?"},
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 150},
    }

    # Save payload with video
    output_file = save_payloads(video_payload, tmp_path, "test_video.jsonl")
    assert output_file.exists(), "Output file should be created"

    # Verify file contains valid JSON with marker objects
    with output_file.open("r") as f:
        content = f.read()
        assert "__llmeter_bytes__" in content, "File should contain marker objects"

    # Load payload and verify bytes are restored
    loaded_payloads = list(load_payloads(output_file))
    assert len(loaded_payloads) == 1, "Should load exactly one payload"

    loaded = loaded_payloads[0]
    original_bytes = video_payload["messages"][0]["content"][0]["video"]["source"][
        "bytes"
    ]
    loaded_bytes = loaded["messages"][0]["content"][0]["video"]["source"]["bytes"]

    assert isinstance(loaded_bytes, bytes), "Loaded bytes should be bytes type"
    assert loaded_bytes == original_bytes, "Bytes should match after round-trip"
    assert loaded == video_payload, "Full payload should match after round-trip"


def test_save_load_multiple_images(tmp_path):
    """
    Test saving and loading payload with multiple images in single payload.

    This test validates that:
    - Payloads with multiple image bytes can be saved to disk
    - All images are correctly serialized with marker objects
    - All images are correctly restored after loading
    - Multiple bytes objects in same payload are handled independently

    **Validates: Requirements 9.8**

    Args:
        tmp_path: Temporary directory for test files (from pytest).
    """
    from llmeter.prompt_utils import load_payloads, save_payloads

    # Create a test payload with multiple images
    multi_image_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": b"\x89PNG\r\n\x1a\n"},  # PNG header
                        }
                    },
                    {"text": "Compare these images:"},
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": b"\xff\xd8\xff\xe0"},  # JPEG header
                        }
                    },
                    {
                        "image": {
                            "format": "png",
                            "source": {
                                "bytes": b"\x89PNG\r\n\x1a\n\x00\x00"
                            },  # Different PNG
                        }
                    },
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 200},
    }

    # Save payload with multiple images
    output_file = save_payloads(multi_image_payload, tmp_path, "test_multi_image.jsonl")
    assert output_file.exists(), "Output file should be created"

    # Verify file contains multiple marker objects
    with output_file.open("r") as f:
        content = f.read()
        marker_count = content.count("__llmeter_bytes__")
        assert marker_count == 3, (
            f"File should contain 3 marker objects, found {marker_count}"
        )

    # Load payload and verify all bytes are restored
    loaded_payloads = list(load_payloads(output_file))
    assert len(loaded_payloads) == 1, "Should load exactly one payload"

    loaded = loaded_payloads[0]
    content = loaded["messages"][0]["content"]

    # Verify first image
    assert isinstance(content[0]["image"]["source"]["bytes"], bytes)
    assert content[0]["image"]["source"]["bytes"] == b"\x89PNG\r\n\x1a\n"

    # Verify second image
    assert isinstance(content[2]["image"]["source"]["bytes"], bytes)
    assert content[2]["image"]["source"]["bytes"] == b"\xff\xd8\xff\xe0"

    # Verify third image
    assert isinstance(content[3]["image"]["source"]["bytes"], bytes)
    assert content[3]["image"]["source"]["bytes"] == b"\x89PNG\r\n\x1a\n\x00\x00"

    # Verify full payload matches
    assert loaded == multi_image_payload, "Full payload should match after round-trip"


@pytest.mark.integ
def test_round_trip_bedrock_converse_structure(
    test_payload_with_image, tmp_path, aws_credentials, aws_region
):
    """
    Test round-trip serialization with actual Bedrock Converse API structure.

    This test validates that:
    - Complete Bedrock Converse API payload structure is preserved
    - All nested fields (modelId, messages, inferenceConfig) are maintained
    - Image bytes in messages[].content[].image.source.bytes path are handled
    - Round-trip produces identical payload structure
    - Loaded payload can be used with the endpoint's invoke method

    **Validates: Requirements 1.6, 9.3**

    Args:
        test_payload_with_image: Test payload with image content (from fixture).
        tmp_path: Temporary directory for test files (from pytest).
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
    """
    from llmeter.endpoints.bedrock import BedrockConverse
    from llmeter.prompt_utils import load_payloads, save_payloads

    # Create a complete Bedrock Converse payload with all typical fields
    complete_payload = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "messages": test_payload_with_image["messages"],
        "inferenceConfig": {
            "maxTokens": 150,
            "temperature": 0.7,
            "topP": 0.9,
        },
        "system": [{"text": "You are a helpful assistant that describes images."}],
    }

    # Save and load the complete payload
    output_file = save_payloads(complete_payload, tmp_path, "test_complete.jsonl")
    loaded_payloads = list(load_payloads(output_file))

    assert len(loaded_payloads) == 1, "Should load exactly one payload"
    loaded = loaded_payloads[0]

    # Verify all top-level fields are preserved
    assert loaded["modelId"] == complete_payload["modelId"]
    assert loaded["inferenceConfig"] == complete_payload["inferenceConfig"]
    assert loaded["system"] == complete_payload["system"]

    # Verify messages structure is preserved
    assert len(loaded["messages"]) == len(complete_payload["messages"])
    assert loaded["messages"][0]["role"] == complete_payload["messages"][0]["role"]

    # Verify image bytes are correctly restored
    original_bytes = complete_payload["messages"][0]["content"][0]["image"]["source"][
        "bytes"
    ]
    loaded_bytes = loaded["messages"][0]["content"][0]["image"]["source"]["bytes"]
    assert isinstance(loaded_bytes, bytes)
    assert loaded_bytes == original_bytes

    # Verify complete equality
    assert loaded == complete_payload, "Complete payload should match after round-trip"

    # Verify the loaded payload can be used with the endpoint's invoke method
    # Extract model_id from the loaded payload
    model_id = loaded.pop("modelId")
    endpoint = BedrockConverse(model_id=model_id, region=aws_region)
    response = endpoint.invoke(loaded)

    # Verify the endpoint successfully processed the loaded payload
    assert response.response_text is not None, "Response should contain text"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )
