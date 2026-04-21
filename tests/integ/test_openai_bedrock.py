# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for OpenAI SDK with Bedrock endpoints.

This module contains integration tests that verify the OpenAI SDK can successfully
call Bedrock models via Bedrock's OpenAI-compatible endpoint (Bedrock Mantle).

Tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run these tests:
    uv run pytest -m integ

Required AWS Permissions:
    - bedrock:InvokeModel
    - bedrock:InvokeModelWithResponseStream

Estimated Cost:
    - ~$0.0001 per test run
    - ~$0.0002 total for all tests in this module
"""

import time

import pytest

try:
    from aws_bedrock_token_generator import provide_token
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_openai_bedrock_non_streaming(
    aws_credentials, aws_region, bedrock_openai_test_model
):
    """
    Test OpenAI SDK with Bedrock non-streaming.

    This test validates that the OpenAI SDK can successfully:
    - Configure with Bedrock's OpenAI-compatible endpoint URL
    - Authenticate using AWS temporary token
    - Make a non-streaming chat completion request
    - Receive a response in OpenAI format
    - Extract response text and token counts
    - Measure response timing
    - Complete without errors

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_openai_test_model: Model ID for OpenAI SDK testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0001 per test run (using Claude Haiku 4.5 with minimal tokens)
    """
    # Generate temporary token for Bedrock authentication
    token = provide_token(region=aws_region)

    # Configure OpenAI client with Bedrock endpoint
    base_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com/openai/v1"
    client = OpenAI(api_key=token, base_url=base_url)

    # Create test payload
    messages = [
        {
            "role": "user",
            "content": "Hello, this is a test message. Please respond with a brief greeting.",
        }
    ]

    # Measure timing and invoke
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=bedrock_openai_test_model,
        messages=messages,
        max_tokens=100,
    )
    response_time = time.perf_counter() - start_time

    # Verify response contains non-empty text
    assert response.choices is not None, "Response choices should not be None"
    assert len(response.choices) > 0, "Response should have at least one choice"
    assert response.choices[0].message.content is not None, (
        "Response content should not be None"
    )
    assert len(response.choices[0].message.content) > 0, (
        "Response content should not be empty"
    )
    assert isinstance(response.choices[0].message.content, str), (
        "Response content should be a string"
    )

    # Verify token counts are present and positive
    assert response.usage is not None, "Response usage should not be None"
    assert response.usage.prompt_tokens is not None, (
        "Input token count should not be None"
    )
    assert response.usage.prompt_tokens > 0, "Input token count should be positive"
    assert response.usage.completion_tokens is not None, (
        "Output token count should not be None"
    )
    assert response.usage.completion_tokens > 0, "Output token count should be positive"

    # Verify response time is measured and positive
    assert response_time > 0, "Response time should be positive"

    # Verify response has an ID
    assert response.id is not None, "Response should have an ID"

    # Verify response model matches request
    assert response.model is not None, "Response should have a model field"


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_openai_bedrock_streaming(
    aws_credentials, aws_region, bedrock_openai_test_model
):
    """
    Test OpenAI SDK with Bedrock streaming.

    This test validates that the OpenAI SDK can successfully:
    - Configure with Bedrock's OpenAI-compatible endpoint URL
    - Authenticate using AWS temporary token
    - Make a streaming chat completion request
    - Receive response chunks in OpenAI format
    - Assemble complete response text from chunks
    - Manually measure time to first token (TTFT) and time to last token (TTLT)
    - Extract token counts from the final response
    - Complete without errors

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_openai_test_model: Model ID for OpenAI SDK testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModelWithResponseStream

    Estimated Cost:
        ~$0.0001 per test run (using Claude Haiku 4.5 with minimal tokens)
    """
    # Generate temporary token for Bedrock authentication
    token = provide_token(region=aws_region)

    # Configure OpenAI client with Bedrock endpoint
    base_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com/openai/v1"
    client = OpenAI(api_key=token, base_url=base_url)

    # Create test payload
    messages = [
        {
            "role": "user",
            "content": "Hello, this is a test message. Please respond with a brief greeting.",
        }
    ]

    # Measure timing and invoke with streaming
    start_time = time.perf_counter()
    stream = client.chat.completions.create(
        model=bedrock_openai_test_model,
        messages=messages,
        max_tokens=100,
        stream=True,
        stream_options={"include_usage": True},
    )

    # Process streaming response
    response_text = ""
    response_id = None
    time_to_first_token = None
    prompt_tokens = None
    completion_tokens = None
    first_chunk = True

    for chunk in stream:
        if first_chunk:
            time_to_first_token = time.perf_counter() - start_time
            response_id = chunk.id
            first_chunk = False

        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                response_text += delta.content

        # Extract usage information from final chunk
        if hasattr(chunk, "usage") and chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    time_to_last_token = time.perf_counter() - start_time

    # Verify response contains non-empty text
    assert response_text is not None, "Response text should not be None"
    assert len(response_text) > 0, "Response text should not be empty"
    assert isinstance(response_text, str), "Response text should be a string"

    # Verify token counts are present and positive
    assert prompt_tokens is not None, "Input token count should not be None"
    assert prompt_tokens > 0, "Input token count should be positive"
    assert completion_tokens is not None, "Output token count should not be None"
    assert completion_tokens > 0, "Output token count should be positive"

    # Verify time to first token is measured and positive
    assert time_to_first_token is not None, "Time to first token should not be None"
    assert time_to_first_token > 0, "Time to first token should be positive"

    # Verify time to last token is measured and positive
    assert time_to_last_token is not None, "Time to last token should not be None"
    assert time_to_last_token > 0, "Time to last token should be positive"

    # Verify TTLT > TTFT (streaming should take time to complete)
    assert time_to_last_token > time_to_first_token, (
        "Time to last token should be greater than time to first token"
    )

    # Verify response has an ID
    assert response_id is not None, "Response should have an ID"


def test_save_load_openai_payload_with_image_url(tmp_path):
    """
    Test saving and loading OpenAI payload with image_url data URI.

    This test validates that:
    - OpenAI payloads with image_url data URIs can be saved to disk
    - Saved payloads contain valid JSON with marker objects
    - Loaded payloads restore bytes objects correctly in image_url.url field
    - Round-trip preserves byte-for-byte equality

    **Validates: Requirements 9.6, 9.7**

    Args:
        tmp_path: Temporary directory for test files (from pytest).
    """
    import base64

    from llmeter.prompt_utils import load_payloads, save_payloads

    # Create a small test image (1x1 red pixel PNG)
    test_image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )

    # Create OpenAI chat.completions payload with image_url data URI
    openai_payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": test_image_bytes  # Store as bytes for serialization
                        },
                    },
                ],
            }
        ],
        "max_tokens": 100,
    }

    # Save payload with image_url
    output_file = save_payloads(openai_payload, tmp_path, "test_openai_image.jsonl")
    assert output_file.exists(), "Output file should be created"

    # Verify file contains valid JSON with marker objects
    with output_file.open("r") as f:
        content = f.read()
        assert "__llmeter_bytes__" in content, "File should contain marker objects"

    # Load payload and verify bytes are restored
    loaded_payloads = list(load_payloads(output_file))
    assert len(loaded_payloads) == 1, "Should load exactly one payload"

    loaded = loaded_payloads[0]
    original_bytes = openai_payload["messages"][0]["content"][1]["image_url"]["url"]
    loaded_bytes = loaded["messages"][0]["content"][1]["image_url"]["url"]

    assert isinstance(loaded_bytes, bytes), "Loaded bytes should be bytes type"
    assert loaded_bytes == original_bytes, "Bytes should match after round-trip"
    assert loaded == openai_payload, "Full payload should match after round-trip"


@pytest.mark.integ
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not installed")
def test_save_load_openai_complete_structure(
    tmp_path,
    aws_credentials,
    aws_region,
    bedrock_openai_multimodal_test_model,
    bedrock_openai_multimodal_endpoint_url,
    test_image_bytes,
):
    """
    Test round-trip with actual OpenAI chat.completions structure and API call.

    This test validates that:
    - Complete OpenAI chat.completions payloads serialize correctly
    - All typical OpenAI fields are preserved
    - Multiple content items with mixed text and images work correctly
    - The messages[].content[].image_url.url path is handled correctly
    - Binary image data is preserved through save/load cycle
    - Loaded payload can be used with the OpenAI client for multimodal requests

    **Validates: Requirements 9.6, 9.7**

    Args:
        tmp_path: Temporary directory for test files (from pytest).
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_openai_multimodal_test_model: Model ID for OpenAI SDK multimodal testing (from fixture).
        bedrock_openai_multimodal_endpoint_url: Bedrock Mantle endpoint URL for multimodal (from fixture).
        test_image_bytes: Tuple of test images as binary data (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0002 per test run (using Qwen3-VL with minimal tokens)
    """
    import base64

    from llmeter.prompt_utils import load_payloads, save_payloads

    # Get test images as binary data
    image1_binary, image2_binary = test_image_bytes

    # Create complete OpenAI payload with images stored as binary bytes
    # Note: We store raw bytes in the payload, which will be serialized to base64 for storage
    complete_payload = {
        "model": bedrock_openai_multimodal_test_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these images briefly:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image1_binary},  # Binary bytes
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image2_binary},  # Binary bytes
                    },
                ],
            },
        ],
        "max_tokens": 100,
    }

    # Save and load the complete payload
    output_file = save_payloads(
        complete_payload, tmp_path, "test_openai_complete.jsonl"
    )
    loaded_payloads = list(load_payloads(output_file))

    assert len(loaded_payloads) == 1, "Should load exactly one payload"
    loaded = loaded_payloads[0]

    # Verify structure is preserved
    assert loaded["model"] == complete_payload["model"]
    assert len(loaded["messages"]) == 1
    assert loaded["messages"][0]["role"] == "user"
    assert len(loaded["messages"][0]["content"]) == 3

    # Verify image bytes are restored correctly from serialization
    loaded_image1_binary = loaded["messages"][0]["content"][1]["image_url"]["url"]
    loaded_image2_binary = loaded["messages"][0]["content"][2]["image_url"]["url"]

    assert isinstance(loaded_image1_binary, bytes), "First image should be bytes"
    assert isinstance(loaded_image2_binary, bytes), "Second image should be bytes"
    assert loaded_image1_binary == image1_binary, "First image bytes should match"
    assert loaded_image2_binary == image2_binary, "Second image bytes should match"

    # Verify full round-trip equality
    assert loaded == complete_payload, "Full payload should match after round-trip"

    # Verify the loaded payload can be used with the OpenAI client
    # Note: OpenAI API expects base64-encoded data URIs (ASCII strings), not raw bytes
    # Multimodal models require the Bedrock Mantle endpoint, not bedrock-runtime
    token = provide_token(region=aws_region)
    client = OpenAI(api_key=token, base_url=bedrock_openai_multimodal_endpoint_url)

    # Convert binary bytes to base64-encoded ASCII strings for API call
    image1_base64_ascii = base64.b64encode(loaded_image1_binary).decode("utf-8")
    image2_base64_ascii = base64.b64encode(loaded_image2_binary).decode("utf-8")

    # Build API payload - data URI format with JPEG MIME type
    api_payload = {
        "model": bedrock_openai_multimodal_test_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these images briefly:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image1_base64_ascii}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image2_base64_ascii}",
                        },
                    },
                ],
            },
        ],
        "max_tokens": 100,
    }

    # Invoke the API with the loaded multimodal payload
    response = client.chat.completions.create(**api_payload)

    # Verify the client successfully processed the loaded payload
    assert response.choices is not None, "Response should contain choices"
    assert len(response.choices) > 0, "Response should have at least one choice"
    assert response.choices[0].message.content is not None, (
        "Response should contain text"
    )
    assert len(response.choices[0].message.content) > 0, (
        "Response text should not be empty"
    )
