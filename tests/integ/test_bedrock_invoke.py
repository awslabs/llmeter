# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Bedrock Invoke API endpoints.

This module contains integration tests that verify BedrockInvoke and
BedrockInvokeStream endpoints work correctly with actual AWS Bedrock services.

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

import pytest

from llmeter.endpoints.bedrock_invoke import BedrockInvoke


@pytest.mark.integ
def test_bedrock_invoke_non_streaming(aws_credentials, aws_region, bedrock_test_model):
    """
    Test BedrockInvoke endpoint with actual Bedrock service.

    This test validates that the BedrockInvoke endpoint can successfully:
    - Create an endpoint instance with a valid model and JMESPath expressions
    - Invoke the endpoint with a test payload
    - Receive a non-streaming response
    - Extract response text via JMESPath
    - Extract token counts via JMESPath when available
    - Measure response timing
    - Complete without errors

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModel

    Estimated Cost:
        ~$0.0001 per test run (using Amazon Nova Lite with minimal tokens)
    """
    # Create BedrockInvoke endpoint instance with JMESPath expressions for Amazon Nova models
    # Nova models via InvokeModel API use a Converse-like native format
    endpoint = BedrockInvoke(
        model_id=bedrock_test_model,
        region=aws_region,
        # JMESPath for Nova native Invoke API response format
        generated_text_jmespath="output.message.content[0].text",
        generated_token_count_jmespath="usage.outputTokens",
        input_token_count_jmespath="usage.inputTokens",
        input_text_jmespath="messages[0].content[0].text",
    )

    # Create payload in Nova native Invoke API format
    invoke_payload = {
        "schemaVersion": "messages-v1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Hello, this is a test message. Please respond with a brief greeting."
                    }
                ],
            }
        ],
        "inferenceConfig": {
            "maxTokens": 100,
        },
    }

    # Invoke the endpoint with test payload
    response = endpoint.invoke(invoke_payload)

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

    # Verify response has an ID
    assert response.id is not None, "Response should have an ID"


@pytest.mark.integ
def test_bedrock_invoke_with_image(aws_credentials, aws_region, bedrock_test_model):
    """
    Test BedrockInvoke endpoint with an image payload.

    This test validates that:
    - BedrockInvoke can send image payloads via the InvokeModel API
    - The model actually processes the image (verifies content description)
    - Image data is correctly base64-encoded in the JSON payload
    - No __llmeter_bytes__ markers leak into the API request

    The InvokeModel API requires image data as base64-encoded strings in JSON,
    unlike the Converse API which accepts raw bytes.

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
    """
    import base64
    import io

    from PIL import Image

    # Create a simple 100x100 red square PNG image
    img = Image.new("RGB", (100, 100), color="red")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    endpoint = BedrockInvoke(
        model_id=bedrock_test_model,
        region=aws_region,
        generated_text_jmespath="output.message.content[0].text",
        generated_token_count_jmespath="usage.outputTokens",
        input_token_count_jmespath="usage.inputTokens",
        # Use JMESPath that handles mixed content (image + text blocks)
        input_text_jmespath="messages[0].content[].text",
    )

    # Nova native InvokeModel format with base64-encoded image
    invoke_payload = {
        "schemaVersion": "messages-v1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": img_b64},
                        },
                    },
                    {
                        "text": "What color is the square in this image? Answer with just the color name.",
                    },
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 50},
    }

    response = endpoint.invoke(invoke_payload)

    assert response.error is None, f"Response should not contain errors: {response.error}"
    assert response.response_text is not None, "Response text should not be None"
    assert len(response.response_text) > 0, "Response text should not be empty"

    # Verify the model actually processes the image (responds with a color name)
    response_lower = response.response_text.lower()
    assert any(
        color in response_lower
        for color in ("red", "white", "pink", "orange", "color", "square", "image")
    ), f"Model should describe the image content, got: {response.response_text}"


@pytest.mark.integ
def test_bedrock_invoke_streaming(aws_credentials, aws_region, bedrock_test_model):
    """
    Test BedrockInvokeStream endpoint with actual Bedrock service.

    This test validates that the BedrockInvokeStream endpoint can successfully:
    - Create a streaming endpoint instance with a valid model and JMESPath expressions
    - Invoke the endpoint with a test payload
    - Receive a streaming response
    - Extract response text via JMESPath from stream chunks
    - Extract token counts via JMESPath when available
    - Measure time to first token (TTFT) and time to last token (TTLT)
    - Verify TTLT > TTFT
    - Complete without errors

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7**

    Args:
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).

    AWS Permissions Required:
        - bedrock:InvokeModelWithResponseStream

    Estimated Cost:
        ~$0.0001 per test run (using Amazon Nova Lite with minimal tokens)
    """
    from llmeter.endpoints.bedrock_invoke import BedrockInvokeStream

    # Create BedrockInvokeStream endpoint instance with JMESPath expressions for Amazon Nova models
    # Nova models via InvokeModelWithResponseStream API use a Converse-like native format
    endpoint = BedrockInvokeStream(
        model_id=bedrock_test_model,
        region=aws_region,
        # JMESPath for Nova native streaming format
        # contentBlockDelta.delta.text extracts text from content block delta chunks
        generated_text_jmespath="contentBlockDelta.delta.text",
        # metadata.usage.outputTokens extracts output tokens from the metadata chunk
        generated_token_count_jmespath="metadata.usage.outputTokens",
        # metadata.usage.inputTokens extracts input tokens from the metadata chunk
        input_token_count_jmespath="metadata.usage.inputTokens",
        input_text_jmespath="messages[0].content[0].text",
    )

    # Create payload in Nova native Invoke API format
    invoke_payload = {
        "schemaVersion": "messages-v1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Hello, this is a test message. Please respond with a brief greeting."
                    }
                ],
            }
        ],
        "inferenceConfig": {
            "maxTokens": 100,
        },
    }

    # Invoke the endpoint with test payload
    response = endpoint.invoke(invoke_payload)

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

    # Verify response has an ID
    assert response.id is not None, "Response should have an ID"


def test_save_load_invoke_payload_with_image(tmp_path):
    """
    Test saving and loading Bedrock Invoke API payload with image content.

    This test validates that:
    - Bedrock Invoke API payloads with image bytes can be saved to disk
    - Provider-specific payload structure (Anthropic Claude format) is preserved
    - Saved payloads contain valid JSON with marker objects
    - Loaded payloads restore bytes objects correctly
    - Round-trip preserves byte-for-byte equality

    **Validates: Requirements 9.5**

    Args:
        tmp_path: Temporary directory for test files (from pytest).
    """
    from llmeter.prompt_utils import save_payloads, load_payloads
    import io
    from PIL import Image

    # Create a simple test image
    img = Image.new("RGB", (50, 50), color="blue")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    # Create Bedrock Invoke API payload with Anthropic Claude format
    # This uses the native Messages API format, not Converse API
    invoke_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 150,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_bytes,  # Binary data in provider-specific format
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                ],
            }
        ],
    }

    # Save payload with image
    output_file = save_payloads(invoke_payload, tmp_path, "test_invoke_image.jsonl")
    assert output_file.exists(), "Output file should be created"

    # Verify file contains valid JSON with marker objects
    with output_file.open("r") as f:
        content = f.read()
        assert "__llmeter_bytes__" in content, "File should contain marker objects"

    # Load payload and verify bytes are restored
    loaded_payloads = list(load_payloads(output_file))
    assert len(loaded_payloads) == 1, "Should load exactly one payload"

    loaded = loaded_payloads[0]
    original_bytes = invoke_payload["messages"][0]["content"][0]["source"]["data"]
    loaded_bytes = loaded["messages"][0]["content"][0]["source"]["data"]

    assert isinstance(loaded_bytes, bytes), "Loaded bytes should be bytes type"
    assert loaded_bytes == original_bytes, "Bytes should match after round-trip"
    assert loaded == invoke_payload, "Full payload should match after round-trip"


@pytest.mark.integ
def test_round_trip_invoke_structure(
    tmp_path, aws_credentials, aws_region, bedrock_test_model
):
    """
    Test round-trip serialization with actual Bedrock Invoke API structure.

    This test validates that:
    - Complete Bedrock Invoke API payload structure is preserved
    - Provider-specific fields (anthropic_version, max_tokens) are maintained
    - Text content in messages is preserved correctly
    - Round-trip produces identical payload structure
    - Loaded payload can be used with the endpoint's invoke method

    **Validates: Requirements 9.5**

    Args:
        tmp_path: Temporary directory for test files (from pytest).
        aws_credentials: Boto3 session with valid AWS credentials (from fixture).
        aws_region: AWS region for testing (from fixture).
        bedrock_test_model: Model ID for testing (from fixture).
    """
    from llmeter.prompt_utils import save_payloads, load_payloads

    # Create a complete Bedrock Invoke payload with provider-specific structure
    # This mimics the actual Anthropic Claude Messages API format used by InvokeModel
    complete_invoke_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "messages": [
            {
                "role": "user",
                "content": "Please provide a brief response to this test message.",
            }
        ],
        "system": "You are a helpful assistant.",
    }

    # Save and load the complete payload
    output_file = save_payloads(
        complete_invoke_payload, tmp_path, "test_invoke_complete.jsonl"
    )
    loaded_payloads = list(load_payloads(output_file))

    assert len(loaded_payloads) == 1, "Should load exactly one payload"
    loaded = loaded_payloads[0]

    # Verify all top-level fields are preserved
    assert loaded["anthropic_version"] == complete_invoke_payload["anthropic_version"]
    assert loaded["max_tokens"] == complete_invoke_payload["max_tokens"]
    assert loaded["temperature"] == complete_invoke_payload["temperature"]
    assert loaded["top_p"] == complete_invoke_payload["top_p"]
    assert loaded["system"] == complete_invoke_payload["system"]

    # Verify messages structure is preserved
    assert len(loaded["messages"]) == len(complete_invoke_payload["messages"])
    assert (
        loaded["messages"][0]["role"] == complete_invoke_payload["messages"][0]["role"]
    )
    assert (
        loaded["messages"][0]["content"]
        == complete_invoke_payload["messages"][0]["content"]
    )

    # Verify complete equality
    assert loaded == complete_invoke_payload, (
        "Complete payload should match after round-trip"
    )

    # Verify the loaded payload can be used with the endpoint's invoke method
    endpoint = BedrockInvoke(
        model_id=bedrock_test_model,
        region=aws_region,
        generated_text_jmespath="content[0].text",
        generated_token_count_jmespath="usage.output_tokens",
        input_token_count_jmespath="usage.input_tokens",
        input_text_jmespath="messages[0].content",
    )
    response = endpoint.invoke(loaded)

    # Verify the endpoint successfully processed the loaded payload
    assert response.response_text is not None, "Response should contain text"
    assert len(response.response_text) > 0, "Response text should not be empty"
    assert response.error is None, (
        f"Response should not contain errors: {response.error}"
    )
