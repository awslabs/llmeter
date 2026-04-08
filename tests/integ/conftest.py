# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures and configuration for integration tests.

This module provides session-scoped fixtures for AWS credentials, configuration,
and test utilities used across all integration tests.

The fixtures in this module are used by all integration test files to:
- Verify AWS credentials are available (skips tests if not)
- Configure AWS region for testing
- Provide test model IDs for different endpoint types
- Generate test payloads (text-only and multimodal with images)

Integration tests are marked with @pytest.mark.integ and are skipped by default
to avoid AWS costs and credential requirements during regular development.

To run integration tests:
    uv run pytest -m integ

Required AWS Permissions:
    - bedrock:InvokeModel (for non-streaming tests)
    - bedrock:InvokeModelWithResponseStream (for streaming tests)
    - sts:GetCallerIdentity (for credential verification)

Estimated Cost:
    - Fixtures themselves have no cost
    - Each test using these fixtures costs ~$0.0001-$0.0002
    - Full integration test suite costs ~$0.0012 per run

Environment Variables:
    - AWS_REGION: AWS region for testing (default: us-east-1)
    - BEDROCK_TEST_MODEL: Model ID for Converse/Invoke tests
      (default: us.anthropic.claude-3-5-sonnet-20241022-v2:0)
    - BEDROCK_OPENAI_TEST_MODEL: Model ID for OpenAI SDK tests
      (default: openai.gpt-oss-20b-1:0)
"""

import os

import boto3
import pytest
from botocore.exceptions import ClientError, NoCredentialsError


@pytest.fixture(scope="session")
def aws_credentials():
    """
    Verify AWS credentials are available and skip tests if not.

    This fixture attempts to verify AWS credentials by calling STS GetCallerIdentity.
    If credentials are not available or invalid, all integration tests will be skipped.

    Returns:
        boto3.Session: A boto3 session with valid AWS credentials.

    Raises:
        pytest.skip: If AWS credentials are not available or invalid.
    """
    try:
        session = boto3.Session()
        sts = session.client("sts")
        sts.get_caller_identity()
        return session
    except (NoCredentialsError, ClientError) as e:
        pytest.skip(
            f"AWS credentials not available: {e}. "
            "Set up AWS CLI or provide credentials to run integration tests."
        )


@pytest.fixture(scope="session")
def aws_region():
    """
    Get AWS region from environment or default to us-east-1.

    Returns:
        str: AWS region name for testing.
    """
    return os.environ.get("AWS_REGION", "us-east-1")


@pytest.fixture(scope="session")
def bedrock_test_model():
    """
    Get test model ID for Converse/Invoke tests.

    The model ID can be overridden via the BEDROCK_TEST_MODEL environment variable.
    Defaults to Claude 3.5 Sonnet v2, which is widely available and cost-effective.

    Returns:
        str: Bedrock model ID for testing.
    """
    return os.environ.get(
        "BEDROCK_TEST_MODEL", "global.amazon.nova-2-lite-v1:0"
    )


@pytest.fixture(scope="session")
def bedrock_openai_test_model():
    """
    Get test model ID for OpenAI SDK tests.

    The model ID can be overridden via the BEDROCK_OPENAI_TEST_MODEL environment variable.
    Defaults to openai.gpt-oss-20b-1:0 which is supported by Bedrock's OpenAI-compatible endpoint.

    Note: Bedrock OpenAI-compatible endpoint only supports OpenAI models available in Bedrock.
    Use the full Bedrock model ID format (e.g., openai.gpt-oss-20b-1:0).

    Returns:
        str: OpenAI model ID for Bedrock OpenAI SDK testing.
    """
    return os.environ.get(
        "BEDROCK_OPENAI_TEST_MODEL", "openai.gpt-oss-20b-1:0"
    )


@pytest.fixture(scope="session")
def bedrock_openai_multimodal_test_model():
    """
    Get test model ID for OpenAI SDK multimodal tests.

    The model ID can be overridden via the BEDROCK_OPENAI_MULTIMODAL_TEST_MODEL environment variable.
    Defaults to qwen.qwen3-vl-235b-a22b-instruct which supports images and is available in Bedrock.

    Note: This model is specifically for testing multimodal content (images, video, etc.)
    via Bedrock's OpenAI-compatible endpoint.

    Returns:
        str: OpenAI multimodal model ID for Bedrock OpenAI SDK testing.
    """
    return os.environ.get(
        "BEDROCK_OPENAI_MULTIMODAL_TEST_MODEL", "qwen.qwen3-vl-235b-a22b-instruct"
    )


@pytest.fixture
def test_image_bytes():
    """
    Create test images as binary JPEG data for multimodal testing.

    Returns two small JPEG images (32x32 pixels) as bytes objects.
    These are used to test binary content serialization and API calls.
    JPEG format is used for broad model compatibility.

    Returns:
        tuple: (image1_bytes, image2_bytes) - Two JPEG images as bytes
    """
    import io
    from PIL import Image

    # 32x32 red square JPEG - binary format
    img1 = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buf1 = io.BytesIO()
    img1.save(buf1, format="JPEG")
    image1_bytes = buf1.getvalue()

    # 32x32 blue square JPEG - binary format
    img2 = Image.new("RGB", (32, 32), color=(0, 0, 255))
    buf2 = io.BytesIO()
    img2.save(buf2, format="JPEG")
    image2_bytes = buf2.getvalue()

    return image1_bytes, image2_bytes


@pytest.fixture(scope="session")
def bedrock_openai_endpoint_url(aws_region):
    """
    Construct Bedrock OpenAI-compatible endpoint URL for standard models.

    Args:
        aws_region: AWS region from the aws_region fixture.

    Returns:
        str: Bedrock OpenAI-compatible endpoint URL.
    """
    return f"https://bedrock-runtime.{aws_region}.amazonaws.com/openai/v1"


@pytest.fixture(scope="session")
def bedrock_openai_multimodal_endpoint_url(aws_region):
    """
    Construct Bedrock OpenAI-compatible endpoint URL for multimodal models (Bedrock Mantle).

    This endpoint supports non-OpenAI models (e.g., Qwen, Kimi) via the OpenAI-compatible
    interface and is required for multimodal content (images, video).

    Args:
        aws_region: AWS region from the aws_region fixture.

    Returns:
        str: Bedrock Mantle endpoint URL for multimodal testing.
    """
    return f"https://bedrock-mantle.{aws_region}.api.aws/v1"


@pytest.fixture
def test_payload():
    """
    Create a simple text-only test payload for model invocation.

    This payload is designed to be minimal, deterministic, and cost-effective.
    It works across all model types and generates predictable responses.

    Returns:
        dict: A test payload with a simple text message.
    """
    return {
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
        "inferenceConfig": {"maxTokens": 100},
    }


@pytest.fixture
def test_payload_with_image():
    """
    Create a test payload with image for multimodal testing.

    This payload includes a 200x200 PNG test image with a distinctive two-color
    split (red left half, blue right half) and a constrained prompt asking the
    model to identify the colors.  This design makes it easy to verify the model
    actually processed the image pixels rather than giving a generic answer.

    The image is generated programmatically to keep the test deterministic
    and avoid external dependencies.

    Returns:
        dict: A test payload with image and text content.
    """
    import io

    from PIL import Image

    # Create a 200x200 image: left half red, right half blue
    img = Image.new("RGB", (200, 200), color="red")
    for x in range(100, 200):
        for y in range(200):
            img.putpixel((x, y), (0, 0, 255))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": img_bytes},
                        }
                    },
                    {
                        "text": (
                            "This image is split into two halves of different colors. "
                            "What are the two colors? Answer with just the color names."
                        ),
                    },
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 100},
    }
