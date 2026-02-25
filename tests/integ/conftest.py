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
def bedrock_openai_endpoint_url(aws_region):
    """
    Construct Bedrock OpenAI-compatible endpoint URL.

    Args:
        aws_region: AWS region from the aws_region fixture.

    Returns:
        str: Bedrock OpenAI-compatible endpoint URL.
    """
    return f"https://bedrock-runtime.{aws_region}.amazonaws.com/openai/v1"


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

    This payload includes a small 100x100 PNG test image (a simple red square)
    encoded as base64, along with a text prompt asking the model to describe
    the image.

    The image is generated programmatically to keep the test deterministic
    and avoid external dependencies.

    Returns:
        dict: A test payload with image and text content.
    """
    import io

    from PIL import Image

    # Create a simple 100x100 red square PNG image
    img = Image.new("RGB", (100, 100), color="red")
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
                        "text": "What do you see in this image? Please provide a brief description."
                    },
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 150},
    }
