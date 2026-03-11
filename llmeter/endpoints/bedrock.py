# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMeter targets for testing the Amazon Bedrock **Converse and ConverseStream** APIs

Alternatively, see:

* [bedrock_invoke][llmeter.endpoints.bedrock_invoke] for testing Bedrock InvokeModel and
  InvokeModelWithResponseStream APIs, or
* [openai][llmeter.endpoints.openai] for testing OpenAI-compatible endpoints from
  [Bedrock Mantle](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html)
"""

import logging
import time
from typing import Any
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from ..prompt_utils import (
    detect_format_from_bytes,
    detect_format_from_file,
    read_file,
)
from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)


def _mime_to_format(mime_type: str) -> str | None:
    """Convert MIME type to Bedrock format string.

    Maps MIME types to format names used by Bedrock Converse API.

    Args:
        mime_type: MIME type (e.g., "image/jpeg", "application/pdf")

    Returns:
        str | None: Format string (e.g., "jpeg", "png", "pdf") or None if not recognized
    """
    mime_map = {
        "image/jpeg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "application/pdf": "pdf",
        "video/mp4": "mp4",
        "video/quicktime": "mov",
        "video/x-msvideo": "avi",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/ogg": "ogg",
    }
    return mime_map.get(mime_type)


def _build_content_blocks(
    user_message: str | list[str] | None,
    images: list[bytes] | list[str] | None,
    documents: list[bytes] | list[str] | None,
    videos: list[bytes] | list[str] | None,
    audio: list[bytes] | list[str] | None,
) -> list[dict]:
    """Build content blocks from parameters.

    Returns list of content block dictionaries in Bedrock Converse API format.

    Args:
        user_message: Text message(s)
        images: List of image bytes or file paths
        documents: List of document bytes or file paths
        videos: List of video bytes or file paths
        audio: List of audio bytes or file paths

    Returns:
        list[dict]: Content blocks

    Raises:
        ValueError: If format cannot be auto-detected from bytes
    """
    content = []

    # Add text blocks first
    if user_message:
        messages = [user_message] if isinstance(user_message, str) else user_message
        for msg in messages:
            content.append({"text": msg})

    # Add media blocks in order: images, videos, audio, documents
    for media_list, media_type in [
        (images, "image"),
        (videos, "video"),
        (audio, "audio"),
        (documents, "document"),
    ]:
        if media_list:
            for item in media_list:
                if isinstance(item, bytes):
                    # Bytes provided directly - detect MIME type from content
                    data = item
                    mime_type = detect_format_from_bytes(data)
                    if mime_type is None:
                        raise ValueError(
                            f"Cannot detect format from bytes for {media_type}. "
                            "Either install puremagic for content-based detection "
                            "or provide file path for extension-based detection."
                        )
                    fmt = _mime_to_format(mime_type)
                    if fmt is None:
                        raise ValueError(
                            f"Unsupported MIME type '{mime_type}' for {media_type}"
                        )
                else:
                    # File path - read and detect MIME type from file
                    data = read_file(item)
                    mime_type = detect_format_from_file(item)
                    if mime_type is None:
                        raise ValueError(f"Cannot detect format from file: {item}")
                    fmt = _mime_to_format(mime_type)
                    if fmt is None:
                        raise ValueError(
                            f"Unsupported MIME type '{mime_type}' for file: {item}"
                        )

                content.append({media_type: {"format": fmt, "source": {"bytes": data}}})

    return content


class BedrockBase(Endpoint):
    """Base class for interacting with Amazon Bedrock endpoints.

    This class provides core functionality for making requests to Amazon Bedrock
    endpoints, handling configuration and client initialization.

    Args:
        model_id (str): The identifier for the model to use
        endpoint_name (str | None, optional): Name of the endpoint. Defaults to None.
        region (str | None, optional): AWS region to use. Defaults to None.
        inference_config (dict | None, optional): Configuration for inference. Defaults to None.
        bedrock_boto3_client (boto3.client, optional): Pre-configured boto3 client. Defaults to None.
        max_attempts (int, optional): Maximum number of retry attempts. Defaults to 3.
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str | None = None,
        region: str | None = None,
        inference_config: dict | None = None,
        bedrock_boto3_client=None,
        max_attempts: int = 3,
    ):
        super().__init__(
            model_id=model_id,
            endpoint_name=endpoint_name or "amazon bedrock",
            provider="bedrock",
        )

        self.region = region or boto3.session.Session().region_name
        logger.info(f"Using AWS region: {self.region}")

        self._bedrock_client = bedrock_boto3_client
        if self._bedrock_client is None:
            config = Config(retries={"max_attempts": max_attempts, "mode": "standard"})
            self._bedrock_client = boto3.client(
                "bedrock-runtime", region_name=self.region, config=config
            )
        self._inference_config = inference_config

    def _parse_payload(self, payload):
        """
        Parse the payload to extract text content.

        Args:
            payload (dict): The payload containing messages.

        Returns:
            str: Concatenated text content from the messages.

        Raises:
            TypeError: If payload is not a dictionary
            KeyError: If required fields are missing from payload
        """
        try:
            if not isinstance(payload, dict):
                raise TypeError("Payload must be a dictionary")

            messages = payload.get("messages", [])
            if not isinstance(messages, list):
                raise TypeError("Messages must be a list")

            texts = []
            for msg in messages:
                if not isinstance(msg, dict):
                    raise TypeError("Each message must be a dictionary")

                for content in msg.get("content", []):
                    if not isinstance(content, dict):
                        raise TypeError("Content must be a dictionary")

                    text_content = content.get("text")
                    if isinstance(text_content, list):
                        texts.extend(text_content)
                    else:
                        texts.append(text_content or "")

            return "\n".join(filter(None, texts))

        except (TypeError, KeyError) as e:
            logger.error(f"Error parsing payload: {e}")
            return ""

        except Exception as e:
            logger.error(f"Unexpected error parsing payload: {e}")
            return ""

    @staticmethod
    def create_payload(
        user_message: str | list[str] | None = None,
        max_tokens: int | None = None,
        *,
        images: list[bytes] | list[str] | None = None,
        documents: list[bytes] | list[str] | None = None,
        videos: list[bytes] | list[str] | None = None,
        audio: list[bytes] | list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Create a payload for the Bedrock Converse API request with optional multi-modal content.

        ⚠️ SECURITY WARNING: Format detection is for testing/development convenience ONLY.
        This method does NOT validate file safety, integrity, or protect against malicious
        content. DO NOT use with untrusted files (user uploads, external sources) without
        proper validation, sanitization, and security measures.

        Args:
            user_message (str | list[str] | None): The user's message or a sequence of messages.
            max_tokens (int | None): The maximum number of tokens to generate. Defaults to 256.
            images (list[bytes] | list[str] | None): List of image bytes or file paths (keyword-only).
            documents (list[bytes] | list[str] | None): List of document bytes or file paths (keyword-only).
            videos (list[bytes] | list[str] | None): List of video bytes or file paths (keyword-only).
            audio (list[bytes] | list[str] | None): List of audio bytes or file paths (keyword-only).
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The formatted payload for the Bedrock API request.

        Raises:
            TypeError: If parameters have invalid types
            ValueError: If parameters have invalid values
            FileNotFoundError: If a file path doesn't exist
            IOError: If a file cannot be read

        Security:
            - Format detection (puremagic/extension) is NOT security validation
            - Malicious files can have misleading extensions or forged magic bytes
            - This method does NOT scan for malware or sanitize content
            - Users MUST validate and sanitize untrusted files before calling this method
            - Intended for testing/development with trusted files only
            - NOT intended for production user uploads without proper security measures

        Examples:
            # Text only (backward compatible)
            create_payload(user_message="Hello")
            create_payload("Hello", 256)  # Positional args still work

            # Single image from file path (trusted source)
            create_payload(
                user_message="What's in this image?",
                images=["photo.jpg"]
            )

            # Multiple images from bytes (trusted source)
            create_payload(
                user_message="Compare these images",
                images=[image_bytes1, image_bytes2]
            )

            # Mixed content (trusted source)
            create_payload(
                user_message="Analyze this",
                images=["chart.png"],
                documents=["report.pdf"]
            )
        """
        # Set default for max_tokens if not provided
        if max_tokens is None:
            max_tokens = 256

        # Check if any multi-modal content is provided
        has_multimodal = any([images, documents, videos, audio])

        # Backward compatibility: if only user_message provided, use old logic
        if not has_multimodal:
            if user_message is None:
                raise ValueError("user_message is required when no media is provided")

            if not isinstance(user_message, (str, list)):
                raise TypeError("user_message must be a string or list of strings")

            if isinstance(user_message, list):
                if not all(isinstance(msg, str) for msg in user_message):
                    raise TypeError("All messages must be strings")
                if not user_message:
                    raise ValueError("user_message list cannot be empty")

            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer")

            if isinstance(user_message, str):
                user_message = [user_message]

            try:
                payload: dict = {
                    "messages": [
                        {"role": "user", "content": [{"text": k}]} for k in user_message
                    ],
                }
                payload.update(kwargs)
                if payload.get("inferenceConfig") is None:
                    payload["inferenceConfig"] = {}

                payload["inferenceConfig"] = {
                    **payload["inferenceConfig"],
                    "maxTokens": max_tokens,
                }
                return payload

            except Exception as e:
                logger.error(f"Error creating payload: {e}")
                raise RuntimeError(f"Failed to create payload: {str(e)}")

        # Multi-modal path: validate types
        if images is not None and not isinstance(images, list):
            raise TypeError("images must be a list")
        if documents is not None and not isinstance(documents, list):
            raise TypeError("documents must be a list")
        if videos is not None and not isinstance(videos, list):
            raise TypeError("videos must be a list")
        if audio is not None and not isinstance(audio, list):
            raise TypeError("audio must be a list")

        # Validate list items are bytes or str
        for media_list, media_name in [
            (images, "images"),
            (documents, "documents"),
            (videos, "videos"),
            (audio, "audio"),
        ]:
            if media_list:
                for item in media_list:
                    if not isinstance(item, (bytes, str)):
                        raise TypeError(
                            f"Items in {media_name} list must be bytes or str (file path), "
                            f"got {type(item).__name__}"
                        )

        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        try:
            # Build content blocks
            content_blocks = _build_content_blocks(
                user_message, images, documents, videos, audio
            )

            payload: dict = {
                "messages": [{"role": "user", "content": content_blocks}],
            }
            payload.update(kwargs)
            if payload.get("inferenceConfig") is None:
                payload["inferenceConfig"] = {}

            payload["inferenceConfig"] = {
                **payload["inferenceConfig"],
                "maxTokens": max_tokens,
            }
            return payload

        except Exception as e:
            logger.error(f"Error creating payload: {e}")
            raise


class BedrockConverse(BedrockBase):
    def _parse_converse_response(self, response: dict) -> InvocationResponse:
        """
        Parse the response from a Bedrock converse API call.

        Args:
            response (dict): Raw response from the Bedrock API containing output text and metadata

        Returns:
            InvocationResponse: Parsed response containing the generated text and metadata

        Raises:
            KeyError: If required fields are missing from the response
            TypeError: If response fields have unexpected types
        """
        try:
            # Direct dictionary access and single-level assignment for better performance
            output = response["output"]["message"]["content"][0]["text"]
            if not isinstance(output, str):
                raise TypeError("Expected string for output text")

            usage = response.get("usage", {})
            retries = response["ResponseMetadata"]["RetryAttempts"]

            return InvocationResponse(
                id=uuid4().hex,
                response_text=output,
                num_tokens_input=usage.get("inputTokens"),
                num_tokens_output=usage.get("outputTokens"),
                retries=retries,
            )

        except KeyError as e:
            logger.error(f"Missing required field in response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Missing required field: {e}",
            )

        except TypeError as e:
            logger.error(f"Unexpected type in response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Type error in response: {e}",
            )

        except Exception as e:
            logger.error(f"Error parsing converse response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Response parsing error: {e}",
            )

    def invoke(self, payload: dict, **kwargs: Any) -> InvocationResponse:
        """
        Invoke the Bedrock converse API with the given payload.

        Args:
            payload (dict): The payload containing the request parameters
            **kwargs: Additional keyword arguments to include in the payload

        Returns:
            InvocationResponse: Response object containing generated text and metadata

        Raises:
            ClientError: If there is an error calling the Bedrock API
            ValueError: If payload is invalid
            TypeError: If payload is not a dictionary
        """
        if not isinstance(payload, dict):
            raise TypeError("Payload must be a dictionary")

        try:
            payload = {**kwargs, **payload}
            if payload.get("inferenceConfig") is None:
                payload["inferenceConfig"] = self._inference_config or {}

            payload["modelId"] = self.model_id
            try:
                start_t = time.perf_counter()
                client_response = self._bedrock_client.converse(**payload)  # type: ignore
                time_to_last_token = time.perf_counter() - start_t
            except ClientError as e:
                logger.error(f"Bedrock API error: {e}")
                return InvocationResponse.error_output(
                    input_payload=payload, id=uuid4().hex, error=str(e)
                )
            except Exception as e:
                logger.error(f"Unexpected error during API call: {e}")
                return InvocationResponse.error_output(
                    input_payload=payload, id=uuid4().hex, error=str(e)
                )

            response = self._parse_converse_response(client_response)  # type: ignore
            response.input_payload = payload
            response.input_prompt = self._parse_payload(payload)
            response.time_to_last_token = time_to_last_token
            return response

        except Exception as e:
            logger.error(f"Error in invoke method: {e}")
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )


class BedrockConverseStream(BedrockConverse):
    def invoke(self, payload: dict, **kwargs: Any) -> InvocationResponse:
        payload = {**kwargs, **payload}
        if payload.get("inferenceConfig") is None:
            payload["inferenceConfig"] = self._inference_config or {}

        payload["modelId"] = self.model_id
        start_t = time.perf_counter()
        try:
            client_response = self._bedrock_client.converse_stream(**payload)  # type: ignore
        except (ClientError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        response = self._parse_conversation_stream(client_response, start_t)
        response.input_payload = payload
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_conversation_stream(
        self, client_response, start_t: float
    ) -> InvocationResponse:
        """
        Parse the streaming response from Bedrock conversation API.

        Args:
            client_response (dict): The raw response from the Bedrock API
            start_t (float): The timestamp when the request was initiated

        Returns:
            InvocationResponse: Parsed response containing the generated text and metadata

        Raises:
            KeyError: If required fields are missing from the response
            TypeError: If response fields have unexpected types
        """
        time_flag = True
        time_to_first_token = None
        time_to_last_token = None
        output_text = ""
        metadata = None

        try:
            for chunk in client_response["stream"]:
                if "contentBlockDelta" in chunk:
                    delta_text = chunk["contentBlockDelta"]["delta"].get("text", "")
                    if not isinstance(delta_text, str):
                        raise TypeError("Expected string for delta text")
                    output_text += delta_text or ""
                    if time_flag:
                        time_to_first_token = time.perf_counter() - start_t
                        time_flag = False

                if "contentBlockStop" in chunk:
                    time_to_last_token = time.perf_counter() - start_t

                if "metadata" in chunk:
                    metadata = chunk["metadata"]

            response = InvocationResponse(
                id=uuid4().hex,
                response_text=output_text,
                time_to_last_token=time_to_last_token,
                time_to_first_token=time_to_first_token,
            )

            if metadata:
                # The latency provided by Bedrock is at the service endpoint time, not client side
                # time_to_last_token = metadata.get("metrics", {}).get("latencyMs")
                try:
                    usage = metadata.get("usage", {})
                    response.num_tokens_input = usage.get("inputTokens")
                    response.num_tokens_output = usage.get("outputTokens")
                except Exception as e:
                    logger.error(f"Error parsing metadata: {e}")
                    return InvocationResponse.error_output(
                        id=uuid4().hex,
                        error=f"Metadata parsing error: {e}",
                    )

            response.retries = client_response["ResponseMetadata"]["RetryAttempts"]

            return response

        except KeyError as e:
            logger.error(f"Missing required field in response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Missing required field: {e}",
            )
        except TypeError as e:
            logger.error(f"Unexpected type in response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Type error in response: {e}",
            )
        except Exception as e:
            logger.error(f"Error parsing conversation stream: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Stream parsing error: {e}",
            )
