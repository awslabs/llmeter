# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Dict, Sequence
from uuid import uuid4

import jmespath
from openai import APIConnectionError, OpenAI
from openai.types.chat import ChatCompletion

from .base import Endpoint, InvocationResponse
from ..prompt_utils import read_file, detect_format_from_bytes, detect_format_from_file

logger = logging.getLogger(__name__)


def _mime_to_openai_format(mime_type: str) -> str | None:
    """Convert MIME type to OpenAI format string.

    OpenAI expects full MIME types for media content.

    Args:
        mime_type: MIME type (e.g., "image/jpeg", "application/pdf")

    Returns:
        str | None: Format string (full MIME type) or None if not recognized
    """
    # OpenAI uses full MIME types
    supported_mimes = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "application/pdf",
        "video/mp4",
        "audio/mpeg",
        "audio/wav",
    }
    return mime_type if mime_type in supported_mimes else None


def _build_content_blocks_openai(
    user_message: str | list[str] | None,
    images: list[bytes] | list[str] | None,
    documents: list[bytes] | list[str] | None,
    videos: list[bytes] | list[str] | None,
    audio: list[bytes] | list[str] | None,
) -> list[dict]:
    """Build content blocks from parameters for OpenAI API.

    Returns list of content block dictionaries in OpenAI format.

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
                    fmt = _mime_to_openai_format(mime_type)
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
                    fmt = _mime_to_openai_format(mime_type)
                    if fmt is None:
                        raise ValueError(
                            f"Unsupported MIME type '{mime_type}' for file: {item}"
                        )

                content.append({media_type: {"format": fmt, "source": {"bytes": data}}})

    return content


logger = logging.getLogger(__name__)


class OpenAIEndpoint(Endpoint):
    """Base class for OpenAI API endpoints.

    Provides common functionality for interacting with OpenAI's API endpoints.
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "openai",
        api_key: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        """Initialize OpenAI endpoint.

        Args:
            model_id (str): ID of the OpenAI model to use
            endpoint_name (str, optional): Name of the endpoint. Defaults to "openai".
            api_key (str | None, optional): OpenAI API key. Defaults to None.
            provider (str, optional): Provider name. Defaults to "openai".
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(
            endpoint_name,
            model_id,
            provider=provider,
        )

        self._client = OpenAI(api_key=api_key, **kwargs)

    def _parse_payload(self, payload):
        """Parse the message content from the payload.

        Args:
            payload (dict): Request payload containing messages

        Returns:
            str: Concatenated message contents
        """
        jmes_path = "[:].content"
        messages = payload.get("messages")
        result = jmespath.search(jmes_path, messages)
        if result is None:
            return ""
        return "\n".join(result)

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str] | None = None,
        max_tokens: int = 256,
        *,
        images: list[bytes] | list[str] | None = None,
        documents: list[bytes] | list[str] | None = None,
        videos: list[bytes] | list[str] | None = None,
        audio: list[bytes] | list[str] | None = None,
        **kwargs,
    ):
        """Create a payload for the OpenAI API request with optional multi-modal content.

        ⚠️ SECURITY WARNING: Format detection is for testing/development convenience ONLY.
        This method does NOT validate file safety, integrity, or protect against malicious
        content. DO NOT use with untrusted files (user uploads, external sources) without
        proper validation, sanitization, and security measures.

        Args:
            user_message (str | Sequence[str] | None): User message(s) to send
            max_tokens (int, optional): Maximum tokens in response. Defaults to 256.
            images (list[bytes] | list[str] | None): List of image bytes or file paths (keyword-only).
            documents (list[bytes] | list[str] | None): List of document bytes or file paths (keyword-only).
            videos (list[bytes] | list[str] | None): List of video bytes or file paths (keyword-only).
            audio (list[bytes] | list[str] | None): List of audio bytes or file paths (keyword-only).
            **kwargs: Additional payload parameters

        Returns:
            dict: Formatted payload for API request

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
        # Check if any multi-modal content is provided
        has_multimodal = any([images, documents, videos, audio])

        # Backward compatibility: if only user_message provided, use old logic
        if not has_multimodal:
            if isinstance(user_message, str):
                user_message = [user_message]
            payload = {
                "messages": [{"role": "user", "content": k} for k in user_message],
                "max_tokens": max_tokens,
            }
            payload.update(kwargs)
            return payload

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
            content_blocks = _build_content_blocks_openai(
                user_message, images, documents, videos, audio
            )

            payload = {
                "messages": [{"role": "user", "content": content_blocks}],
                "max_tokens": max_tokens,
            }
            payload.update(kwargs)
            return payload

        except Exception as e:
            logger.error(f"Error creating payload: {e}")
            raise


class OpenAICompletionEndpoint(OpenAIEndpoint):
    """Endpoint for OpenAI chat completion API."""

    def invoke(self, payload: Dict, **kwargs) -> InvocationResponse:
        """Invoke the OpenAI chat completion API.

        Args:
            payload (Dict): Request payload
            **kwargs: Additional parameters for the request

        Returns:
            InvocationResponse: Response from the API
        """
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id

        start_t = time.perf_counter()
        try:
            client_response: ChatCompletion = self._client.chat.completions.create(
                **payload
            )

        except (APIConnectionError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )

        response = self._parse_converse_response(client_response, start_t)
        response.input_payload = payload
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_converse_response(self, client_response: ChatCompletion, start_t: float):
        """Parse the OpenAI chat completion API response into an InvocationResponse object.

        Args:
            client_response (ChatCompletion): Raw response from OpenAI chat completion API
            start_t (float): Start time of the API call in seconds

        Returns:
            InvocationResponse: Parsed response object containing:
                - Response ID
                - Response text content
                - Token counts for input/output
                - Time to last token
        """

        usage = client_response.usage

        return InvocationResponse(
            id=client_response.id,
            response_text=client_response.choices[0].message.content,
            num_tokens_input=usage and usage.prompt_tokens,
            num_tokens_output=usage and usage.completion_tokens,
            time_to_last_token=time.perf_counter() - start_t,
        )


class OpenAICompletionStreamEndpoint(OpenAIEndpoint):
    """Endpoint for OpenAI streaming chat completion API."""

    def invoke(self, payload: Dict, **kwargs) -> InvocationResponse:
        """Invoke the OpenAI streaming chat completion API.

        Args:
            payload (Dict): Request payload
            **kwargs: Additional parameters for the request

        Returns:
            InvocationResponse: Response from the API
        """
        payload = {**kwargs, **payload}

        payload["model"] = self.model_id

        if not payload.get("stream"):
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        try:
            start_t = time.perf_counter()
            client_response: ChatCompletion = self._client.chat.completions.create(
                **payload
            )
        except (APIConnectionError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        response = self._parse_converse_stream_response(client_response, start_t)
        response.input_payload = payload
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_converse_stream_response(
        self, client_response: ChatCompletion, start_t: float
    ) -> InvocationResponse:
        """Parse the streaming API response from OpenAI chat completion API.

        Args:
            client_response (ChatCompletion): Raw API response stream containing chunks of completion text
            start_t (float): Start time of the API call in seconds

        Returns:
            InvocationResponse: Parsed response object containing:
                - Response ID
                - Concatenated response text from all chunks
                - Token counts for input/output
                - Time to first token and last token
        """

        prompt_tokens = None
        completion_tokens = None

        first_chunk = next(client_response)  # type: ignore
        time_to_first_token = time.perf_counter() - start_t
        response_id = first_chunk.id  # type: ignore
        response_text = first_chunk.choices[0].delta.content or ""

        for chunk in client_response:
            if chunk.choices and chunk.choices[0].delta.content:  # type: ignore
                response_text += chunk.choices[0].delta.content or ""  # type: ignore
            if hasattr(chunk, "usage") and chunk.usage is not None:  # type: ignore
                prompt_tokens = chunk.usage.prompt_tokens  # type: ignore
                completion_tokens = chunk.usage.completion_tokens  # type: ignore
        time_to_last_token = time.perf_counter() - start_t

        return InvocationResponse(
            id=response_id,
            response_text=response_text,
            num_tokens_input=prompt_tokens,
            num_tokens_output=completion_tokens,
            time_to_first_token=time_to_first_token,
            time_to_last_token=time_to_last_token,
        )
