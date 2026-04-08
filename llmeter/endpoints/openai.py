# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMeter targets for testing OpenAI ChatCompletions-compatible endpoints (wherever they're hosted)"""

import base64
import logging
import os
import time
from typing import Any, Dict, Sequence
from uuid import uuid4

import jmespath
from openai import APIConnectionError, OpenAI
from openai.types.chat import ChatCompletion

from .base import Endpoint, InvocationResponse
from ..prompt_utils import read_file, detect_format_from_bytes, detect_format_from_file

logger = logging.getLogger(__name__)

# MIME types supported by OpenAI, grouped by content part type
_OPENAI_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_OPENAI_AUDIO_MIMES = {"audio/mpeg", "audio/wav"}
_OPENAI_FILE_MIMES = {"application/pdf"}

# Map MIME → OpenAI audio "format" field value
_MIME_TO_OPENAI_AUDIO_FMT: dict[str, str] = {
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
}


def _mime_to_openai_format(mime_type: str) -> dict | None:
    """Convert a MIME type and raw bytes into an OpenAI content-part dict.

    Returns a *partial* content block (without the data filled in) so the caller
    can see which content-part shape to use, or ``None`` if the MIME type is not
    supported by OpenAI.

    The returned dict has the ``"type"`` key already set and the nested structure
    ready for the caller to inject base64 data.

    Supported mappings:

    * ``image/*``  → ``{"type": "image_url", ...}``
    * ``audio/*``  → ``{"type": "input_audio", ...}``
    * ``application/pdf`` → ``{"type": "file", ...}``
    """
    if mime_type in _OPENAI_IMAGE_MIMES:
        return {"_kind": "image", "mime": mime_type}
    if mime_type in _OPENAI_AUDIO_MIMES:
        return {"_kind": "audio", "fmt": _MIME_TO_OPENAI_AUDIO_FMT[mime_type]}
    if mime_type in _OPENAI_FILE_MIMES:
        return {"_kind": "file", "mime": mime_type}
    return None


def _make_openai_content_block(
    data: bytes,
    mime_type: str,
    *,
    filename: str | None = None,
) -> dict:
    """Build a single OpenAI content-part dict from raw bytes and MIME type.

    Args:
        data: Raw binary content.
        mime_type: Detected MIME type (must be in the supported set).
        filename: Optional filename hint (used for file/document parts).

    Returns:
        A dict ready to be appended to the ``content`` list of a message.

    Raises:
        ValueError: If *mime_type* is not supported by OpenAI.
    """
    info = _mime_to_openai_format(mime_type)
    if info is None:
        raise ValueError(f"Unsupported MIME type for OpenAI: '{mime_type}'")

    b64 = base64.b64encode(data).decode("utf-8")
    kind = info["_kind"]

    if kind == "image":
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
        }
    if kind == "audio":
        return {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": info["fmt"]},
        }
    # kind == "file"
    block: dict = {
        "type": "file",
        "file": {"file_data": f"data:{mime_type};base64,{b64}"},
    }
    if filename:
        block["file"]["filename"] = filename
    return block


def _build_content_blocks_openai(
    user_message: str | list[str] | None,
    images: list[bytes] | list[str] | None,
    documents: list[bytes] | list[str] | None,
    videos: list[bytes] | list[str] | None,
    audio: list[bytes] | list[str] | None,
) -> list[dict]:
    """Build content blocks from parameters for OpenAI Chat Completions API.

    Produces content-part dicts that conform to the OpenAI SDK types:

    * Text → ``{"type": "text", "text": "..."}``
    * Image → ``{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}``
    * Audio → ``{"type": "input_audio", "input_audio": {"data": "...", "format": "wav"|"mp3"}}``
    * Document (PDF) → ``{"type": "file", "file": {"file_data": "data:application/pdf;base64,..."}}``

    Args:
        user_message: Text message(s).
        images: Image bytes or file paths.
        documents: Document bytes or file paths (currently only PDF is supported by OpenAI).
        videos: Video bytes or file paths.  OpenAI Chat Completions does not support
            inline video; a ``ValueError`` is raised if any are provided.
        audio: Audio bytes or file paths.

    Returns:
        list[dict]: Content blocks.

    Raises:
        ValueError: If format cannot be detected, MIME type is unsupported, or
            video content is provided (not supported by OpenAI inline).
    """
    if videos:
        raise ValueError(
            "OpenAI Chat Completions API does not support inline video content. "
            "Consider extracting frames as images instead."
        )

    content: list[dict] = []

    # Text blocks
    if user_message:
        messages = [user_message] if isinstance(user_message, str) else user_message
        for msg in messages:
            content.append({"type": "text", "text": msg})

    # Media blocks — images, audio, documents
    for media_list, media_label in [
        (images, "image"),
        (audio, "audio"),
        (documents, "document"),
    ]:
        if not media_list:
            continue
        for item in media_list:
            if isinstance(item, bytes):
                data = item
                mime_type = detect_format_from_bytes(data)
                if mime_type is None:
                    raise ValueError(
                        f"Cannot detect format from bytes for {media_label}. "
                        "Either install puremagic for content-based detection "
                        "or provide a file path for extension-based detection."
                    )
            else:
                data = read_file(item)
                mime_type = detect_format_from_file(item)
                if mime_type is None:
                    raise ValueError(f"Cannot detect format from file: {item}")

            filename = os.path.basename(item) if isinstance(item, str) else None
            content.append(
                _make_openai_content_block(data, mime_type, filename=filename)
            )

    return content


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
        **kwargs: Any,
    ):
        """Initialize OpenAI endpoint.

        Args:
            model_id (str): ID of the OpenAI model to use
            endpoint_name (str, optional): Name of the endpoint. Defaults to "openai".
            api_key (str | None, optional): OpenAI API key. Defaults to None.
            provider (str, optional): Provider name. Defaults to "openai".
            **kwargs (Any): Additional arguments passed to OpenAI client
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
        **kwargs: Any,
    ) -> dict:
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
    """Endpoint for OpenAI-compatible Chat Completion APIs (non-streaming mode)"""

    def invoke(self, payload: Dict, **kwargs: Any) -> InvocationResponse:
        """Invoke the OpenAI chat completion API.

        Args:
            payload (Dict): Request payload
            **kwargs (Any): Additional parameters for the request

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
    """Endpoint for OpenAI-compatible Chat Completion APIs (streaming mode)"""

    def invoke(self, payload: Dict, **kwargs: Any) -> InvocationResponse:
        """Invoke the OpenAI streaming chat completion API.

        Args:
            payload (Dict): Request payload
            **kwargs (Any): Additional parameters for the request

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
