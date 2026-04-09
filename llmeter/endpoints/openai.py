# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMeter targets for testing OpenAI ChatCompletions-compatible endpoints (wherever they're hosted)"""

import base64
import logging
import time
from typing import Any, Dict
from uuid import uuid4

import jmespath
from openai import APIConnectionError, OpenAI
from openai.types.chat import ChatCompletion

from ..prompt_utils import (
    ContentItem,
    MediaContent,
    VideoContent,
)
from .base import Endpoint, InvocationResponse

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


def _make_openai_content_block(item: MediaContent) -> dict:
    """Build a single OpenAI content-part dict from a MediaContent object.

    Raises:
        ValueError: If the MIME type is not supported by OpenAI, or if
            ``VideoContent`` is provided (not supported inline).
    """
    if isinstance(item, VideoContent):
        raise ValueError(
            "OpenAI Chat Completions API does not support inline video content. "
            "Consider extracting frames as images instead."
        )

    mime = item.mime_type
    b64 = base64.b64encode(item.data).decode("utf-8")

    if mime in _OPENAI_IMAGE_MIMES:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }
    if mime in _OPENAI_AUDIO_MIMES:
        return {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": _MIME_TO_OPENAI_AUDIO_FMT[mime]},
        }
    if mime in _OPENAI_FILE_MIMES:
        block: dict = {
            "type": "file",
            "file": {"file_data": f"data:{mime};base64,{b64}"},
        }
        if item.source_path:
            import os

            block["file"]["filename"] = os.path.basename(item.source_path)
        return block

    raise ValueError(f"Unsupported MIME type for OpenAI: '{mime}'")


def _build_content_blocks_openai(items: list[ContentItem]) -> list[dict]:
    """Convert an ordered list of content items to OpenAI content blocks.

    Args:
        items: Ordered sequence of strings and/or ``MediaContent`` objects.

    Returns:
        list[dict]: Content blocks conforming to OpenAI SDK types.

    Raises:
        ValueError: If a MIME type is unsupported or video is provided.
        TypeError: If an item has an unexpected type.
    """
    blocks: list[dict] = []
    for item in items:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, MediaContent):
            blocks.append(_make_openai_content_block(item))
        else:
            raise TypeError(
                f"Content items must be str or MediaContent, got {type(item).__name__}"
            )
    return blocks


class OpenAIEndpoint(Endpoint):
    """Base class for OpenAI API endpoints."""

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
            model_id: ID of the OpenAI model to use
            endpoint_name: Name of the endpoint. Defaults to "openai".
            api_key: OpenAI API key. Defaults to None.
            provider: Provider name. Defaults to "openai".
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(endpoint_name, model_id, provider=provider)
        self._client = OpenAI(api_key=api_key, **kwargs)

    def _parse_payload(self, payload):
        jmes_path = "[:].content"
        messages = payload.get("messages")
        result = jmespath.search(jmes_path, messages)
        if result is None:
            return ""
        return "\n".join(result)

    @staticmethod
    def create_payload(
        user_message: str | list[ContentItem],
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> dict:
        """Create a payload for the OpenAI API request.

        Args:
            user_message: A single text string, or an ordered list mixing strings
                and :class:`~llmeter.prompt_utils.MediaContent` objects.
            max_tokens: Maximum tokens in response. Defaults to 256.
            **kwargs: Additional payload parameters.

        Returns:
            dict: Formatted payload for API request.

        Examples:
            Text only::

                create_payload("Hello")

            Image with text::

                create_payload([
                    ImageContent.from_path("photo.jpg"),
                    "What's in this image?",
                ])
        """
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        if isinstance(user_message, str):
            items: list[ContentItem] = [user_message]
        elif isinstance(user_message, list):
            items = user_message
        else:
            raise TypeError(
                "user_message must be a str or list of str/MediaContent, "
                f"got {type(user_message).__name__}"
            )

        if not items:
            raise ValueError("user_message must not be empty")

        # Text-only shortcut: single string → simple content field
        if len(items) == 1 and isinstance(items[0], str):
            payload = {
                "messages": [{"role": "user", "content": items[0]}],
                "max_tokens": max_tokens,
            }
            payload.update(kwargs)
            return payload

        content_blocks = _build_content_blocks_openai(items)
        payload = {
            "messages": [{"role": "user", "content": content_blocks}],
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)
        return payload


class OpenAICompletionEndpoint(OpenAIEndpoint):
    """Endpoint for OpenAI-compatible Chat Completion APIs (non-streaming mode)"""

    def invoke(self, payload: Dict, **kwargs: Any) -> InvocationResponse:
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
