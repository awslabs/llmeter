# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMeter targets for testing OpenAI ChatCompletions-compatible endpoints (wherever they're hosted)"""

import base64
import logging
import os
import time
from typing import Any, Literal, cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    CompletionCreateParams,
)
from openai.types.chat.chat_completion_content_part_param import (
    File as ChatCompletionFile,
)

from ..prompt_utils import (
    ContentItem,
    MediaContent,
    VideoContent,
)
from .base import Endpoint, InvocationResponse, llmeter_invoke

logger = logging.getLogger(__name__)

# MIME types supported by OpenAI, grouped by content part type
_OPENAI_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_OPENAI_AUDIO_MIMES = {"audio/mpeg", "audio/wav"}
_OPENAI_FILE_MIMES = {"application/pdf"}

# Map MIME → OpenAI audio "format" field value
_MIME_TO_OPENAI_AUDIO_FMT: dict[str, Literal["mp3", "wav"]] = {
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
}


def _make_openai_content_block(item: MediaContent) -> ChatCompletionContentPartParam:
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
        block: ChatCompletionFile = {
            "type": "file",
            "file": {"file_data": f"data:{mime};base64,{b64}"},
        }
        if item.source_path:
            block["file"]["filename"] = os.path.basename(item.source_path)
        return block

    raise ValueError(f"Unsupported MIME type for OpenAI: '{mime}'")


def _build_content_blocks_openai(
    items: list[ContentItem],
) -> list[ChatCompletionContentPartParam]:
    """Convert an ordered list of content items to OpenAI content blocks.

    Args:
        items: Ordered sequence of strings and/or ``MediaContent`` objects.

    Returns:
        list[dict]: Content blocks conforming to OpenAI SDK types.

    Raises:
        ValueError: If a MIME type is unsupported or video is provided.
        TypeError: If an item has an unexpected type.
    """
    blocks: list[ChatCompletionContentPartParam] = []
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

    def _parse_payload(self, payload: CompletionCreateParams | dict) -> str:
        """Extract the user message text from a ChatCompletions request payload

        Args:
            payload: Request payload containing ``messages``

        Returns:
            str: Concatenated message contents
        """
        messages = payload.get("messages")
        if not messages:
            return ""
        if isinstance(messages, list):
            contents: list[str] = []
            for msg in messages:
                content = msg.get("content")
                if not content:
                    continue
                if isinstance(content, str):
                    contents.append(content)
                else:
                    for part in content:
                        if "text" in part:
                            contents.append(part["text"])
                        # For now ignore file, image_url, input_audio, refusal, etc.
            return "\n".join(contents)
        return ""

    @staticmethod
    def create_payload(
        user_message: str | list[ContentItem], max_tokens: int = 256, **kwargs: Any
    ) -> CompletionCreateParams:
        """Create a payload for the OpenAI Chat Completions API request.

        This is a convenience helper. You can also build the payload directly
        using ``openai.types.chat.CompletionCreateParams`` (though the `model`
        field is optional).

        Args:
            user_message: A single text string, or an ordered list mixing strings
                and :class:`~llmeter.prompt_utils.MediaContent` objects.
            max_tokens: Maximum tokens in response. Defaults to 256.
            **kwargs: Additional payload parameters.

        Returns:
            dict: Formatted OpenAI CompletionCreateParams input payload

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
        else:
            content_blocks = _build_content_blocks_openai(items)
            payload = {
                "messages": [{"role": "user", "content": content_blocks}],
                "max_tokens": max_tokens,
            }
        payload.update(kwargs)
        return cast(CompletionCreateParams, payload)


class OpenAICompletionEndpoint(OpenAIEndpoint):
    """Endpoint for OpenAI-compatible Chat Completion APIs (non-streaming mode)"""

    @llmeter_invoke
    def invoke(self, payload: CompletionCreateParams) -> InvocationResponse:
        """Invoke the OpenAI chat completion API."""
        client_response: ChatCompletion = self._client.chat.completions.create(
            **payload
        )
        return self.parse_response(client_response, self._start_t)

    def prepare_payload(self, payload, **kwargs):
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id
        return payload

    def parse_response(self, client_response: ChatCompletion, start_t: float):
        usage = client_response.usage
        cached_tokens = None
        if usage and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", None)
        return InvocationResponse(
            id=client_response.id,
            response_text=client_response.choices[0].message.content,
            num_tokens_input=usage and usage.prompt_tokens,
            num_tokens_output=usage and usage.completion_tokens,
            num_tokens_input_cached=cached_tokens,
        )


class OpenAICompletionStreamEndpoint(OpenAIEndpoint):
    """Endpoint for OpenAI-compatible Chat Completion APIs (streaming mode)"""

    @llmeter_invoke
    def invoke(self, payload: CompletionCreateParams) -> InvocationResponse:
        """Invoke the OpenAI streaming chat completion API."""
        client_response = self._client.chat.completions.create(**payload)
        return self.parse_response(client_response, self._start_t)

    def prepare_payload(self, payload, **kwargs):
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id
        if not payload.get("stream"):
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    def parse_response(self, client_response, start_t: float) -> InvocationResponse:
        """Parse the streaming API response from OpenAI chat completion API.

        Args:
            client_response: Stream of ``ChatCompletionChunk`` objects
            start_t: Start time of the API call in seconds

        Returns:
            InvocationResponse with concatenated text, token counts, TTFT and TTLT
        """
        prompt_tokens = None
        completion_tokens = None
        cached_tokens = None
        response_text = ""
        response_id = None
        time_to_first_token = None

        for chunk in client_response:
            chunk: ChatCompletionChunk
            if response_id is None:
                response_id = chunk.id

            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    if time_to_first_token is None:
                        time_to_first_token = time.perf_counter() - start_t
                    response_text += content

            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
                if chunk.usage.prompt_tokens_details:
                    cached_tokens = getattr(
                        chunk.usage.prompt_tokens_details, "cached_tokens", None
                    )

        time_to_last_token = time.perf_counter() - start_t

        if time_to_first_token is None:
            time_to_first_token = time_to_last_token

        return InvocationResponse(
            id=response_id,
            response_text=response_text,
            num_tokens_input=prompt_tokens,
            num_tokens_output=completion_tokens,
            num_tokens_input_cached=cached_tokens,
            time_to_first_token=time_to_first_token,
            time_to_last_token=time_to_last_token,
        )
