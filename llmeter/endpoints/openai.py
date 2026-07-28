# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMeter targets for testing OpenAI ChatCompletions-compatible endpoints (wherever they're hosted)"""

# Python Built-Ins:
import base64
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Literal, Generic, Iterable, TypeVar, cast

# External Dependencies:
import httpx  # (Indirect dependency of OpenAI)
from openai import OpenAI
from openai._constants import DEFAULT_MAX_RETRIES
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    CompletionCreateParams,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from openai.types.chat.chat_completion_content_part_param import (
    File as ChatCompletionFile,
)

# Local Dependencies:
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
_MIME_TO_OPENAI_AUDIO_FMT: dict[str, Literal["mp3", "wav"]] = {
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
}

TOpenAICompletionBase = TypeVar(
    "TOpenAICompletionBase", bound=ChatCompletion | Iterable[ChatCompletionChunk]
)


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


class OpenAIEndpoint(Endpoint[TOpenAICompletionBase], Generic[TOpenAICompletionBase]):
    """Base class for OpenAI API endpoints."""

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "openai",
        api_key: str | None = None,
        provider: str = "openai",
        organization: str | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        websocket_base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | dict | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI endpoint.

        Args:
            model_id: ID of the OpenAI model to use
            endpoint_name: Name of the endpoint. Defaults to "openai".
            api_key: OpenAI API key. Defaults to None (reads from OPENAI_API_KEY env var).
            provider: Provider name. Defaults to "openai".
            organization: OpenAI organization ID. Defaults to None.
            project: OpenAI project ID. Defaults to None.
            base_url: Override the default base URL for the API.
            websocket_base_url: Override the default base URL for websocket connections.
            timeout: Request timeout in seconds, or an httpx.Timeout for granular control. If a
                dict is passed (as is the case after serialization), it'll be interpreted as a set
                of arguments to create an httpx.Timeout
            max_retries: Maximum number of retries for failed requests.
            default_headers: Additional headers to send with every request.
            default_query: Additional query parameters to send with every request.
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(endpoint_name, model_id, provider=provider)
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "organization": organization,
            "project": project,
            "max_retries": max_retries,
            **kwargs,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if websocket_base_url is not None:
            client_kwargs["websocket_base_url"] = websocket_base_url
        if timeout is not None:
            if isinstance(timeout, dict):
                timeout = httpx.Timeout(**timeout)
            client_kwargs["timeout"] = timeout
        if default_headers is not None:
            client_kwargs["default_headers"] = default_headers
        if default_query is not None:
            client_kwargs["default_query"] = default_query
        self._client = OpenAI(**client_kwargs)

    @property
    def project(self) -> str | None:
        """The OpenAI project ID configured on the client."""
        return self._client.project

    def _get_llmeter_state(self) -> dict:
        """Extract serializable state from the endpoint and its underlying client.

        This override extends the default state (which pulls immediate properties of the Endpoint)
        to include other params set on the OpenAI client but *not* persisted directly on the
        Endpoint object. Any config we don't persist will get lost on a serialization round trip,
        but anything we pull in should be restored because __init__ passes extra **kwargs to the
        OpenAI client constructor.
        """
        state = super()._get_llmeter_state()
        state["organization"] = self._client.organization
        state["base_url"] = str(self._client.base_url)
        state["max_retries"] = self._client.max_retries
        if self._client.websocket_base_url is not None:
            state["websocket_base_url"] = str(self._client.websocket_base_url)
        timeout = self._client.timeout
        if isinstance(timeout, httpx.Timeout):
            state["timeout"] = {
                "connect": timeout.connect,
                "read": timeout.read,
                "write": timeout.write,
                "pool": timeout.pool,
            }
        elif timeout is not None:
            state["timeout"] = timeout
        if self._client._custom_headers:
            state["default_headers"] = dict(self._client._custom_headers)
        if self._client._custom_query:
            state["default_query"] = dict(self._client._custom_query)
        return state

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


class OpenAICompletionEndpoint(OpenAIEndpoint[ChatCompletion]):
    """Endpoint for OpenAI-compatible Chat Completion APIs (non-streaming mode)"""

    @OpenAIEndpoint.llmeter_invoke
    def invoke(self, payload: CompletionCreateParamsNonStreaming) -> ChatCompletion:
        """Invoke the OpenAI chat completion API."""
        client_response: ChatCompletion = self._client.chat.completions.create(
            **payload
        )
        return client_response

    def prepare_payload(self, payload):
        """Ensure payload specifies correct model ID and streaming disabled"""
        return {
            **payload,
            "model": self.model_id,
            "stream": False,
        }

    def process_raw_response(
        self, raw_response: ChatCompletion, start_t: float, response: InvocationResponse
    ) -> None:
        response.time_to_last_token = time.perf_counter() - start_t
        response.id = raw_response.id

        response.response_text = raw_response.choices[0].message.content

        usage = raw_response.usage
        if usage:
            response.num_tokens_input = usage.prompt_tokens
            response.num_tokens_output = usage.completion_tokens
            if usage.prompt_tokens_details:
                response.num_tokens_input_cached = getattr(
                    usage.prompt_tokens_details, "cached_tokens", None
                )
            if usage.completion_tokens_details:
                response.num_tokens_output_reasoning = getattr(
                    usage.completion_tokens_details, "reasoning_tokens", None
                )


class OpenAICompletionStreamEndpoint(OpenAIEndpoint[Iterable[ChatCompletionChunk]]):
    """Endpoint for OpenAI-compatible Chat Completion APIs (streaming mode)"""

    @OpenAIEndpoint.llmeter_invoke
    def invoke(self, payload: CompletionCreateParamsStreaming):
        """Invoke the OpenAI streaming chat completion API."""
        client_response = self._client.chat.completions.create(**payload)
        return client_response

    def prepare_payload(self, payload):
        """Ensure payload specifies correct model ID and streaming settings"""
        payload = {
            **payload,
            "model": self.model_id,
            "stream": True,
        }
        if not payload.get("stream_options"):
            payload["stream_options"] = {"include_usage": True}
        return payload

    def process_raw_response(
        self,
        raw_response: Iterable[ChatCompletionChunk],
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        """Parse the streaming API response from OpenAI chat completion API."""
        got_chunk_id = False
        for chunk in raw_response:
            now = time.perf_counter()

            if not got_chunk_id and chunk.id is not None:
                response.id = chunk.id
                got_chunk_id = True

            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    if response.response_text is None:
                        response.time_to_first_token = now - start_t
                        response.response_text = content
                    else:
                        response.response_text += content
                    response.time_to_last_token = now - start_t

            if chunk.usage is not None:
                response.num_tokens_input = chunk.usage.prompt_tokens
                response.num_tokens_output = chunk.usage.completion_tokens
                if chunk.usage.prompt_tokens_details:
                    response.num_tokens_input_cached = getattr(
                        chunk.usage.prompt_tokens_details, "cached_tokens", None
                    )
                if chunk.usage.completion_tokens_details:
                    response.num_tokens_output_reasoning = getattr(
                        chunk.usage.completion_tokens_details,
                        "reasoning_tokens",
                        None,
                    )
