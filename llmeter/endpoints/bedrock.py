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
from typing import Any, Generic, TypeVar

import boto3
from botocore.config import Config

try:
    from mypy_boto3_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamResponseTypeDef,
    )
except ImportError:
    ConverseResponseTypeDef = TypeVar("ConverseResponseTypeDef")
    ConverseStreamResponseTypeDef = TypeVar("ConverseStreamResponseTypeDef")

from ..prompt_utils import (
    AudioContent,
    ContentItem,
    DocumentContent,
    ImageContent,
    MediaContent,
    VideoContent,
)
from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)

# Error event types that can appear in Bedrock streaming responses.
# Shared by both ConverseStream and InvokeModelWithResponseStream APIs.
# See: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ResponseStream.html
BEDROCK_STREAM_ERROR_TYPES = frozenset(
    {
        "internalServerException",
        "modelStreamErrorException",
        "modelTimeoutException",
        "serviceUnavailableException",
        "throttlingException",
        "validationException",
    }
)


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


# Mapping from MediaContent subclass to Bedrock content-block key
_MEDIA_TYPE_KEY: dict[type, str] = {
    ImageContent: "image",
    AudioContent: "audio",
    VideoContent: "video",
    DocumentContent: "document",
}


def _build_content_blocks(items: list[ContentItem]) -> list[dict]:
    """Convert an ordered list of content items to Bedrock Converse content blocks.

    Args:
        items: Ordered sequence of strings and/or ``MediaContent`` objects.

    Returns:
        list[dict]: Content blocks in Bedrock Converse API format.

    Raises:
        ValueError: If a MIME type is not supported by Bedrock.
        TypeError: If an item has an unexpected type.
    """
    blocks: list[dict] = []
    for item in items:
        if isinstance(item, str):
            blocks.append({"text": item})
        elif isinstance(item, MediaContent):
            key = _MEDIA_TYPE_KEY.get(type(item))
            if key is None:
                raise TypeError(f"Unsupported content type: {type(item).__name__}")
            fmt = _mime_to_format(item.mime_type)
            if fmt is None:
                raise ValueError(f"Unsupported MIME type '{item.mime_type}' for {key}")
            blocks.append({key: {"format": fmt, "source": {"bytes": item.data}}})
        else:
            raise TypeError(
                f"Content items must be str or MediaContent, got {type(item).__name__}"
            )
    return blocks


TBedrockConverseResponseBase = TypeVar(
    "TBedrockConverseResponseBase",
    bound=ConverseResponseTypeDef | ConverseStreamResponseTypeDef,
)


class BedrockBase(
    Endpoint[TBedrockConverseResponseBase], Generic[TBedrockConverseResponseBase]
):
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
        user_message: str | list[ContentItem],
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Create a payload for the Bedrock Converse API request with optional multi-modal content.

        ⚠️ SECURITY WARNING: Format detection is for testing/development convenience ONLY.
        This method does NOT validate file safety, integrity, or protect against malicious
        content. DO NOT use with untrusted files (user uploads, external sources) without
        proper validation, sanitization, and security measures.

        Args:
            user_message: A single text string, or an ordered list mixing strings
                and :class:`~llmeter.prompt_utils.MediaContent` objects
                (``ImageContent``, ``AudioContent``, ``VideoContent``,
                ``DocumentContent``).  The order of items in the list controls
                the order of content blocks in the API request.
            max_tokens: Maximum number of tokens to generate. Defaults to 256.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The formatted payload for the Bedrock API request.

        Raises:
            TypeError: If parameters have invalid types.
            ValueError: If parameters have invalid values.

        Examples:
            Text only::

                create_payload("Hello")

            Image with text (order preserved)::

                create_payload([
                    ImageContent.from_path("photo.jpg"),
                    "What's in this image?",
                ])

            Mixed content::

                create_payload([
                    "Compare the chart with the report:",
                    ImageContent.from_path("chart.png"),
                    DocumentContent.from_path("report.pdf"),
                ])
        """
        if max_tokens is None:
            max_tokens = 256
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        # Normalise to a list of ContentItem
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

        content_blocks = _build_content_blocks(items)

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

    def prepare_payload(self, payload):
        """Enforce required properties on the input to Bedrock Converse* APIs

        In particular, ensure `modelId` is set in line with this Endpoint's configured model ID
        and apply `self._inference_config` if no config was explicitly set on the payload.
        """
        payload = {**payload}
        if payload.get("inferenceConfig") is None:
            payload["inferenceConfig"] = self._inference_config or {}
        payload["modelId"] = self.model_id
        return payload


class BedrockConverse(BedrockBase[ConverseResponseTypeDef]):
    @BedrockBase.llmeter_invoke
    def invoke(self, payload: dict) -> ConverseResponseTypeDef:
        """Invoke the Bedrock converse API with the given payload."""
        client_response = self._bedrock_client.converse(**payload)  # type: ignore
        return client_response

    def process_raw_response(
        self, raw_response, start_t: float, response: InvocationResponse
    ) -> None:
        """Parse the response from a Bedrock converse API call.

        Args:
            response: Raw response from the Bedrock API.
            start_t: The timestamp when the request was initiated.

        Returns:
            InvocationResponse with the generated text and metadata.
        """
        resp_meta = raw_response.get("ResponseMetadata", {})
        response.id = resp_meta.get("RequestId")
        response.retries = resp_meta.get("RetryAttempts")

        text_parts = [
            part["text"]
            for part in raw_response["output"]["message"]["content"]
            if "text" in part
        ]
        response.response_text = "".join(text_parts)

        usage = raw_response.get("usage", {})
        response.num_tokens_input = usage.get("inputTokens")
        response.num_tokens_input_cached = usage.get("cacheReadInputTokens")
        response.num_tokens_output = usage.get("outputTokens")


class BedrockConverseStream(BedrockBase[ConverseStreamResponseTypeDef]):
    """Streaming endpoint for the Bedrock Converse API.

    When extended thinking is enabled, the stream contains `reasoningContent` deltas before the
    visible `text` deltas.  The `ttft_visible_tokens_only` parameter controls which delta sets
    `time_to_first_token`:

    * `True` (default) - TTFT is set on the first `text` delta. Reasoning deltas are ignored for
      timing.
    * `False` - TTFT is set on the first delta of any kind, including reasoning content.

    Args:
        model_id: Bedrock model identifier.
        endpoint_name: Display name.  Defaults to `None`.
        region: AWS region.  Defaults to `None`.
        inference_config: Default inference configuration.
        bedrock_boto3_client: Pre-configured boto3 client.
        max_attempts: Maximum retry attempts.  Defaults to 3.
        ttft_visible_tokens_only: When `True` (default), TTFT measures time to first visible text
            token.  When `False`, TTFT includes reasoning content deltas.
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str | None = None,
        region: str | None = None,
        inference_config: dict | None = None,
        bedrock_boto3_client=None,
        max_attempts: int = 3,
        ttft_visible_tokens_only: bool = True,
    ):
        super().__init__(
            model_id=model_id,
            endpoint_name=endpoint_name,
            region=region,
            inference_config=inference_config,
            bedrock_boto3_client=bedrock_boto3_client,
            max_attempts=max_attempts,
        )
        self.ttft_visible_tokens_only = ttft_visible_tokens_only

    @Endpoint.llmeter_invoke
    def invoke(self, payload: dict):
        client_response = self._bedrock_client.converse_stream(**payload)  # type: ignore
        return client_response

    def process_raw_response(
        self, raw_response, start_t: float, response: InvocationResponse
    ) -> None:
        """Parse the streaming response from a Bedrock ConverseStream API call.

        Only `text` deltas contribute to `response_text`. `reasoningContent` deltas are used solely
        for TTFT measurement when `ttft_visible_tokens_only` is `False`.

        Args:
            raw_response: The raw response from the Bedrock API.
            start_t: The timestamp when the request was initiated.
            response: The output response object to be updated in-place.
        """
        response.id = raw_response["ResponseMetadata"].get("RequestId")
        response.retries = raw_response["ResponseMetadata"]["RetryAttempts"]

        for chunk in raw_response["stream"]:
            now = time.perf_counter()

            if "contentBlockDelta" in chunk:
                delta = chunk["contentBlockDelta"]["delta"]

                if "reasoningContent" in delta:
                    # Reasoning delta -- only counts for TTFT when
                    # ttft_visible_tokens_only is False.
                    if (
                        not self.ttft_visible_tokens_only
                        and response.time_to_first_token is None
                    ):
                        response.time_to_first_token = now - start_t

                elif "text" in delta:
                    delta_text = delta["text"]
                    if not isinstance(delta_text, str):
                        raise TypeError("Expected string for delta text")
                    if delta_text:
                        if response.time_to_first_token is None:
                            response.time_to_first_token = now - start_t
                        if response.response_text is None:
                            response.response_text = delta_text
                        else:
                            response.response_text += delta_text
                        response.time_to_last_token = now - start_t

            if "contentBlockStop" in chunk:
                response.time_to_last_token = now - start_t

            if "metadata" in chunk:
                usage = chunk["metadata"].get("usage", {})
                input_tokens = usage.get("inputTokens")
                if input_tokens is not None:
                    if response.num_tokens_input is None:
                        response.num_tokens_input = input_tokens
                    else:
                        response.num_tokens_input += input_tokens
                output_tokens = usage.get("outputTokens")
                if output_tokens is not None:
                    if response.num_tokens_output is None:
                        response.num_tokens_output = output_tokens
                    else:
                        response.num_tokens_output += output_tokens
                cache_read_input_tokens = usage.get("cacheReadInputTokens")
                if cache_read_input_tokens is not None:
                    if response.num_tokens_input_cached is None:
                        response.num_tokens_input_cached = cache_read_input_tokens
                    else:
                        response.num_tokens_input_cached += cache_read_input_tokens

            # Detect Bedrock stream error events
            for error_type in BEDROCK_STREAM_ERROR_TYPES:
                if error_type in chunk:
                    response.time_to_last_token = now - start_t
                    raise RuntimeError(
                        f"Bedrock {error_type}: {chunk[error_type]['message']}"
                    )
