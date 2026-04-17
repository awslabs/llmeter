# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMeter targets for testing Anthropic Messages API endpoints

Supports the Anthropic Messages API via:

* **Direct Anthropic API** — using an API key against ``api.anthropic.com``
* **Amazon Bedrock** — using ``AnthropicBedrock`` with AWS credentials and
  ARN-versioned model IDs (``InvokeModel``-based integration)
* **Amazon Bedrock Mantle** — using ``AnthropicBedrockMantle`` with the
  ``/anthropic/v1/messages`` endpoint and SSE streaming

All three paths use the same ``anthropic`` Python SDK and the same Messages API
shape, so a single pair of endpoint classes covers them.

Requires the ``anthropic`` optional dependency::

    pip install 'llmeter[anthropic]'

For Bedrock support, install the bedrock extra instead::

    pip install 'llmeter[anthropic-bedrock]'
"""

import logging
import time
from typing import Any

from ..utils import DeferredError
from .base import Endpoint, InvocationResponse, llmeter_invoke

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError as e:
    logger.debug(
        "anthropic not available. Install with: pip install 'llmeter[anthropic]' "
        "or pip install 'llmeter[anthropic-bedrock]' for Bedrock support"
    )
    anthropic = DeferredError(e)


def _build_anthropic_client(
    provider: str,
    api_key: str | None = None,
    aws_region: str | None = None,
    **kwargs: Any,
):
    """Build the appropriate Anthropic client based on provider.

    Args:
        provider: One of ``"anthropic"``, ``"bedrock"``, or ``"bedrock-mantle"``.
        api_key: API key for direct Anthropic API access.
        aws_region: AWS region for Bedrock providers.
        **kwargs: Additional keyword arguments passed to the client constructor.

    Returns:
        An Anthropic client instance.

    Raises:
        ValueError: If provider is not recognized.
    """
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=api_key, **kwargs)
    elif provider == "bedrock":
        return anthropic.AnthropicBedrock(aws_region=aws_region, **kwargs)
    elif provider == "bedrock-mantle":
        return anthropic.AnthropicBedrockMantle(aws_region=aws_region, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Use 'anthropic', 'bedrock', or 'bedrock-mantle'."
        )


class AnthropicMessagesEndpoint(Endpoint):
    """Base class for Anthropic Messages API endpoints.

    Works with the direct Anthropic API, Amazon Bedrock, and Amazon Bedrock
    Mantle.  The ``provider`` argument selects which client to instantiate.

    Args:
        model_id: Model identifier (e.g. ``"claude-opus-4-7"``,
            ``"anthropic.claude-opus-4-7"`` for Bedrock Mantle,
            ``"global.anthropic.claude-opus-4-6-v1"`` for Bedrock).
        endpoint_name: Display name for this endpoint.  Defaults to
            ``"anthropic-messages"``.
        provider: Backend to use.  One of ``"anthropic"`` (direct API),
            ``"bedrock"`` (InvokeModel-based), or ``"bedrock-mantle"``
            (Messages API via Mantle).  Defaults to ``"anthropic"``.
        api_key: API key for the direct Anthropic API.  Ignored for Bedrock
            providers.
        aws_region: AWS region for Bedrock providers.  Ignored for direct API.
        **kwargs: Additional keyword arguments forwarded to the underlying
            ``anthropic`` client constructor (e.g. ``base_url``,
            ``max_retries``, ``timeout``).
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "anthropic-messages",
        provider: str = "anthropic",
        api_key: str | None = None,
        aws_region: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            endpoint_name=endpoint_name,
            model_id=model_id,
            provider=provider,
        )
        self.aws_region = aws_region
        self._client = _build_anthropic_client(
            provider=provider,
            api_key=api_key,
            aws_region=aws_region,
            **kwargs,
        )

    def _parse_payload(self, payload: dict) -> str:
        """Extract user message text from an Anthropic Messages API payload.

        Args:
            payload: Request payload containing ``messages``.

        Returns:
            Concatenated message text content.
        """
        messages = payload.get("messages")
        if not messages:
            return ""
        contents: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if not content:
                continue
            if isinstance(content, str):
                contents.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        contents.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        contents.append(block.get("text", ""))
        return "\n".join(contents)

    @staticmethod
    def create_payload(
        user_message: str,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> dict:
        """Create a payload for the Anthropic Messages API.

        This is a convenience helper.  You can also build the payload dict
        directly following the `Anthropic Messages API reference
        <https://docs.anthropic.com/en/api/messages>`_.

        Args:
            user_message: The user message text.
            max_tokens: Maximum tokens to generate.  Defaults to 256.
            **kwargs: Additional payload parameters (``system``,
                ``temperature``, ``top_p``, ``top_k``, ``stop_sequences``,
                etc.).

        Returns:
            dict: Formatted payload for the Anthropic Messages API.

        Raises:
            ValueError: If ``max_tokens`` is not a positive integer.
            TypeError: If ``user_message`` is not a string.

        Examples:
            Text only::

                create_payload("Hello, Claude!")

            With system prompt::

                create_payload(
                    "Explain quantum computing",
                    system="You are a physics professor.",
                    max_tokens=1024,
                )
        """
        if not isinstance(user_message, str):
            raise TypeError(
                f"user_message must be a str, got {type(user_message).__name__}"
            )
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        payload: dict = {
            "messages": [{"role": "user", "content": user_message}],
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)
        return payload


class AnthropicMessages(AnthropicMessagesEndpoint):
    """Endpoint for the Anthropic Messages API (non-streaming).

    Examples:
        Direct Anthropic API::

            endpoint = AnthropicMessages(model_id="claude-opus-4-7")

        Amazon Bedrock (InvokeModel-based)::

            endpoint = AnthropicMessages(
                model_id="global.anthropic.claude-opus-4-6-v1",
                provider="bedrock",
                aws_region="us-west-2",
            )

        Amazon Bedrock Mantle::

            endpoint = AnthropicMessages(
                model_id="anthropic.claude-opus-4-7",
                provider="bedrock-mantle",
                aws_region="us-east-1",
            )
    """

    @llmeter_invoke
    def invoke(self, payload: dict) -> InvocationResponse:
        """Invoke the Anthropic Messages API (non-streaming)."""
        client_response = self._client.messages.create(**payload)
        return self.parse_response(client_response, self._start_t)

    def prepare_payload(self, payload: dict, **kwargs: Any) -> dict:
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id
        return payload

    def parse_response(self, client_response, start_t: float) -> InvocationResponse:
        """Parse a non-streaming Anthropic Messages API response.

        Args:
            client_response: The ``Message`` object returned by the API.
            start_t: Start time of the API call.

        Returns:
            InvocationResponse with extracted text and token counts.
        """
        # Extract text from content blocks
        response_text = ""
        for block in client_response.content:
            if block.type == "text":
                response_text += block.text

        usage = client_response.usage
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        cached_tokens = None
        if usage:
            cached_tokens = getattr(usage, "cache_read_input_tokens", None)

        return InvocationResponse(
            id=client_response.id,
            response_text=response_text,
            num_tokens_input=input_tokens,
            num_tokens_output=output_tokens,
            num_tokens_input_cached=cached_tokens,
        )


class AnthropicMessagesStream(AnthropicMessagesEndpoint):
    """Endpoint for the Anthropic Messages API (streaming).

    Uses ``client.messages.create(..., stream=True)`` to stream SSE events,
    enabling time-to-first-token and time-to-last-token measurements.

    Examples:
        Direct Anthropic API::

            endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")

        Amazon Bedrock Mantle::

            endpoint = AnthropicMessagesStream(
                model_id="anthropic.claude-opus-4-7",
                provider="bedrock-mantle",
                aws_region="us-east-1",
            )
    """

    @llmeter_invoke
    def invoke(self, payload: dict) -> InvocationResponse:
        """Invoke the Anthropic Messages API with streaming."""
        client_response = self._client.messages.create(**payload)
        return self.parse_response(client_response, self._start_t)

    def prepare_payload(self, payload: dict, **kwargs: Any) -> dict:
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id
        payload["stream"] = True
        return payload

    def parse_response(self, client_response, start_t: float) -> InvocationResponse:
        """Parse a streaming Anthropic Messages API response.

        Processes SSE events to extract text, token counts, and timing
        information (TTFT and TTLT).

        Args:
            client_response: The streaming iterator of SSE events.
            start_t: Start time of the API call.

        Returns:
            InvocationResponse with concatenated text, token counts, TTFT,
            and TTLT.
        """
        response_text = ""
        response_id = None
        input_tokens = None
        output_tokens = None
        cached_tokens = None
        time_to_first_token = None

        for event in client_response:
            event_type = event.type

            if event_type == "message_start":
                response_id = event.message.id
                # input token count is available in the initial message usage
                if event.message.usage:
                    input_tokens = getattr(
                        event.message.usage, "input_tokens", None
                    )
                    cached_tokens = getattr(
                        event.message.usage, "cache_read_input_tokens", None
                    )

            elif event_type == "content_block_delta":
                delta = event.delta
                if getattr(delta, "type", None) == "text_delta":
                    text = getattr(delta, "text", "")
                    if text and time_to_first_token is None:
                        time_to_first_token = time.perf_counter() - start_t
                    response_text += text

            elif event_type == "message_delta":
                # output token count comes in the message_delta usage
                if event.usage:
                    output_tokens = getattr(event.usage, "output_tokens", None)

        time_to_last_token = time.perf_counter() - start_t

        if time_to_first_token is None:
            time_to_first_token = time_to_last_token

        return InvocationResponse(
            id=response_id,
            response_text=response_text,
            num_tokens_input=input_tokens,
            num_tokens_output=output_tokens,
            num_tokens_input_cached=cached_tokens,
            time_to_first_token=time_to_first_token,
            time_to_last_token=time_to_last_token,
        )
