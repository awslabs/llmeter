# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMeter endpoints for the Anthropic Messages API.

Supports any client provided by the `anthropic` Python SDK:

* `"anthropic"` - Direct API at `api.anthropic.com`
* `"bedrock-mantle"` - Amazon Bedrock Mantle (requires `anthropic[bedrock]`)
* `"vertex"` - Google Vertex AI (requires `anthropic[vertex]`)
* `"foundry"` - Azure Foundry

Install the base dependency::

    pip install 'llmeter[anthropic]'

For Bedrock Mantle support::

    pip install 'llmeter[anthropic-bedrock]'

### Extended thinking

Claude models can perform internal reasoning ("thinking") before producing a visible answer. The
configuration for this is controlled via the `thinking` parameter on the request payload (also
available in the
[`create_payload`][llmeter.endpoints.anthropic_messages.AnthropicMessagesEndpoint.create_payload]
utility function).

It's important to understand how these extra thinking/reasoning tokens that *aren't* part of the
"final" output will be treated for response timing and token counting.

#### Token accounting

The Anthropic API reports a single `output_tokens` count that **includes** both thinking and
visible text tokens.  There is no separate `reasoning_tokens` field. As a result:

* [`InvocationResponse.num_tokens_output`][llmeter.endpoints.base.InvocationResponse] reflects the
  total billed output tokens (thinking and output).
* [`InvocationResponse.num_tokens_output_reasoning`][llmeter.endpoints.base.InvocationResponse] is
  always ``None`` for Anthropic endpoints because the API does not provide this breakdown.

This differs from OpenAI, which reports `reasoning_tokens` separately.  When comparing across
providers, keep in mind that LLMeter's `num_tokens_output` is semantically consistent (total billed
output) but the reasoning breakdown is only available where the provider exposes it.

#### Time to first token (TTFT) and the ``display`` setting

The `display` field on the thinking configuration controls whether thinking content is streamed
back to the client:

* `"summarized"` (default on most models) - `thinking_delta` events stream before the visible text.
* `"omitted"` (default on Claude Opus 4.7 and Mythos) - no `thinking_delta` events are emitted;
  only a `signature_delta` signals that the thinking block completed.

The
[`AnthropicMessagesStream.ttft_visible_tokens_only`][llmeter.endpoints.anthropic_messages.AnthropicMessagesStream]
parameter controls how
[`InvocationResponse.time_to_first_token`][llmeter.endpoints.base.InvocationResponse] is measured:

* `True` (default) - TTFT records the first **visible** `text_delta`. Thinking events
  (`thinking_delta`, `signature_delta`) are ignored.  This measures the latency the end user
  experiences before seeing output.
* `False` -- TTFT records the first token of **any** kind, including `thinking_delta` (summarized
  mode) or `signature_delta` (omitted mode). This measures when the model first started producing
  output.

Because `display: "omitted"` suppresses `thinking_delta` events entirely, the `signature_delta` is
the earliest signal available.  With `ttft_visible_tokens_only=False`, the measured TTFT will
therefore differ between summarized and omitted modes for the same model and prompt: summarized
mode captures the first thinking token, while omitted mode captures the signature that arrives
after all thinking is complete.
"""

import logging
import time
from typing import Any, Generic, Iterable, TypeVar

import anthropic
from anthropic.types import (
    Message,
    MessageCreateParams,
    RawMessageStreamEvent,
)

from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)

TAnthropicResponseBase = TypeVar(
    "TAnthropicResponseBase",
    bound=Message | Iterable[RawMessageStreamEvent],
)

_ANTHROPIC_CLIENTS: dict[str, type] = {
    "anthropic": anthropic.Anthropic,
    "bedrock-mantle": anthropic.AnthropicBedrockMantle,
    "vertex": anthropic.AnthropicVertex,
    "foundry": anthropic.AnthropicFoundry,
}


class AnthropicMessagesEndpoint(
    Endpoint[TAnthropicResponseBase], Generic[TAnthropicResponseBase]
):
    """Base class for Anthropic Messages API endpoints.

    Works with any client provided by the ``anthropic`` SDK.  The ``provider``
    argument selects which client to instantiate.

    Args:
        model_id: Model identifier (e.g. ``"claude-opus-4-7"`` for direct API,
            ``"anthropic.claude-opus-4-7"`` for Bedrock Mantle).
        endpoint_name: Display name for this endpoint.  Defaults to
            ``"anthropic-messages"``.
        provider: Backend to use -- one of ``"anthropic"``,
            ``"bedrock-mantle"``, ``"vertex"``, or ``"foundry"``.
            Defaults to ``"anthropic"``.
        api_key: API key for the direct Anthropic API.
        aws_region: AWS region for Bedrock Mantle.
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
        client_cls = _ANTHROPIC_CLIENTS.get(provider)
        if client_cls is None:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Use one of: {', '.join(_ANTHROPIC_CLIENTS)}."
            )
        if api_key is not None:
            kwargs["api_key"] = api_key
        if aws_region is not None:
            kwargs["aws_region"] = aws_region
        self._client = client_cls(**kwargs)

    def _parse_payload(self, payload: MessageCreateParams | dict) -> str:
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
        thinking: dict | None = None,
        **kwargs: Any,
    ) -> MessageCreateParams:
        """Create a payload for the Anthropic Messages API.

        This is a convenience helper.  You can also build the payload dict directly following the
        [Anthropic Messages API reference](https://docs.anthropic.com/en/api/messages)

        Args:
            user_message: The user message text.
            max_tokens: Maximum tokens to generate.  Defaults to 256.
            thinking: Extended thinking configuration.  Common values:

                * ``{"type": "adaptive"}`` -- adaptive thinking
                  (recommended for Claude Opus 4.6 / Sonnet 4.6 and later).
                * ``{"type": "enabled", "budget_tokens": 10000}`` -- manual
                  thinking budget (deprecated on Claude 4.6+, unsupported on
                  Opus 4.7+).
                * ``{"type": "disabled"}`` -- explicitly disable thinking.
                * ``None`` (default) -- omit the parameter, letting the API
                  use its default behavior.

                The ``display`` key controls how thinking content is returned
                in streaming responses:

                * ``"summarized"`` (default on most models) -- thinking
                  blocks contain summarized text; ``thinking_delta`` events
                  stream before the visible text.
                * ``"omitted"`` (default on Opus 4.7 / Mythos) -- thinking
                  blocks have an empty ``thinking`` field; no
                  ``thinking_delta`` events are emitted, only a
                  ``signature_delta``.  This reduces time-to-first-text-token
                  when streaming.

                Example with display:

                ```python
                create_payload(
                    "Solve this problem",
                    max_tokens=16000,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 10000,
                        "display": "omitted",
                    },
                )
                ```

            **kwargs: Additional payload parameters (``system``,
                ``temperature``, ``top_p``, ``top_k``, ``stop_sequences``,
                etc.).

        Returns:
            dict: Formatted payload for the Anthropic Messages API.

        Raises:
            ValueError: If ``max_tokens`` is not a positive integer.
            TypeError: If ``user_message`` is not a string.

        Examples:
            Text only:

            ```python
            create_payload("Hello, Claude!")
            ```

            With system prompt:

            ```python
            create_payload(
                "Explain quantum computing",
                system="You are a physics professor.",
                max_tokens=1024,
            )
            ```

            With adaptive thinking:

            ```python
            create_payload(
                "Prove that there are infinitely many primes.",
                max_tokens=16000,
                thinking={"type": "adaptive"},
            )
            ```

            With thinking explicitly disabled:

            ```python
            create_payload(
                "Hello!",
                thinking={"type": "disabled"},
            )
            ```
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
        if thinking is not None:
            payload["thinking"] = thinking
        payload.update(kwargs)
        return payload  # type: ignore[return-value]


class AnthropicMessages(AnthropicMessagesEndpoint[Message]):
    """Endpoint for the Anthropic Messages API (non-streaming).

    When extended thinking is enabled, the response may contain `thinking` content blocks
    alongside `text` blocks.  Only `text` blocks contribute to
    [`InvocationResponse.response_text`][llmeter.endpoints.base.InvocationResponse].
    The reported `num_tokens_output` is the total billed count (thinking + text);
    `num_tokens_output_reasoning` is `None` because the Anthropic API does not provide a separate
    thinking token count.

    Examples:
        Direct Anthropic API:

        ```python
        endpoint = AnthropicMessages(model_id="claude-opus-4-7")
        ```

        Amazon Bedrock Mantle:

        ```python
        endpoint = AnthropicMessages(
            model_id="anthropic.claude-opus-4-7",
            provider="bedrock-mantle",
            aws_region="us-east-1",
        )
        ```
    """

    @AnthropicMessagesEndpoint.llmeter_invoke
    def invoke(self, payload: MessageCreateParams) -> Message:
        """Invoke the Anthropic Messages API (non-streaming)."""
        client_response = self._client.messages.create(**payload)
        return client_response

    def prepare_payload(self, payload: dict, **kwargs: Any) -> dict:
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id
        return payload

    def process_raw_response(
        self,
        raw_response: Message,
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        """Parse a non-streaming Anthropic Messages API response.

        Only `text` content blocks are extracted into `response_text`. `thinking` and
        `redacted_thinking` blocks are skipped.

        Args:
            raw_response: The `Message` object returned by the API.
            start_t: Start time of the API call.
            response: The LLMeter response object to be populated in-place.
        """
        response.time_to_last_token = time.perf_counter() - start_t
        response.id = raw_response.id

        # Extract text from content blocks (skip thinking/redacted_thinking)
        response_text = ""
        for block in raw_response.content:
            if block.type == "text":
                response_text += block.text
        response.response_text = response_text

        usage = raw_response.usage
        if usage:
            response.num_tokens_input = getattr(usage, "input_tokens", None)
            response.num_tokens_output = getattr(usage, "output_tokens", None)
            response.num_tokens_input_cached = getattr(
                usage, "cache_read_input_tokens", None
            )


class AnthropicMessagesStream(
    AnthropicMessagesEndpoint[Iterable[RawMessageStreamEvent]]
):
    """Endpoint for the Anthropic Messages API (streaming).

    Uses `client.messages.create(..., stream=True)` to stream SSE events, enabling
    time-to-first-token and time-to-last-token measurements.

    #### Extended thinking and TTFT

    When extended thinking is enabled, the stream contains thinking-related events before the
    visible text.  The `ttft_visible_tokens_only` parameter controls which event sets
    `time_to_first_token`:

    * `True` (default) - TTFT is set on the first `text_delta`. Thinking events are ignored. Use
      this to measure the latency an end user experiences before seeing output.
    * `False` - TTFT is set on the first event of any kind, including `thinking_delta` (when
      `display` is `"summarized"`) or`signature_delta` (when `display` is `"omitted"`).  Use this
      to measure when the model first started producing output.

    The `display` setting on the thinking configuration affects which events are emitted:

    * `"summarized"` - `thinking_delta` events stream before the text. With
      `ttft_visible_tokens_only=False`, TTFT captures the first thinking token.
    * `"omitted"` - no `thinking_delta` events; only a `signature_delta` signals the end of the
      thinking block.  With `ttft_visible_tokens_only=False`, TTFT captures the signature, which
      arrives later than a thinking delta would.

    This means that for the same model and prompt, measured TTFT with
    `ttft_visible_tokens_only=False` will differ between summarized and omitted modes.  Summarized
    mode captures the first thinking token; omitted mode captures the signature that arrives after
    all thinking is complete.

    Args:
        model_id: Model identifier.
        endpoint_name: Display name.  Defaults to `"anthropic-messages"`.
        provider: Backend to use.  Defaults to `"anthropic"`.
        api_key: API key for the direct Anthropic API.
        aws_region: AWS region for Bedrock Mantle.
        ttft_visible_tokens_only: When `True` (default), TTFT measures time to first visible text
            token.  When `False`, TTFT includes thinking/signature events.  See above for details.
        **kwargs: Additional arguments forwarded to the client constructor.

    Examples:
        Direct Anthropic API:

        ```python
        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
        ```

        Measure TTFT including thinking:

        ```python
        endpoint = AnthropicMessagesStream(
            model_id="claude-sonnet-4-6",
            ttft_visible_tokens_only=False,
        )
        ```

        Amazon Bedrock Mantle:

        ```python
        endpoint = AnthropicMessagesStream(
            model_id="anthropic.claude-opus-4-7",
            provider="bedrock-mantle",
            aws_region="us-east-1",
        )
        ```
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "anthropic-messages",
        provider: str = "anthropic",
        api_key: str | None = None,
        aws_region: str | None = None,
        ttft_visible_tokens_only: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            model_id=model_id,
            endpoint_name=endpoint_name,
            provider=provider,
            api_key=api_key,
            aws_region=aws_region,
            **kwargs,
        )
        self.ttft_visible_tokens_only = ttft_visible_tokens_only

    @AnthropicMessagesEndpoint.llmeter_invoke
    def invoke(self, payload: MessageCreateParams) -> Iterable[RawMessageStreamEvent]:
        """Invoke the Anthropic Messages API with streaming."""
        client_response = self._client.messages.create(**payload)
        return client_response

    def prepare_payload(self, payload: dict, **kwargs: Any) -> dict:
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id
        payload["stream"] = True
        return payload

    def process_raw_response(
        self,
        raw_response: Iterable[RawMessageStreamEvent],
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        """Parse a streaming Anthropic Messages API response.

        Processes SSE events to extract text, token counts, and timing.

        Only `text_delta` events contribute to `response_text`. `thinking_delta` and
        `signature_delta` events are used solely for TTFT measurement when
        `ttft_visible_tokens_only` is `False`.

        Args:
            raw_response: The streaming iterator of SSE events.
            start_t: Start time of the API call.
            response: The LLMeter response object to be populated in-place.
        """
        _THINKING_DELTA_TYPES = frozenset(("thinking_delta", "signature_delta"))

        for event in raw_response:
            now = time.perf_counter()
            event_type = event.type

            if event_type == "message_start":
                response.id = event.message.id
                if event.message.usage:
                    response.num_tokens_input = getattr(
                        event.message.usage, "input_tokens", None
                    )
                    response.num_tokens_input_cached = getattr(
                        event.message.usage, "cache_read_input_tokens", None
                    )

            elif event_type == "content_block_delta":
                delta = event.delta
                delta_type = getattr(delta, "type", None)

                if delta_type in _THINKING_DELTA_TYPES:
                    if (
                        not self.ttft_visible_tokens_only
                        and response.time_to_first_token is None
                    ):
                        response.time_to_first_token = now - start_t

                elif delta_type == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        if response.time_to_first_token is None:
                            response.time_to_first_token = now - start_t
                        if response.response_text is None:
                            response.response_text = text
                        else:
                            response.response_text += text
                        response.time_to_last_token = now - start_t

            elif event_type == "message_delta":
                if event.usage:
                    response.num_tokens_output = getattr(
                        event.usage, "output_tokens", None
                    )
