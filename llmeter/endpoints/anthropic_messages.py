# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLMeter endpoints for the Anthropic Messages API.

Supports any client provided by the `anthropic` Python SDK:

* `"anthropic"` - Direct API at `api.anthropic.com`
* `"bedrock-mantle"` - Amazon Bedrock Mantle
* `"vertex"` - Google Vertex AI (requires `anthropic[vertex]`)
* `"foundry"` - Azure Foundry

Install the dependency::

    pip install 'llmeter[anthropic]'

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
visible text tokens, and separately reports a breakdown via
`usage.output_tokens_details.thinking_tokens`. As a result:

* [`InvocationResponse.num_tokens_output`][llmeter.endpoints.base.InvocationResponse] reflects the
  total billed output tokens (thinking and output).
* [`InvocationResponse.num_tokens_output_reasoning`][llmeter.endpoints.base.InvocationResponse] is
  populated from `thinking_tokens`, so `num_tokens_output - num_tokens_output_reasoning`
  approximates the visible output. It is `None` if the API response omits the breakdown (older API
  versions, or a gateway that strips it), and `0` for a request that did no thinking.

!!! note
    Anthropic computes `thinking_tokens` by re-tokenizing the raw reasoning text, so it can differ
    from the model's exact generation count by a small number of tokens. It also reflects the
    *raw* reasoning rather than the possibly-shorter summarized thinking returned in the response
    body. Treat derived visible-token counts as close approximations rather than exact figures.

This is more granular than it used to be: earlier LLMeter versions always reported `None` here for
Anthropic endpoints, because the breakdown was not available.

#### Time to first token (TTFT) and the ``display`` setting

The `display` field on the thinking configuration controls whether thinking content is streamed
back to the client:

* `"summarized"` (default on most models) - `thinking_delta` events stream before the visible text.
* `"omitted"` (default on Claude Opus 4.7 and Mythos) - no `thinking_delta` events are emitted;
  only a `signature_delta` signals that the thinking block completed.

LLMeter records two first-token metrics for streaming responses:

* [`InvocationResponse.time_to_first_token`][llmeter.endpoints.base.InvocationResponse] - the first
  output token of any kind, i.e. the first `thinking_delta` when the model thinks.
* [`InvocationResponse.time_to_first_content_token`][llmeter.endpoints.base.InvocationResponse] -
  the first visible `text_delta`, which for a thinking model includes the whole thinking phase.

With `display: "omitted"` no thinking token is ever streamed, so `time_to_first_token` records the
trailing `signature_delta` instead - the first model output actually received. That arrives *after*
thinking completes, making it a poor proxy for when generation started, which is precisely what the
accompanying [`reasoning_type`][llmeter.endpoints.base.ReasoningType] of `"redacted"` flags. See
[`AnthropicMessagesStream`][llmeter.endpoints.anthropic_messages.AnthropicMessagesStream] for detail.
"""

# Python Built-Ins:
import inspect
import logging
import time
from typing import Any, Generic, Iterable, TypeVar

# External Dependencies:
import anthropic
from anthropic.types import (
    Message,
    MessageCreateParams,
    RawMessageStreamEvent,
)
import httpx  # (Indirect dependency of anthropic)

# Local Dependencies:
from .base import (
    Endpoint,
    InvocationResponse,
    ReasoningType,
    warn_if_ttft_visible_tokens_only_set,
)

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


def _extract_thinking_tokens(usage: Any) -> int | None:
    """Read `usage.output_tokens_details.thinking_tokens` from an Anthropic usage object.

    Returns `None` when the API did not report the breakdown (older API versions, or a
    provider/gateway that strips it).

    Handles both shapes the field can arrive in, since LLMeter supports a range of `anthropic`
    SDK versions:

    * a modelled `OutputTokensDetails` object, on SDK versions that know the field
    * a plain `dict`, on older SDKs where it lands in the model's permitted "extra" fields

    Args:
        usage: An Anthropic `Usage` or `MessageDeltaUsage` object.

    Returns:
        The thinking-token count, or `None` if unavailable.
    """
    details = getattr(usage, "output_tokens_details", None)
    if details is None:
        return None
    value = (
        details.get("thinking_tokens")
        if isinstance(details, dict)
        else getattr(details, "thinking_tokens", None)
    )
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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

    # Explicit typing to keep pyright happy:
    default_reasoning_visibility: ReasoningType | None

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "anthropic-messages",
        provider: str = "anthropic",
        api_key: str | None = None,
        aws_region: str | None = None,
        default_reasoning_visibility: ReasoningType | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            endpoint_name=endpoint_name,
            model_id=model_id,
            provider=provider,
        )
        # Note we deliberately allow passing None to mean "use this endpoint's default" in line
        # with other endpoints - but for Claude this default is "summary" (correct since Claude 4):
        self.default_reasoning_visibility = default_reasoning_visibility or "summary"
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
        if isinstance(kwargs.get("timeout"), dict):
            kwargs["timeout"] = httpx.Timeout(**kwargs["timeout"])
        self._client = client_cls(**kwargs)

    def _get_llmeter_state(self) -> dict:
        """Extract serializable state by introspecting the underlying client.

        Since some constructor params are set on the underlying Anthropic client but *not*
        persisted directly on the Endpoint object, we need to extend the default serialization
        behaviour to preserve these extra config fields so they survive a serialization round trip.

        Rather than hardcoding per-provider fields (since there are different ANTHROPIC_CLIENTS
        classes), we inspect the actual client class's __init__ signature and capture any matching
        instance attributes — automatically adapting to whichever provider is in use.  Secrets
        (params containing 'key', 'token', or 'credential') and non-serializable objects are
        excluded.
        """
        skip_params = frozenset(
            {
                "http_client",
                "_strict_response_validation",
                # default_headers/default_query properties return merged SDK headers including
                # secrets; we read _custom_headers/_custom_query directly below instead.
                "default_headers",
                "default_query",
            }
        )
        skip_substrings = ("key", "token", "credential")

        state = super()._get_llmeter_state()
        sig = inspect.signature(type(self._client).__init__)
        for name, param in sig.parameters.items():
            if name == "self" or param.kind in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                continue
            if name in skip_params or any(s in name for s in skip_substrings):
                continue
            if not hasattr(self._client, name):
                continue
            val = getattr(self._client, name)
            if isinstance(val, httpx.URL):
                val = str(val)
            elif isinstance(val, httpx.Timeout):
                val = {
                    "connect": val.connect,
                    "read": val.read,
                    "write": val.write,
                    "pool": val.pool,
                }
            state[name] = val
        if self._client._custom_headers:
            state["default_headers"] = dict(self._client._custom_headers)
        if self._client._custom_query:
            state["default_query"] = dict(self._client._custom_query)
        return state

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
    The reported `num_tokens_output` is the total billed count (thinking + text), and
    `num_tokens_output_reasoning` is populated from `usage.output_tokens_details.thinking_tokens`
    where the API reports it.

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

        # Extract text from content blocks (skip thinking/redacted_thinking), noting whether the
        # model reasoned and how that reasoning was disclosed. Non-streaming responses have no
        # first-token timings, but the disclosure level is still meaningful information about the
        # request - and `num_tokens_output_reasoning` is captured below regardless.
        response_text = ""
        saw_thinking_block = False
        saw_redacted_thinking_block = False
        for block in raw_response.content:
            if block.type == "text":
                response_text += block.text
            elif block.type == "thinking":
                saw_thinking_block = True
            elif block.type == "redacted_thinking":
                saw_redacted_thinking_block = True
        response.response_text = response_text

        if saw_redacted_thinking_block:
            response.reasoning_type = "redacted"
        elif saw_thinking_block:
            response.reasoning_type = self.default_reasoning_visibility or "unknown"

        usage = raw_response.usage
        if usage:
            response.num_tokens_input = getattr(usage, "input_tokens", None)
            response.num_tokens_output = getattr(usage, "output_tokens", None)
            response.num_tokens_input_cached = getattr(
                usage, "cache_read_input_tokens", None
            )
            response.num_tokens_output_reasoning = _extract_thinking_tokens(usage)


class AnthropicMessagesStream(
    AnthropicMessagesEndpoint[Iterable[RawMessageStreamEvent]]
):
    """Endpoint for the Anthropic Messages API (streaming).

    Uses `client.messages.create(..., stream=True)` to stream SSE events, enabling
    time-to-first-token and time-to-last-token measurements.

    #### Extended thinking and TTFT

    When extended thinking is enabled, the stream contains thinking-related events before the
    visible text. Both phases are timed, on separate metrics:

    * [`time_to_first_token`][llmeter.endpoints.base.InvocationResponse] is set by the first
      `thinking_delta`, or by the first `text_delta` if the model does not think.
    * [`time_to_first_content_token`][llmeter.endpoints.base.InvocationResponse] is set by the
      first `text_delta`, and therefore includes the whole thinking phase.

    The `display` setting on the thinking configuration affects which events are emitted, and this
    has an important consequence:

    * `"summarized"` - `thinking_delta` events stream before the text, so both metrics are
      populated as described above.
    * `"omitted"` (the default on Claude Opus 4.7 and Mythos) - no `thinking_delta` events are
      emitted at all. The only pre-text signal is a `signature_delta`, which arrives *after*
      thinking is complete. `time_to_first_token` records it anyway, since it is the first model
      output received - but being an end-of-thinking timestamp it is a poor proxy for when
      generation started, and is not comparable with TTFT from a model that streams its reasoning.
      [`reasoning_type`][llmeter.endpoints.base.ReasoningType] is set to `"redacted"` to make that
      visible, and is what stops
      [`time_per_output_token`][llmeter.endpoints.base.InvocationResponse] being derived from the
      whole measured window. TPOT is still available, from the answer-only pairing that
      `num_tokens_output_reasoning` makes possible.

    A `redacted_thinking` block is treated the same way: it arrives whole (the Messages API has no
    redacted-thinking *delta* type), so it too sets `time_to_first_token` and yields `"redacted"`.
    Partial redaction - readable thinking *and* a redacted block - also resolves to `"redacted"`,
    conservatively, because some of the reasoning was not delivered as plain text.

    !!! note
        To measure first-token latency on a model that defaults to `display: "omitted"`, request
        `thinking={"type": "adaptive", "display": "summarized"}` explicitly.

    Args:
        model_id: Model identifier.
        endpoint_name: Display name.  Defaults to `"anthropic-messages"`.
        provider: Backend to use.  Defaults to `"anthropic"`.
        api_key: API key for the direct Anthropic API.
        aws_region: AWS region for Bedrock Mantle.
        default_reasoning_visibility: What to record as
            [`reasoning_type`][llmeter.endpoints.base.InvocationResponse] when thinking deltas
            *are* streamed. Defaults to `"summary"`, which is correct for Claude 4 models (they
            return summarized thinking). **Set this to `"verbatim"` for Claude 3.7 Sonnet**, which
            returns the full thinking output - the stream is identical in both cases, so LLMeter
            cannot detect the difference. Pass `"unknown"` to decline guessing. Has no effect when
            thinking is not streamed at all: that case is detected structurally and always recorded
            as `"redacted"`.
        ttft_visible_tokens_only: **Deprecated and ignored.** Both first-token metrics are now
            always recorded. Accepted only so that endpoint configurations saved by earlier
            LLMeter versions continue to load.
        **kwargs: Additional arguments forwarded to the client constructor.

    Examples:
        Direct Anthropic API:

        ```python
        endpoint = AnthropicMessagesStream(model_id="claude-opus-4-7")
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
        default_reasoning_visibility: ReasoningType | None = None,
        ttft_visible_tokens_only: bool | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_id=model_id,
            endpoint_name=endpoint_name,
            provider=provider,
            api_key=api_key,
            aws_region=aws_region,
            default_reasoning_visibility=default_reasoning_visibility,
            **kwargs,
        )
        warn_if_ttft_visible_tokens_only_set(ttft_visible_tokens_only)

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

        Only `text_delta` events contribute to `response_text`; thinking events are timed but their
        content is discarded. See the class docstring for how `display: "omitted"` affects
        `time_to_first_token`.

        Args:
            raw_response: The streaming iterator of SSE events.
            start_t: Start time of the API call.
            response: The LLMeter response object to be populated in-place.
        """
        saw_thinking_delta = False
        saw_thinking_signature = False
        saw_redacted_thinking_block = False

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

            elif event_type == "content_block_start":
                # `redacted_thinking` blocks arrive *whole* through this event: the Messages API has
                # no redacted-thinking delta type, so without handling it an entirely-redacted
                # thinking block is invisible. There would be no `thinking_delta`, and no
                # `signature_delta` either (the block carries `data`, not `signature`), so the
                # response would be silently mislabelled as having done no reasoning at all -- and
                # TTFT would be taken from the first *text* delta.
                block_type = getattr(
                    getattr(event, "content_block", None), "type", None
                )
                if block_type == "redacted_thinking":
                    saw_redacted_thinking_block = True
                    if response.time_to_first_token is None:
                        response.time_to_first_token = now - start_t

            elif event_type == "content_block_delta":
                delta = event.delta
                delta_type = getattr(delta, "type", None)

                if delta_type == "thinking_delta":
                    # A real output token, just not a visible one.
                    saw_thinking_delta = True
                    if response.time_to_first_token is None:
                        response.time_to_first_token = now - start_t

                elif delta_type == "signature_delta":
                    # The encrypted thinking content. It arrives after the thinking block is
                    # complete, so it is a poor proxy for when generation *started* -- but it is
                    # still the first model output received, which is what TTFT measures. The
                    # accompanying `reasoning_type="redacted"` is what tells consumers (and the
                    # Runner) not to pair this TTFT against the full output token count.
                    saw_thinking_signature = True
                    if response.time_to_first_token is None:
                        response.time_to_first_token = now - start_t

                elif delta_type == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        if response.time_to_first_token is None:
                            response.time_to_first_token = now - start_t
                        if response.time_to_first_content_token is None:
                            response.time_to_first_content_token = now - start_t
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
                    # `message_delta` usage is cumulative, so overwrite rather than accumulate.
                    # Guarded so a later delta without the breakdown can't clear an earlier value.
                    thinking_tokens = _extract_thinking_tokens(event.usage)
                    if thinking_tokens is not None:
                        response.num_tokens_output_reasoning = thinking_tokens

        if saw_redacted_thinking_block:
            # Positive evidence of redaction takes precedence over any declared default, including
            # when only *part* of the thinking was redacted: the answer-only pairing is valid either
            # way, so preferring it is the conservative choice.
            response.reasoning_type = "redacted"
        elif saw_thinking_delta:
            # Thinking was streamed, but whether it is the raw reasoning or a summary of it depends
            # on the model generation, not on anything in the stream: Claude 3.7 Sonnet returns the
            # full thinking output, while Claude 4 models return summarized thinking. Since that is
            # not observable here, fall back to what the caller declared (resolved in __init__).
            # The `or` is a guard for a subclass that clears the attribute: `"unknown"` records that
            # reasoning happened without claiming how it was disclosed, which is distinct from
            # `None` (no reasoning at all).
            response.reasoning_type = self.default_reasoning_visibility or "unknown"
        elif saw_thinking_signature:
            # `display: "omitted"`: the model demonstrably produced thinking tokens -- hence the
            # signature -- but none of their content was streamed. Note a signature also accompanies
            # *readable* thinking, which is why this is only reached when no thinking delta was seen.
            response.reasoning_type = "redacted"
