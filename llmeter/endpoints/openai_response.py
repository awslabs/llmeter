# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Python Built-Ins:
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Iterable, TypeVar, cast

# External Dependencies:
import httpx  # (Indirect dependency of OpenAI)
from openai import OpenAI
from openai._constants import DEFAULT_MAX_RETRIES
from openai.types.responses import Response, ResponseCreateParams, ResponseStreamEvent
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
    ResponseCreateParamsStreaming,
)

# Local Dependencies:
from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)

TOpenAIResponseBase = TypeVar(
    "TOpenAIResponseBase", bound=Response | Iterable[ResponseStreamEvent]
)


class OpenAIEndpointBase(Endpoint[TOpenAIResponseBase], Generic[TOpenAIResponseBase]):
    """Base class for OpenAI Responses API endpoints (streaming and non-streaming)"""

    def __init__(
        self,
        endpoint_name: str,
        model_id: str,
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
        """Initialize Response API endpoint.

        Args:
            endpoint_name: Name of the endpoint
            model_id: ID of the OpenAI model to use
            api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)
            provider: Provider name (default: "openai")
            organization: OpenAI organization ID. Defaults to None.
            project: OpenAI project ID. Defaults to None.
            base_url: Override the default base URL for the API.
            websocket_base_url: Override the default base URL for websocket connections.
            timeout: Request timeout in seconds, or an httpx.Timeout for granular control. If a
                dict is passed (as is the case after serialization), it'll be interpreted as a set
                of arguments to create an httpx.Timeout.
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
        Endpoint object.
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

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str],
        max_output_tokens: int = 256,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> ResponseCreateParams:
        """Create a payload for the Responses API request.

        This is a convenience helper. You can also build the payload directly
        using ``openai.types.responses.ResponseCreateParams``.

        Args:
            user_message: User message(s) to send (can be string or array of messages)
            max_output_tokens: Maximum tokens in response (default: 256)
            instructions: Optional system-level instructions
            **kwargs: Additional payload parameters (temperature, top_p, text.format, etc.)

        Returns:
            ResponseCreateParams formatted for Responses API
        """
        if isinstance(user_message, str):
            input_value: str | list[dict] = user_message
        else:
            input_value = [{"role": "user", "content": msg} for msg in user_message]

        payload: dict = {
            "input": input_value,
            "max_output_tokens": max_output_tokens,
        }

        if instructions:
            payload["instructions"] = instructions

        payload.update(kwargs)
        return cast(ResponseCreateParams, payload)

    def _parse_payload(self, payload: ResponseCreateParams | dict):
        """Extract the user message text from a Response API payload.

        Handles both string input and message-array input formats.

        Args:
            payload: Request payload containing ``input``

        Returns:
            str: Extracted message content
        """
        input_value = payload.get("input")
        if input_value is None:
            return ""
        if isinstance(input_value, str):
            return input_value
        if isinstance(input_value, list):
            contents: list[str] = []
            for msg in input_value:
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


class OpenAIResponseEndpoint(OpenAIEndpointBase[Response]):
    """Endpoint for OpenAI Responses API (non-streaming).

    This endpoint provides access to OpenAI's newer Responses API which offers
    structured outputs, better response format control, and improved multi-turn
    conversation handling.
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "openai-response",
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
        """Initialize Response API endpoint.

        Args:
            model_id: ID of the OpenAI model to use
            endpoint_name: Name of the endpoint (default: "openai-response")
            api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)
            provider: Provider name (default: "openai")
            organization: OpenAI organization ID. Defaults to None.
            project: OpenAI project ID. Defaults to None.
            base_url: Override the default base URL for the API.
            websocket_base_url: Override the default base URL for websocket connections.
            timeout: Request timeout in seconds, or an httpx.Timeout for granular control.
            max_retries: Maximum number of retries for failed requests.
            default_headers: Additional headers to send with every request.
            default_query: Additional query parameters to send with every request.
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(
            endpoint_name,
            model_id,
            api_key=api_key,
            provider=provider,
            organization=organization,
            project=project,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            **kwargs,
        )

    @OpenAIEndpointBase.llmeter_invoke
    def invoke(self, payload: ResponseCreateParamsNonStreaming) -> Response:
        """Invoke the Responses API."""
        client_response = self._client.responses.create(**payload)
        return client_response

    def prepare_payload(self, payload):
        """Ensure payload specifies correct model ID and streaming disabled"""
        return {
            **payload,
            "model": self.model_id,
            "stream": False,
        }

    def process_raw_response(
        self, raw_response: Response, start_t: float, response: InvocationResponse
    ) -> None:
        # Check for API-level errors (e.g. status="failed" with an error object)
        if getattr(raw_response, "status", None) == "failed":
            error_obj = getattr(raw_response, "error", None)
            if error_obj is not None:
                error_msg = getattr(error_obj, "message", None) or str(error_obj)
                error_code = getattr(error_obj, "code", None)
                if error_code:
                    error_msg = f"{error_code}: {error_msg}"
            else:
                error_msg = "Response API request failed"
            response.error = error_msg
            return

        response.time_to_last_token = time.perf_counter() - start_t
        response.id = raw_response.id
        response.response_text = raw_response.output_text

        usage = raw_response.usage
        if usage is not None:
            response.num_tokens_input = usage.input_tokens
            response.num_tokens_output = usage.output_tokens
            details = getattr(usage, "input_tokens_details", None)
            if details:
                response.num_tokens_input_cached = getattr(
                    details, "cached_tokens", None
                )
            output_details = getattr(usage, "output_tokens_details", None)
            if output_details:
                response.num_tokens_output_reasoning = getattr(
                    output_details, "reasoning_tokens", None
                )


class OpenAIResponseStreamEndpoint(OpenAIEndpointBase[Iterable[ResponseStreamEvent]]):
    """Endpoint for OpenAI Responses API (streaming).

    This endpoint provides streaming access to OpenAI's Responses API, enabling time-to-first-token
    measurements and incremental response processing.

    Args:
        ttft_visible_tokens_only: Controls how `time_to_first_token` is measured for reasoning
            models. When `True` (default), TTFT records the time to the first *visible* text token
            (`response.output_text.delta`), ignoring reasoning events. When `False`, TTFT records
            the time to the first token of any kind — including reasoning summary or reasoning text
            deltas — giving a measure of when the model first started producing output. Has no
            effect for non-reasoning models.
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "openai-response-stream",
        api_key: str | None = None,
        provider: str = "openai",
        ttft_visible_tokens_only: bool = True,
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
        """Initialize streaming Response API endpoint.

        Args:
            model_id: ID of the OpenAI model to use
            endpoint_name: Name of the endpoint (default: "openai-response-stream")
            api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)
            provider: Provider name (default: "openai")
            ttft_visible_tokens_only: When True (default), TTFT measures time to first visible text
                token. When False, TTFT includes reasoning token events.
            organization: OpenAI organization ID. Defaults to None.
            project: OpenAI project ID. Defaults to None.
            base_url: Override the default base URL for the API.
            websocket_base_url: Override the default base URL for websocket connections.
            timeout: Request timeout in seconds, or an httpx.Timeout for granular control.
            max_retries: Maximum number of retries for failed requests.
            default_headers: Additional headers to send with every request.
            default_query: Additional query parameters to send with every request.
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(
            endpoint_name,
            model_id,
            api_key=api_key,
            provider=provider,
            organization=organization,
            project=project,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            **kwargs,
        )
        self.ttft_visible_tokens_only = ttft_visible_tokens_only

    @OpenAIEndpointBase.llmeter_invoke
    def invoke(self, payload: ResponseCreateParamsStreaming):
        """Invoke the Responses API with streaming."""
        client_response = self._client.responses.create(**payload)
        return client_response

    def prepare_payload(self, payload):
        """Ensure payload specifies correct model ID and streaming options"""
        payload = {**payload, "model": self.model_id}
        if not payload.get("stream"):
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    def process_raw_response(
        self,
        raw_response: Iterable[ResponseStreamEvent],
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        """Parse streaming Response API output into InvocationResponse.

        Processes typed events from the stream:

        - `ResponseCreatedEvent`: captures `response.id`
        - `ResponseTextDeltaEvent`: accumulates text deltas, records TTFT
        - `ResponseCompletedEvent`: extracts usage from `response.usage`
        - `ResponseFailedEvent`: captures API-level errors
        - Reasoning events (`response.reasoning_summary_text.delta`,
          `response.reasoning_text.delta`): when `ttft_visible_tokens_only` is ``False``, these set
          TTFT on the first reasoning token.
        """
        _REASONING_DELTA_TYPES = frozenset(
            (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            )
        )

        for event in raw_response:
            now = time.perf_counter()
            if event.type == "response.created":
                response.id = event.response.id

            elif event.type == "response.output_text.delta":
                if response.response_text is None:
                    response.response_text = event.delta
                    if response.time_to_first_token is None:
                        response.time_to_first_token = now - start_t
                else:
                    response.response_text += event.delta
                response.time_to_last_token = now - start_t

            elif (
                not self.ttft_visible_tokens_only
                and event.type in _REASONING_DELTA_TYPES
            ):
                if response.time_to_first_token is None:
                    response.time_to_first_token = now - start_t

            elif event.type == "response.completed":
                usage = event.response.usage
                if usage is not None:
                    response.num_tokens_input = usage.input_tokens
                    response.num_tokens_output = usage.output_tokens
                    details = getattr(usage, "input_tokens_details", None)
                    if details:
                        response.num_tokens_input_cached = getattr(
                            details, "cached_tokens", None
                        )
                    output_details = getattr(usage, "output_tokens_details", None)
                    if output_details:
                        response.num_tokens_output_reasoning = getattr(
                            output_details, "reasoning_tokens", None
                        )

            elif event.type == "response.failed":
                error_obj = getattr(event.response, "error", None)
                if error_obj is not None:
                    error_msg = getattr(error_obj, "message", None) or str(error_obj)
                    error_code = getattr(error_obj, "code", None)
                    if error_code:
                        error_msg = f"{error_code}: {error_msg}"
                else:
                    error_msg = "Response API request failed"
                response.error = error_msg
                response.time_to_last_token = now - start_t
