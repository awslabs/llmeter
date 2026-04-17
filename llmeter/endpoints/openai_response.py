# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections.abc import Sequence
from typing import Any, cast

from openai import (
    OpenAI,
)
from openai.types.responses import Response, ResponseCreateParams
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)

from .base import Endpoint, InvocationResponse, llmeter_invoke

logger = logging.getLogger(__name__)


class OpenAIResponseEndpoint(Endpoint):
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
        **kwargs: Any,
    ):
        """Initialize Response API endpoint.

        Args:
            model_id: ID of the OpenAI model to use
            endpoint_name: Name of the endpoint (default: "openai-response")
            api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)
            provider: Provider name (default: "openai")
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(endpoint_name, model_id, provider=provider)
        self._client = OpenAI(api_key=api_key, **kwargs)

    @llmeter_invoke
    def invoke(self, payload: ResponseCreateParamsNonStreaming) -> InvocationResponse:
        """Invoke the Responses API."""
        client_response = self._client.responses.create(**payload)
        return self.parse_response(client_response, self._start_t)

    def prepare_payload(self, payload, **kwargs):
        payload = {**kwargs, **payload}  # type: ignore
        payload["model"] = self.model_id
        return payload

    def parse_response(
        self, client_response: Response, start_t: float
    ) -> InvocationResponse:
        """Parse Response API output into InvocationResponse.

        Args:
            client_response: Raw ``Response`` object from OpenAI Responses API
            start_t: Start time of the API call

        Returns:
            InvocationResponse with extracted fields
        """
        usage = client_response.usage

        input_tokens = None
        output_tokens = None
        cached_tokens = None
        if usage is not None:
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            details = getattr(usage, "input_tokens_details", None)
            if details:
                cached_tokens = getattr(details, "cached_tokens", None)

        return InvocationResponse(
            id=client_response.id,
            response_text=client_response.output_text,
            num_tokens_input=input_tokens,
            num_tokens_output=output_tokens,
            num_tokens_input_cached=cached_tokens,
        )

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str],
        max_output_tokens: int = 256,
        instructions: str | None = None,
        **kwargs,
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


class OpenAIResponseStreamEndpoint(OpenAIResponseEndpoint):
    """Endpoint for OpenAI Responses API (streaming).

    This endpoint provides streaming access to OpenAI's Responses API, enabling
    time-to-first-token measurements and incremental response processing.
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "openai-response-stream",
        api_key: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        """Initialize streaming Response API endpoint.

        Args:
            model_id: ID of the OpenAI model to use
            endpoint_name: Name of the endpoint (default: "openai-response-stream")
            api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)
            provider: Provider name (default: "openai")
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(
            model_id=model_id,
            endpoint_name=endpoint_name,
            api_key=api_key,
            provider=provider,
            **kwargs,
        )

    @llmeter_invoke
    def invoke(self, payload: ResponseCreateParams) -> InvocationResponse:
        """Invoke the Responses API with streaming."""
        client_response = self._client.responses.create(**payload)
        return self.parse_response(client_response, self._start_t)

    def prepare_payload(self, payload, **kwargs):
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id
        if not payload.get("stream"):
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    def parse_response(self, client_response, start_t: float) -> InvocationResponse:
        """Parse streaming Response API output into InvocationResponse.

        Processes typed events from the stream:
        - ``ResponseCreatedEvent``: captures ``response.id``
        - ``ResponseTextDeltaEvent``: accumulates text deltas, records TTFT
        - ``ResponseCompletedEvent``: extracts usage from ``response.usage``

        Args:
            client_response: Streaming response from OpenAI Responses API
            start_t: Start time of the API call

        Returns:
            InvocationResponse with extracted fields including TTFT
        """
        input_tokens = None
        output_tokens = None
        cached_tokens = None
        response_text = ""
        response_id = None
        time_to_first_token = None

        for event in client_response:
            if event.type == "response.created":
                response_id = event.response.id

            elif event.type == "response.output_text.delta":
                if time_to_first_token is None:
                    time_to_first_token = time.perf_counter() - start_t
                response_text += event.delta

            elif event.type == "response.completed":
                usage = event.response.usage
                if usage is not None:
                    input_tokens = usage.input_tokens
                    output_tokens = usage.output_tokens
                    details = getattr(usage, "input_tokens_details", None)
                    if details:
                        cached_tokens = getattr(details, "cached_tokens", None)

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
