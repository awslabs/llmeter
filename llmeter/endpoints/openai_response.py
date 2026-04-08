# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Dict, Sequence
from uuid import uuid4

from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from .base import InvocationResponse
from .openai import OpenAIEndpoint

logger = logging.getLogger(__name__)


class OpenAIResponseEndpoint(OpenAIEndpoint):
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
        **kwargs,
    ):
        """Initialize Response API endpoint.

        Args:
            model_id: ID of the OpenAI model to use
            endpoint_name: Name of the endpoint (default: "openai-response")
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

    def invoke(self, payload: Dict, **kwargs) -> InvocationResponse:
        """Invoke the Responses API.

        Args:
            payload: Request payload containing messages and parameters
            **kwargs: Additional parameters to merge with payload

        Returns:
            InvocationResponse with response text, timing, and token counts
        """
        # Merge kwargs with payload and add model_id
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id

        start_t = time.perf_counter()
        try:
            client_response = self._client.responses.create(**payload)
        except APIConnectionError as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except AuthenticationError as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except RateLimitError as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except BadRequestError as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except Exception as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )

        response = self._parse_response(client_response, start_t)
        response.input_payload = payload
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_response(self, client_response, start_t: float) -> InvocationResponse:
        """Parse Response API output into InvocationResponse.

        Args:
            client_response: Raw response from OpenAI Responses API
            start_t: Start time of the API call

        Returns:
            InvocationResponse with extracted fields
        """
        usage = client_response.usage

        # Use output_text helper to extract text directly
        response_text = client_response.output_text

        # Usage may be None depending on the provider (e.g. Bedrock Mantle)
        input_tokens = None
        output_tokens = None
        if usage is not None:
            # Response API uses input_tokens/output_tokens,
            # but some providers may use prompt_tokens/completion_tokens
            input_tokens = getattr(usage, "input_tokens", None)
            if input_tokens is None:
                input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            if output_tokens is None:
                output_tokens = getattr(usage, "completion_tokens", None)

        return InvocationResponse(
            id=client_response.id,
            response_text=response_text,
            num_tokens_input=input_tokens,
            num_tokens_output=output_tokens,
            time_to_last_token=time.perf_counter() - start_t,
        )

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str],
        max_output_tokens: int = 256,
        instructions: str | None = None,
        **kwargs,
    ) -> Dict:
        """Create a payload for the Responses API request.

        Args:
            user_message: User message(s) to send (can be string or array of messages)
            max_output_tokens: Maximum tokens in response (default: 256)
            instructions: Optional system-level instructions
            **kwargs: Additional payload parameters (temperature, top_p, text.format, etc.)

        Returns:
            Dict formatted for Responses API
        """
        # Handle input field - can be string or array of messages
        if isinstance(user_message, str):
            input_value = user_message
        else:
            # Convert sequence of strings to message array
            input_value = [{"role": "user", "content": msg} for msg in user_message]

        payload = {
            "input": input_value,
            "max_output_tokens": max_output_tokens,
        }

        # Add instructions if provided
        if instructions:
            payload["instructions"] = instructions

        payload.update(kwargs)
        return payload

    def _parse_payload(self, payload):
        """Parse the message content from the payload.

        Overrides the base class method to handle Response API format which uses
        'input' field instead of 'messages' field.

        Args:
            payload (dict): Request payload containing input

        Returns:
            str: Extracted message content
        """
        input_value = payload.get("input")

        # Handle None or missing input
        if input_value is None:
            return ""

        # Handle string input (return as-is)
        if isinstance(input_value, str):
            return input_value

        # Handle message array input (concatenate contents)
        if isinstance(input_value, list):
            contents = []
            for msg in input_value:
                if isinstance(msg, dict) and "content" in msg:
                    contents.append(msg["content"])
            return "\n".join(contents)

        # Return empty string if no messages
        return ""


class OpenAIResponseStreamEndpoint(OpenAIEndpoint):
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

    def invoke(self, payload: Dict, **kwargs) -> InvocationResponse:
        """Invoke the Responses API with streaming.

        Args:
            payload: Request payload containing messages and parameters
            **kwargs: Additional parameters to merge with payload

        Returns:
            InvocationResponse with response text, timing (TTFT, TTLT), and token counts
        """
        # Merge kwargs with payload and add model_id
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id

        # Set streaming parameters
        if not payload.get("stream"):
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        try:
            start_t = time.perf_counter()
            client_response = self._client.responses.create(**payload)
        except APIConnectionError as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except AuthenticationError as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except RateLimitError as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except BadRequestError as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except Exception as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )

        response = self._parse_stream_response(client_response, start_t)
        response.input_payload = payload
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_stream_response(
        self, client_response, start_t: float
    ) -> InvocationResponse:
        """Parse streaming Response API output into InvocationResponse.

        The Response API streams typed events (not chunks with output arrays):
        - ResponseCreatedEvent: contains response.id
        - ResponseTextDeltaEvent: contains delta (str) with text fragment
        - ResponseCompletedEvent: contains response with full output and usage

        Args:
            client_response: Streaming response from OpenAI Responses API
            start_t: Start time of the API call

        Returns:
            InvocationResponse with extracted fields including TTFT
        """
        input_tokens = None
        output_tokens = None
        response_text = ""
        response_id = None
        time_to_first_token = None

        for event in client_response:
            # Capture response ID from the first event that has it
            if response_id is None:
                if hasattr(event, "response") and event.response:
                    response_id = event.response.id
                elif hasattr(event, "id"):
                    response_id = event.id

            # ResponseTextDeltaEvent — incremental text output
            if event.type == "response.output_text.delta":
                if time_to_first_token is None:
                    time_to_first_token = time.perf_counter() - start_t
                response_text += event.delta or ""

            # ResponseCompletedEvent — final event with full response and usage
            elif event.type == "response.completed":
                if hasattr(event, "response") and event.response:
                    usage = getattr(event.response, "usage", None)
                    if usage is not None:
                        input_tokens = getattr(usage, "input_tokens", None)
                        if input_tokens is None:
                            input_tokens = getattr(usage, "prompt_tokens", None)
                        output_tokens = getattr(usage, "output_tokens", None)
                        if output_tokens is None:
                            output_tokens = getattr(
                                usage, "completion_tokens", None
                            )

        time_to_last_token = time.perf_counter() - start_t

        # If no text deltas were received, TTFT falls back to TTLT
        if time_to_first_token is None:
            time_to_first_token = time_to_last_token

        return InvocationResponse(
            id=response_id,
            response_text=response_text,
            num_tokens_input=input_tokens,
            num_tokens_output=output_tokens,
            time_to_first_token=time_to_first_token,
            time_to_last_token=time_to_last_token,
        )

    def _parse_payload(self, payload):
        """Parse the message content from the payload.

        Overrides the base class method to handle Response API format which uses
        'input' field instead of 'messages' field.

        Args:
            payload (dict): Request payload containing input

        Returns:
            str: Extracted message content
        """
        input_value = payload.get("input")

        # Handle None or missing input
        if input_value is None:
            return ""

        # Handle string input (return as-is)
        if isinstance(input_value, str):
            return input_value

        # Handle message array input (concatenate contents)
        if isinstance(input_value, list):
            contents = []
            for msg in input_value:
                if isinstance(msg, dict) and "content" in msg:
                    contents.append(msg["content"])
            return "\n".join(contents)

        # Return empty string if no messages
        return ""
