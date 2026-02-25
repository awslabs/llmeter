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


class ResponseEndpoint(OpenAIEndpoint):
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
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except AuthenticationError as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except RateLimitError as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except BadRequestError as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except Exception as e:
            logger.error(e)
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

        return InvocationResponse(
            id=client_response.id,
            response_text=response_text,
            num_tokens_input=usage and usage.prompt_tokens,
            num_tokens_output=usage and usage.completion_tokens,
            time_to_last_token=time.perf_counter() - start_t,
        )

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str],
        max_tokens: int = 256,
        instructions: str | None = None,
        **kwargs,
    ) -> Dict:
        """Create a payload for the Responses API request.

        Args:
            user_message: User message(s) to send (can be string or array of messages)
            max_tokens: Maximum tokens in response (default: 256)
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
            "max_tokens": max_tokens,
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


class ResponseStreamEndpoint(OpenAIEndpoint):
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
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except AuthenticationError as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except RateLimitError as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except BadRequestError as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        except Exception as e:
            logger.error(e)
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

        Args:
            client_response: Streaming response from OpenAI Responses API
            start_t: Start time of the API call

        Returns:
            InvocationResponse with extracted fields including TTFT
        """
        prompt_tokens = None
        completion_tokens = None
        response_text = ""

        # Process first chunk to get TTFT
        first_chunk = next(client_response)
        time_to_first_token = time.perf_counter() - start_t
        response_id = first_chunk.id

        # Extract text from first chunk's output array
        if first_chunk.output and len(first_chunk.output) > 0:
            for item in first_chunk.output:
                if item.type == "message" and hasattr(item, "content"):
                    for content_item in item.content:
                        if content_item.type == "output_text":
                            response_text += content_item.text or ""

        # Process remaining chunks
        for chunk in client_response:
            # Extract text from output delta items
            if chunk.output and len(chunk.output) > 0:
                for item in chunk.output:
                    if item.type == "message" and hasattr(item, "content"):
                        for content_item in item.content:
                            if content_item.type == "output_text":
                                response_text += content_item.text or ""

            # Extract usage from final chunk
            if hasattr(chunk, "usage") and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        time_to_last_token = time.perf_counter() - start_t

        return InvocationResponse(
            id=response_id,
            response_text=response_text,
            num_tokens_input=prompt_tokens,
            num_tokens_output=completion_tokens,
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
