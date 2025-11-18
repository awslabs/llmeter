# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Dict, Sequence
from uuid import uuid4

import jmespath
from openai import APIConnectionError, OpenAI
from openai.types.chat import ChatCompletion

from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)


class OpenAIEndpoint(Endpoint):
    """Base class for OpenAI API endpoints.

    Provides common functionality for interacting with OpenAI's API endpoints.
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "openai",
        api_key: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        """Initialize OpenAI endpoint.

        Args:
            model_id (str): ID of the OpenAI model to use
            endpoint_name (str, optional): Name of the endpoint. Defaults to "openai".
            api_key (str | None, optional): OpenAI API key. Defaults to None.
            provider (str, optional): Provider name. Defaults to "openai".
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(
            endpoint_name,
            model_id,
            provider=provider,
        )

        self._client = OpenAI(api_key=api_key, **kwargs)

    def _parse_payload(self, payload):
        """Parse the message content from the payload.

        Args:
            payload (dict): Request payload containing messages

        Returns:
            str: Concatenated message contents
        """
        jmes_path = "[:].content"
        messages = payload.get("messages")
        # return "\n".join([k for j in jmespath.search(jmes_path, messages) for k in j])
        return "\n".join(jmespath.search(jmes_path, messages))

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str], max_tokens: int = 256, **kwargs
    ):
        """Create a payload for the OpenAI API request.

        Args:
            user_message (str | Sequence[str]): User message(s) to send
            max_tokens (int, optional): Maximum tokens in response. Defaults to 256.
            **kwargs: Additional payload parameters

        Returns:
            dict: Formatted payload for API request
        """
        if isinstance(user_message, str):
            user_message = [user_message]
        payload = {
            "messages": [{"role": "user", "content": k} for k in user_message],
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)
        return payload


class OpenAICompletionEndpoint(OpenAIEndpoint):
    """Endpoint for OpenAI chat completion API."""

    def invoke(self, payload: Dict, **kwargs) -> InvocationResponse:
        """Invoke the OpenAI chat completion API.

        Args:
            payload (Dict): Request payload
            **kwargs: Additional parameters for the request

        Returns:
            InvocationResponse: Response from the API
        """
        payload = {**kwargs, **payload}
        payload["model"] = self.model_id

        start_t = time.perf_counter()
        try:
            client_response: ChatCompletion = self._client.chat.completions.create(
                **payload
            )

        except (APIConnectionError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )

        response = self._parse_converse_response(client_response, start_t)
        response.input_payload = payload
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_converse_response(self, client_response: ChatCompletion, start_t: float):
        """Parse the OpenAI chat completion API response into an InvocationResponse object.

        Args:
            client_response (ChatCompletion): Raw response from OpenAI chat completion API
            start_t (float): Start time of the API call in seconds

        Returns:
            InvocationResponse: Parsed response object containing:
                - Response ID
                - Response text content
                - Token counts for input/output
                - Time to last token
        """

        usage = client_response.usage

        return InvocationResponse(
            id=client_response.id,
            response_text=client_response.choices[0].message.content,
            num_tokens_input=usage and usage.prompt_tokens,
            num_tokens_output=usage and usage.completion_tokens,
            time_to_last_token=time.perf_counter() - start_t,
        )


class OpenAICompletionStreamEndpoint(OpenAIEndpoint):
    """Endpoint for OpenAI streaming chat completion API."""

    def invoke(self, payload: Dict, **kwargs) -> InvocationResponse:
        """Invoke the OpenAI streaming chat completion API.

        Args:
            payload (Dict): Request payload
            **kwargs: Additional parameters for the request

        Returns:
            InvocationResponse: Response from the API
        """
        payload = {**kwargs, **payload}

        payload["model"] = self.model_id

        if not payload.get("stream"):
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        try:
            start_t = time.perf_counter()
            client_response: ChatCompletion = self._client.chat.completions.create(
                **payload
            )
        except (APIConnectionError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        response = self._parse_converse_stream_response(client_response, start_t)
        response.input_payload = payload
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_converse_stream_response(
        self, client_response: ChatCompletion, start_t: float
    ) -> InvocationResponse:
        """Parse the streaming API response from OpenAI chat completion API.

        Args:
            client_response (ChatCompletion): Raw API response stream containing chunks of completion text
            start_t (float): Start time of the API call in seconds

        Returns:
            InvocationResponse: Parsed response object containing:
                - Response ID
                - Concatenated response text from all chunks
                - Token counts for input/output
                - Time to first token and last token
        """

        response_id = None
        prompt_tokens = None
        completion_tokens = None

        first_chunk = next(client_response)  # type: ignore
        time_to_first_token = time.perf_counter() - start_t
        if response_id is None:
            response_id = first_chunk.id  # type: ignore
        response_text = first_chunk.choices[0].delta.content or ""

        for chunk in client_response:
            if chunk.choices[0].delta.content is not None:  # type: ignore
                response_text += chunk.choices[0].delta.content or ""  # type: ignore
            if hasattr(chunk, "usage") and chunk.usage is not None:  # type: ignore
                prompt_tokens = chunk.usage.prompt_tokens  # type: ignore
                completion_tokens = chunk.usage.completion_tokens  # type: ignore
        time_to_last_token = time.perf_counter() - start_t

        return InvocationResponse(
            id=response_id,
            response_text=response_text,
            num_tokens_input=prompt_tokens,
            num_tokens_output=completion_tokens,
            time_to_first_token=time_to_first_token,
            time_to_last_token=time_to_last_token,
        )
