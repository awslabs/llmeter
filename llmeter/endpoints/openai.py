# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Sequence
from uuid import uuid4

import jmespath
from openai import APIConnectionError, OpenAI
from openai.types.chat import ChatCompletion

from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)


class OpenAIEndpoint(Endpoint):
    def __init__(
        self,
        model_id: str,
        endpoint_name: str = "openai",
        api_key: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            endpoint_name,
            model_id,
            *args,
            **kwargs,
        )

        self._client = OpenAI(api_key=api_key, **kwargs)

    def _parse_payload(self, payload):
        jpath = "[:].content"
        messages = payload.get("messages")
        return "\n".join([k for j in jmespath.search(jpath, messages) for k in j])

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str], max_tokens: int = 256, **kwargs
    ):
        if isinstance(user_message, str):
            user_message = [user_message]
        payload = {
            "messages": [{"role": "user", "content": k} for k in user_message],
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)
        return payload


class OpenAICompletionEndpoint(OpenAIEndpoint):
    def invoke(self, payload: Dict, **kwargs) -> InvocationResponse:
        payload = {**kwargs, **payload}

        payload["model"] = self.model_id
        try:
            client_response: ChatCompletion = self._client.chat.completions.create(
                **payload
            )
        except (APIConnectionError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(id=uuid4().hex, error=str(e))
        response = self._parse_converse_response(client_response)
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_converse_response(self, client_response: ChatCompletion):
        usage = client_response.usage

        return InvocationResponse(
            id=client_response.id,
            response_text=client_response.choices[0].message.content,
            num_tokens_input=usage and usage.prompt_tokens,
            num_tokens_output=usage and usage.completion_tokens,
        )
