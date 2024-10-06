# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import time
from typing import Sequence
from uuid import uuid4

import litellm
from litellm import CustomStreamWrapper, completion
from litellm.types.utils import ModelResponse
from litellm.utils import get_llm_provider  # type: ignore

from llmeter.endpoints import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)

litellm.json_logs = True  # type: ignore
litellm.turn_off_message_logging = True
litellm.suppress_debug_info = True

os.environ["LITELLM_LOG"] = "CRITICAL"
os.environ["LITELLM_DONT_SHOW_FEEDBACK_BOX"] = "true"


class LiteLLMBase(Endpoint):
    def __init__(
        self,
        litellm_model: str,
        model_id: str | None = None,
    ):
        self.litellm_model = litellm_model
        model_id_inferred, provider, _, _ = get_llm_provider(litellm_model)

        logger.info(f"Using model {model_id_inferred} from provider {provider}")
        super().__init__(
            model_id=model_id or model_id_inferred,
            provider=provider,
            endpoint_name=model_id_inferred,
        )

    def _parse_payload(self, payload):
        return json.dumps(payload.get("messages"))

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str],
        max_tokens: int = 256,
        system_message: str | None = None,
        **kwargs,
    ):
        """
        Create a payload for the LiteLLM `completion()` request.

        Args:
            user_message (str | Sequence[str]): The user's message or a sequence of messages.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 256.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The formatted payload for the Bedrock API request.
        """

        if isinstance(user_message, str):
            user_message = [user_message]
        payload = {
            "messages": [{"role": "user", "content": k} for k in user_message],
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)
        if system_message:
            payload["messages"].append({"role": "system", "content": system_message})
        return payload


class LiteLLM(LiteLLMBase):
    def invoke(self, payload, **kwargs):
        try:
            response = completion(model=self.litellm_model, **payload, **kwargs)
            assert isinstance(response, ModelResponse)
            response = self._parse_converse_response(response)
            response.input_prompt = self._parse_payload(payload)
            return response

        except Exception as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                id=uuid4().hex, error=str(e), input_prompt=self._parse_payload(payload)
            )

    def _parse_converse_response(
        self, client_response: ModelResponse
    ) -> InvocationResponse:
        response = InvocationResponse(
            id=client_response.id,
            response_text=client_response.choices[0].message.content,  # type: ignore
        )
        try:
            usage = client_response.usage  # type: ignore
            response.num_tokens_input = usage.prompt_tokens
            response.num_tokens_output = usage.completion_tokens
        except AttributeError:
            pass

        return response


class LiteLLMStreaming(LiteLLMBase):
    def invoke(self, payload, **kwargs):
        if ("stream" not in kwargs) or ("stream" not in payload):
            kwargs["stream"] = True

        if ("stream_options" not in kwargs) or ("stream_options" not in payload):
            kwargs["stream_options"] = {"include_usage": True}

        try:
            start_t = time.perf_counter()
            response = completion(model=self.litellm_model, **payload, **kwargs)
        except Exception as e:
            logger.exception(e)
            return InvocationResponse.error_output(
                id=uuid4().hex, error=str(e), input_prompt=self._parse_payload(payload)
            )

        assert isinstance(response, CustomStreamWrapper)
        response = self._parse_stream(response, start_t)
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_stream(
        self, client_response: CustomStreamWrapper, start_t: float
    ) -> InvocationResponse:
        usage = None
        time_flag = True
        time_to_first_token = None
        output_text = ""
        for chunk in client_response:
            output_text += chunk.choices[0].delta.content or ""  # type: ignore
            if time_flag:
                time_to_first_token = time.perf_counter() - start_t
                time_flag = False
                id = chunk.id
            try:
                usage = chunk.usage  # type: ignore
            except AttributeError:
                continue

        time_to_last_token = time.perf_counter() - start_t

        response = InvocationResponse(
            id=id,
            response_text=output_text,
            num_tokens_input=usage and usage.prompt_tokens,
            num_tokens_output=usage and usage.completion_tokens,
            time_to_first_token=time_to_first_token,
            time_to_last_token=time_to_last_token,
        )
        if response.num_tokens_output and time_to_last_token and time_to_first_token:
            generation_time = time_to_last_token - time_to_first_token
            response.time_per_output_token = (response.num_tokens_output - 1) and (
                generation_time / (response.num_tokens_output - 1)
            )
        return response
