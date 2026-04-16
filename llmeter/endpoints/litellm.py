# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import time
from typing import Any, Sequence

import litellm
from litellm import CustomStreamWrapper, completion
from litellm.types.utils import ModelResponse
from litellm.utils import get_llm_provider  # type: ignore

from . import Endpoint, InvocationResponse

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

    def prepare_payload(self, payload, **kwargs):
        if kwargs:
            payload = {**payload, **kwargs}
        return payload

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str],
        max_tokens: int = 256,
        system_message: str | None = None,
        **kwargs: Any,
    ) -> dict:
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
    def invoke(self, payload):
        response = completion(model=self.litellm_model, **payload)
        if not isinstance(response, ModelResponse):
            raise ValueError(f"Expected ModelResponse, got {type(response)}")
        return self.parse_response(response, self._start_t)

    def parse_response(
        self, client_response: ModelResponse, start_t: float
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
    def invoke(self, payload):
        response = completion(model=self.litellm_model, **payload)

        if not isinstance(response, CustomStreamWrapper):
            raise ValueError(f"Expected CustomStreamWrapper, got {type(response)}")
        return self.parse_response(response, self._start_t)

    def prepare_payload(self, payload, **kwargs):
        # Make a copy of payload to avoid modifying the original
        payload_copy = payload.copy()

        # Merge kwargs, excluding stream-related keys (we control those)
        for key, value in kwargs.items():
            if key not in ["stream", "stream_options"]:
                payload_copy[key] = value

        # Ensure streaming is enabled
        payload_copy["stream"] = True

        # Handle stream_options - merge if exists in kwargs, otherwise set default
        if "stream_options" in kwargs:
            existing_options = kwargs.get("stream_options", {})
            payload_copy["stream_options"] = {**existing_options, "include_usage": True}
        elif "stream_options" not in payload_copy:
            payload_copy["stream_options"] = {"include_usage": True}
        else:
            existing_options = payload_copy.get("stream_options", {})
            payload_copy["stream_options"] = {**existing_options, "include_usage": True}

        return payload_copy

    def parse_response(
        self, client_response: CustomStreamWrapper, start_t: float
    ) -> InvocationResponse:
        usage = None
        time_flag = True
        time_to_first_token = None
        output_text = ""
        id = None

        for chunk in client_response:
            content = chunk.choices[0].delta.content or ""  # type: ignore
            output_text += content

            # Record time to first token only when we get actual content
            if time_flag and content:
                time_to_first_token = time.perf_counter() - start_t
                time_flag = False

            # Always capture the ID from the first chunk
            if id is None:
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
        return response
