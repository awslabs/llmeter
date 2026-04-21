# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import time
from typing import Any, Generic, Sequence, TypeVar

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

TLiteLLMResponseBase = TypeVar(
    "TLiteLLMResponseBase",
    bound=CustomStreamWrapper | ModelResponse,
)


class LiteLLMBase(Endpoint[TLiteLLMResponseBase], Generic[TLiteLLMResponseBase]):
    """Base class for (streaming or non-streaming) LiteLLM-based Endpoints"""

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


class LiteLLM(LiteLLMBase[ModelResponse]):
    """Endpoint for LiteLLM SDK-based models (non-streaming mode)"""

    @LiteLLMBase.llmeter_invoke
    def invoke(self, payload) -> ModelResponse:
        # In non-streaming mode, completion always returns a ModelResponse:
        return completion(**payload)  # type: ignore

    def prepare_payload(self, payload: dict) -> dict:
        # Make a copy of payload to avoid modifying the original
        payload_copy = payload.copy()
        # Ensure correct model ID
        payload_copy["model"] = self.litellm_model
        # Ensure streaming is disabled
        payload_copy["stream"] = False
        return payload_copy

    def process_raw_response(
        self, raw_response, start_t: float, response: InvocationResponse
    ) -> None:
        response.time_to_last_token = time.perf_counter() - start_t
        response.id = raw_response.id

        try:
            usage = raw_response.usage  # type: ignore
            response.num_tokens_input = usage.prompt_tokens
            response.num_tokens_output = usage.completion_tokens
        except AttributeError:
            pass

        response.response_text = raw_response.choices[0].message.content


class LiteLLMStreaming(LiteLLMBase[CustomStreamWrapper]):
    @LiteLLMBase.llmeter_invoke
    def invoke(self, payload) -> CustomStreamWrapper:
        # In streaming mode, completion always returns a CustomStreamWrapper:
        return completion(**payload)  # type: ignore

    def prepare_payload(self, payload):
        # Make a copy of payload to avoid modifying the original
        payload_copy = payload.copy()

        # Ensure correct model ID
        payload_copy["model"] = self.litellm_model

        # Ensure streaming is enabled
        payload_copy["stream"] = True

        # Ensure stream_options includes usage
        existing_options = payload_copy.get("stream_options", {})
        payload_copy["stream_options"] = {**existing_options, "include_usage": True}

        return payload_copy

    def process_raw_response(
        self,
        raw_response: CustomStreamWrapper,
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        usage = None
        got_chunk_id = False

        for chunk in raw_response:
            now = time.perf_counter()

            if not got_chunk_id and chunk.id is not None:
                response.id = chunk.id
                got_chunk_id = True

            content = chunk.choices[0].delta.content or ""
            if content:
                if response.response_text is None:
                    response.response_text = content
                    response.time_to_first_token = now - start_t
                else:
                    response.response_text += content
                response.time_to_last_token = now - start_t

            try:
                usage = chunk.usage  # type: ignore
            except AttributeError:
                continue

        if usage:
            response.num_tokens_input = usage.prompt_tokens
            response.num_tokens_output = usage.completion_tokens
