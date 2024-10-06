# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Dict, Sequence
from uuid import uuid4

import boto3
import jmespath
from botocore.exceptions import ClientError

from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)


class BedrockBase(Endpoint):
    def __init__(
        self,
        model_id: str,
        endpoint_name: str | None = None,
        region: str | None = None,
        inference_config: Dict | None = None,
    ):
        """
        Base class for Amazon Bedrock endpoints.

        This class provides the foundation for interacting with Amazon Bedrock services,
        including client initialization and payload handling.
        """

        super().__init__(
            model_id=model_id, endpoint_name=endpoint_name or "", provider="bedrock"
        )
        """
        Initialize the BedrockBase instance.

        Args:
            model_id (str): The ID of the model to use.
            region (str | None, optional): The AWS region to use. If None, uses the default session region.
            inference_config (Dict | None, optional): Configuration for inference.
        """

        self.endpoint_name = "amazon bedrock"

        self.region = region or boto3.session.Session().region_name
        logger.info(f"Using AWS region: {self.region}")

        self._bedrock_client = boto3.client("bedrock-runtime", region_name=self.region)
        self._inference_config = inference_config

    def _parse_payload(self, payload):
        """
        Parse the payload to extract text content.

        Args:
            payload (dict): The payload containing messages.

        Returns:
            str: Concatenated text content from the messages.
        """
        jpath = "[:].content[:].text"
        messages = payload.get("messages")
        return "\n".join([k for j in jmespath.search(jpath, messages) for k in j])

    @staticmethod
    def create_payload(
        user_message: str | Sequence[str], max_tokens: int = 256, **kwargs
    ):
        """
        Create a payload for the Bedrock Converse API request.

        Args:
            user_message (str | Sequence[str]): The user's message or a sequence of messages.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 256.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The formatted payload for the Bedrock API request.
        """

        if isinstance(user_message, str):
            user_message = [user_message]
        payload: dict = {
            "messages": [
                {"role": "user", "content": [{"text": k}]} for k in user_message
            ],
        }
        payload.update(kwargs)
        if payload.get("inferenceConfig") is None:
            payload["inferenceConfig"] = {}

        payload["inferenceConfig"] = {
            **payload["inferenceConfig"],
            "maxTokens": max_tokens,
        }
        return payload


class BedrockConverse(BedrockBase):
    def _parse_converse_response(self, response: Dict) -> InvocationResponse:
        output_text = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        # metrics = response.get("metrics", {})

        return InvocationResponse(
            id=uuid4().hex,
            response_text=output_text,
            num_tokens_input=usage.get("inputTokens"),
            num_tokens_output=usage.get("outputTokens"),
            # time_to_last_token=metrics.get("latencyMs") / 1e3,
        )

    def invoke(self, payload: Dict, **kwargs) -> InvocationResponse:
        payload = {**kwargs, **payload}
        if payload.get("inferenceConfig") is None:
            payload["inferenceConfig"] = self._inference_config or {}

        payload["modelId"] = self.model_id
        try:
            start_t = time.perf_counter()
            client_response = self._bedrock_client.converse(**payload)
            time_to_last_token = time.perf_counter() - start_t
        except (ClientError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(id=uuid4().hex, error=str(e))
        response = self._parse_converse_response(client_response)  # type: ignore
        response.input_prompt = self._parse_payload(payload)
        response.time_to_last_token = time_to_last_token
        return response


class BedrockConverseStream(BedrockConverse):
    def invoke(self, payload: Dict, **kwargs) -> InvocationResponse:
        payload = {**kwargs, **payload}
        if payload.get("inferenceConfig") is None:
            payload["inferenceConfig"] = self._inference_config or {}

        payload["modelId"] = self.model_id
        start_t = time.perf_counter()
        try:
            client_response = self._bedrock_client.converse_stream(**payload)
        except (ClientError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(id=uuid4().hex, error=str(e))
        response = self._parse_conversation_stream(client_response, start_t)  # type: ignore
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_conversation_stream(
        self, client_response: Dict, start_t: float
    ) -> InvocationResponse:
        time_flag = True
        time_to_first_token = None
        output_text = ""
        for chunk in client_response["stream"]:
            if "contentBlockDelta" in chunk:
                output_text += chunk["contentBlockDelta"]["delta"].get("text") or ""
                if time_flag:
                    time_to_first_token = time.perf_counter() - start_t
                    time_flag = False

            if "contentBlockStop" in chunk:
                time_to_last_token = time.perf_counter() - start_t

            if "metadata" in chunk:
                metadata = chunk["metadata"]

        response = InvocationResponse(
            id=uuid4().hex,
            response_text=output_text,
            time_to_last_token=time_to_last_token,
            time_to_first_token=time_to_first_token,
        )

        if metadata:
            # time_to_last_token = metadata.get("metrics", {}).get("latencyMs")
            usage = metadata.get("usage", {})
            response.num_tokens_input = usage.get("inputTokens")
            response.num_tokens_output = usage.get("outputTokens")
            if (
                response.num_tokens_output
                and time_to_last_token
                and time_to_first_token
            ):
                generation_time = time_to_last_token - time_to_first_token
                response.time_per_output_token = (response.num_tokens_output - 1) and (
                    generation_time / (response.num_tokens_output - 1)
                )

        return response
