# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from typing import Any

import boto3
import jmespath
from botocore.config import Config

from .base import Endpoint, InvocationResponse
from .bedrock import BEDROCK_STREAM_ERROR_TYPES

logger = logging.getLogger(__name__)


class BedrockInvoke(Endpoint):
    """LLMeter Endpoint for Amazon Bedrock InvokeModel API (non-streaming)"""

    def __init__(
        self,
        model_id: str,
        endpoint_name: str | None = None,
        region: str | None = None,
        bedrock_boto3_client: Any = None,
        max_attempts: int = 3,
        generated_text_jmespath: str = "choices[0].message.content",
        generated_token_count_jmespath: str | None = "usage.completion_tokens",
        input_text_jmespath: str = "messages[].content[].text",
        input_token_count_jmespath: str | None = "usage.prompt_tokens",
    ):
        """Create a BedrockInvoke Endpoint

        The default ..._jmespath parameters assume your target model uses an OpenAI
        ChatCompletions-like API, which is true for many (but not all) Bedrock models. You'll need
        to override these if targeting a model with different request/response format.

        Args:
            model_id:
                The identifier for the model to use
            endpoint_name:
                Name of the endpoint. Defaults to None.
            region:
                AWS region to use. Defaults to bedrock_boto3_client's, or configured from AWS CLI.
            bedrock_boto3_client:
                Optional pre-configured boto3 client, otherwise one will be created.
            max_attempts:
                Maximum number of retry attempts. Defaults to 3.
            generated_text_jmespath:
                JMESPath query to extract generated text from model response.
            generated_token_count_jmespath:
                JMESPath query to extract generated token count from model response.
            input_text_jmespath:
                JMESPath query to extract input text from the model request payload.
            input_token_count_jmespath:
                JMESPath query to extract input token count from the response.
        """
        super().__init__(
            model_id=model_id,
            endpoint_name=endpoint_name or "amazon bedrock",
            provider="bedrock",
        )

        self.generated_text_jmespath = generated_text_jmespath
        self.generated_token_count_jmespath = generated_token_count_jmespath
        self.input_text_jmespath = input_text_jmespath
        self.input_token_count_jmespath = input_token_count_jmespath

        self.region = (
            region
            or (bedrock_boto3_client and bedrock_boto3_client.meta.region_name)
            or boto3.session.Session().region_name
        )
        logger.info(f"Using AWS region: {self.region}")

        self._bedrock_client = bedrock_boto3_client
        if self._bedrock_client is None:
            config = Config(retries={"max_attempts": max_attempts, "mode": "standard"})
            self._bedrock_client = boto3.client(
                "bedrock-runtime", region_name=self.region, config=config
            )

    def _parse_payload(self, payload: dict) -> str:
        """Parse the payload to extract text content.

        Args:
            payload: The payload containing messages.

        Returns:
            str: Concatenated text content from the messages.

        Raises:
            TypeError: If payload is not a dictionary
            KeyError: If required fields are missing from payload
        """
        try:
            query_results = jmespath.search(self.input_text_jmespath, payload)
        except Exception:
            logger.exception(
                "Failed to query path '%s' from payload '%s'",
                self.input_text_jmespath,
                payload,
            )
            return ""

        if isinstance(query_results, str):
            return query_results
        elif isinstance(query_results, list):
            return "\n".join(query_results)
        else:
            raise TypeError(
                "Failed to extract input text from payload. JMESPath query result should be a "
                "string or list of strings. Got %s from payload %s"
                % (query_results, payload)
            )

    @staticmethod
    def create_payload(
        user_message: str | list[str], max_tokens: int | None = 256, **kwargs: Any
    ) -> dict:
        """
        Create a payload, assuming your target Bedrock model supports ChatCompletions-like API

        Args:
            user_message: The user's message or a sequence of messages.
            max_tokens: The maximum number of tokens to generate. Defaults to 256.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The formatted payload for the Bedrock API request.

        Raises:
            TypeError: If user_message is not a string or list of strings
            ValueError: If max_tokens is not a positive integer
        """
        if not isinstance(user_message, (str, list)):
            raise TypeError("user_message must be a string or list of strings")

        if isinstance(user_message, list):
            if not all(isinstance(msg, str) for msg in user_message):
                raise TypeError("All messages must be strings")
            if not user_message:
                raise ValueError("user_message list cannot be empty")

        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        if isinstance(user_message, str):
            user_message = [user_message]

        try:
            payload: dict = {
                "messages": [
                    {"role": "user", "content": [{"text": k, "type": "text"}]}
                    for k in user_message
                ],
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            payload.update(kwargs)
            return payload

        except Exception as e:
            logger.exception("Failed to create InvokeModel payload")
            raise RuntimeError(f"Failed to create payload: {str(e)}") from e

    def parse_response(self, response: dict, start_t: float) -> InvocationResponse:
        """Parse the response from a Bedrock InvokeModel API call.

        Args:
            response: Raw response from the Bedrock API.
            start_t: The timestamp when the request was initiated.

        Returns:
            InvocationResponse with the generated text and metadata.
        """
        response_body_json = response["body"].read().decode("utf-8")
        response_body = json.loads(response_body_json)

        response_text = jmespath.search(self.generated_text_jmespath, response_body)
        if isinstance(response_text, list):
            response_text = "\n".join(response_text)

        id = response_body.get("id") or response.get("ResponseMetadata", {}).get(
            "RequestId"
        )
        retries = response.get("ResponseMetadata", {}).get("RetryAttempts")
        num_tokens_input = (
            jmespath.search(self.input_token_count_jmespath, response_body)
            if self.input_token_count_jmespath
            else None
        )
        num_tokens_output = (
            jmespath.search(self.generated_token_count_jmespath, response_body)
            if self.generated_token_count_jmespath
            else None
        )

        return InvocationResponse(
            id=id,
            response_text=response_text,
            num_tokens_input=num_tokens_input,
            num_tokens_output=num_tokens_output,
            retries=retries,
        )

    def invoke(self, payload: dict) -> InvocationResponse:
        """Invoke the Bedrock InvokeModel API with the given payload."""
        req_body = json.dumps(payload).encode("utf-8")

        client_response = self._bedrock_client.invoke_model(  # type: ignore
            accept="application/json",
            body=req_body,
            contentType="application/json",
            modelId=self.model_id,
            # TODO: Provide config for other optional arguments
            # trace, guardrailIdentifier/Version, performanceConfigLatency, serviceTier
        )
        return self.parse_response(client_response, self._start_t)  # type: ignore


class BedrockInvokeStream(BedrockInvoke):
    """LLMeter Endpoint for Amazon Bedrock InvokeModelWithResponseStream API"""

    def __init__(
        self,
        model_id: str,
        endpoint_name: str | None = None,
        region: str | None = None,
        bedrock_boto3_client: Any = None,
        max_attempts: int = 3,
        generated_text_jmespath: str = "choices[0].delta.content",
        generated_token_count_jmespath: str
        | None = '"amazon-bedrock-invocationMetrics".outputTokenCount',
        input_text_jmespath: str = "messages[].content[].text",
        input_token_count_jmespath: str
        | None = '"amazon-bedrock-invocationMetrics".inputTokenCount',
    ):
        """Create a BedrockInvokeStream Endpoint

        The default ..._jmespath parameters assume your target model uses an OpenAI
        ChatCompletions-like streaming API, which is true for many (but not all) Bedrock models.
        You'll need to override these if targeting a model with different request/response format.

        Args:
            model_id:
                The identifier for the model to use
            endpoint_name:
                Name of the endpoint. Defaults to None.
            region:
                AWS region to use. Defaults to bedrock_boto3_client's, or configured from AWS CLI.
            bedrock_boto3_client:
                Optional pre-configured boto3 client, otherwise one will be created.
            max_attempts:
                Maximum number of retry attempts. Defaults to 3.
            generated_text_jmespath:
                JMESPath query to extract incremental text from *a chunk of* the model response.
            generated_token_count_jmespath:
                JMESPath query to extract generated token count from *a chunk of* model response.
            input_text_jmespath:
                JMESPath query to extract input text from the model request payload.
            input_token_count_jmespath:
                JMESPath query to extract input token count from *a chunk of* the model response.
        """
        super().__init__(
            model_id=model_id,
            endpoint_name=endpoint_name,
            region=region,
            bedrock_boto3_client=bedrock_boto3_client,
            max_attempts=max_attempts,
            generated_text_jmespath=generated_text_jmespath,
            generated_token_count_jmespath=generated_token_count_jmespath,
            input_text_jmespath=input_text_jmespath,
            input_token_count_jmespath=input_token_count_jmespath,
        )

    def invoke(self, payload: dict) -> InvocationResponse:
        req_body = json.dumps(payload).encode("utf-8")

        client_response = self._bedrock_client.invoke_model_with_response_stream(  # type: ignore
            accept="application/json",
            body=req_body,
            contentType="application/json",
            modelId=self.model_id,
            # TODO: Provide config for other optional arguments
            # trace, guardrailIdentifier/Version, performanceConfigLatency, serviceTier
        )
        return self.parse_response(client_response, self._start_t)

    def parse_response(self, client_response, start_t: float) -> InvocationResponse:
        """Parse the streaming response from Bedrock InvokeModel API.

        Args:
            client_response: The raw response from the Bedrock API.
            start_t: The timestamp when the request was initiated.

        Returns:
            InvocationResponse with the generated text and metadata.
        """
        output_text = ""
        chunks = []
        time_to_first_token = None
        time_to_last_token = None
        error = None

        for event in client_response["body"]:
            if "chunk" in event:
                chunk_bytes = event["chunk"]["bytes"]
                chunk_data = json.loads(chunk_bytes)
                chunk_text = jmespath.search(self.generated_text_jmespath, chunk_data)
                if isinstance(chunk_text, list):
                    chunk_text = "".join(chunk_text)
                if chunk_text:
                    now = time.perf_counter()
                    if time_to_first_token is None:
                        time_to_first_token = now - start_t
                    time_to_last_token = now - start_t
                    output_text += chunk_text
                chunks.append(chunk_data)
            else:
                # Non-chunk events: check for Bedrock error events, skip
                # everything else (e.g. messageStart, contentBlockStart).
                for error_type in BEDROCK_STREAM_ERROR_TYPES:
                    if error_type in event:
                        error = f"Bedrock {error_type}: {event[error_type]['message']}"
                        logger.error(error)
                        break

        # Post-process additional data from chunks
        # (after performance timing, to avoid counting JMESPath overhead)
        num_tokens_input = None
        num_tokens_output = None
        resp_id = None
        for chunk in chunks:
            if "id" in chunk:
                resp_id = chunk["id"]
            chk_tokens_input = (
                jmespath.search(self.input_token_count_jmespath, chunk)
                if self.input_token_count_jmespath
                else None
            )
            if chk_tokens_input is not None:
                num_tokens_input = chk_tokens_input
            chk_tokens_output = (
                jmespath.search(self.generated_token_count_jmespath, chunk)
                if self.generated_token_count_jmespath
                else None
            )
            if chk_tokens_output is not None:
                num_tokens_output = chk_tokens_output

        return InvocationResponse(
            id=resp_id or client_response.get("ResponseMetadata", {}).get("RequestId"),
            response_text=output_text,
            time_to_first_token=time_to_first_token,
            time_to_last_token=time_to_last_token,
            num_tokens_input=num_tokens_input,
            num_tokens_output=num_tokens_output,
            error=error,
            retries=client_response.get("ResponseMetadata", {}).get("RetryAttempts"),
        )
