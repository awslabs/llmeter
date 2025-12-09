# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import jmespath

from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)


class BedrockInvoke(Endpoint):
    """LLMeter Endpoint for Amazon Bedrock InvokeModel API (non-streaming)"""

    def __init__(
        self,
        model_id: str,
        endpoint_name: str | None = None,
        region: str | None = None,
        bedrock_boto3_client=None,
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
        user_message: str | list[str], max_tokens: int | None = 256, **kwargs
    ):
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

    def _parse_response(self, response: dict) -> InvocationResponse:
        """
        Parse the response from a Bedrock InvokeModel API call.

        Args:
            response (dict): Raw response from the Bedrock API containing output text and metadata

        Returns:
            InvocationResponse: Parsed response containing the generated text and metadata

        Raises:
            KeyError: If required fields are missing from the response
            TypeError: If response fields have unexpected types
        """
        try:
            response_body_json = response["body"].read().decode("utf-8")
            response_body = json.loads(response_body_json)
        except json.JSONDecodeError as e:
            logger.exception(
                "Error parsing response as JSON. Accept header='%s'",
                response.get("accept"),
            )
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Failed to parse endpoint response as JSON: {e}",
            )

        try:
            response_text = jmespath.search(self.generated_text_jmespath, response_body)
            if isinstance(response_text, list):
                response_text = "\n".join(response_text)

            id = response_body.get("id", uuid4().hex)
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

        except KeyError as e:
            logger.exception(f"Missing required field in response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Missing required field: {e}",
            )

        except TypeError as e:
            logger.exception(f"Unexpected type in response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Type error in response: {e}",
            )

        except Exception as e:
            logger.exception(f"Error parsing InvokeModel response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Response parsing error: {e}",
            )

    def invoke(self, payload: dict) -> InvocationResponse:
        """Invoke the Bedrock InvokeModel API with the given payload.

        Args:
            payload (dict): The payload containing the request parameters

        Returns:
            InvocationResponse: Response object containing generated text and metadata

        Raises:
            ClientError: If there is an error calling the Bedrock API
            ValueError: If payload is invalid
            TypeError: If payload is not a dictionary
        """
        if not isinstance(payload, dict):
            raise TypeError("Payload must be a dictionary")

        try:
            req_body = json.dumps(payload).encode("utf-8")
            try:
                start_t = time.perf_counter()
                client_response = self._bedrock_client.invoke_model(  # type: ignore
                    accept="application/json",
                    body=req_body,
                    contentType="application/json",
                    modelId=self.model_id,
                    # TODO: Provide config for other optional arguments
                    # trace, guardrailIdentifier/Version, performanceConfigLatency, serviceTier
                )
                time_to_last_token = time.perf_counter() - start_t
            except ClientError as e:
                logger.error(f"Bedrock API error: {e}")
                return InvocationResponse.error_output(
                    input_payload=payload, id=uuid4().hex, error=str(e)
                )
            except Exception as e:
                logger.error(f"Unexpected error during API call: {e}")
                return InvocationResponse.error_output(
                    input_payload=payload, id=uuid4().hex, error=str(e)
                )

            response = self._parse_response(client_response)  # type: ignore
            response.input_payload = payload
            response.input_prompt = self._parse_payload(payload)
            response.time_to_last_token = time_to_last_token
            return response

        except Exception as e:
            logger.error(f"Error in invoke method: {e}")
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )


class BedrockInvokeStream(BedrockInvoke):
    """LLMeter Endpoint for Amazon Bedrock InvokeModelWithResponseStream API"""

    def __init__(
        self,
        model_id: str,
        endpoint_name: str | None = None,
        region: str | None = None,
        bedrock_boto3_client=None,
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
        try:
            start_t = time.perf_counter()
            client_response = self._bedrock_client.invoke_model_with_response_stream(  # type: ignore
                accept="application/json",
                body=req_body,
                contentType="application/json",
                modelId=self.model_id,
                # TODO: Provide config for other optional arguments
                # trace, guardrailIdentifier/Version, performanceConfigLatency, serviceTier
            )
        except (ClientError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        response = self._parse_response_stream(client_response, start_t)
        response.input_payload = payload
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_response_stream(
        self, client_response, start_t: float
    ) -> InvocationResponse:
        """Parse the streaming response from Bedrock InovkeModel API.

        Args:
            client_response (dict): The raw response from the Bedrock API
            start_t (float): The timestamp when the request was initiated

        Returns:
            InvocationResponse: Parsed response containing the generated text and metadata

        Raises:
            KeyError: If required fields are missing from the response
            TypeError: If response fields have unexpected types
        """
        output_text = ""
        chunks = []
        time_to_first_token = None
        time_to_last_token = None

        try:
            for event in client_response["body"]:
                if "chunk" in event:
                    chunk_bytes = event["chunk"]["bytes"]
                    chunk_data = json.loads(chunk_bytes)
                    chunk_text = jmespath.search(
                        self.generated_text_jmespath, chunk_data
                    )
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
                    for etype in (
                        "internalServerException",
                        "modelStreamErrorException",
                        "validationException",
                        "throttlingException",
                        "modelTimeoutException",
                        "serviceUnavailableException",
                    ):
                        if etype in event:
                            msg = f"Bedrock {etype}: {event[etype]['message']}"
                            logger.error(msg)
                            return InvocationResponse.error_output(
                                id=uuid4().hex,
                                error=msg,
                            )
                    # Unknown event type - probably an error
                    return InvocationResponse.error_output(
                        id=uuid4().hex,
                        error=f"Unknown event type in response stream: {event}",
                    )

            # Post-process additional data from chunks:
            # (Which we do after performance timing, to avoid counting JMESPath overhead)
            num_tokens_input = None
            num_tokens_output = None
            resp_id = None
            for chunk in chunks:
                if "id" in chunk:
                    resp_id = chunk["id"]
                # Usage counts should only appear once in the stream. We'll overwrite if duplicated:
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

            response = InvocationResponse(
                id=resp_id or uuid4().hex,
                response_text=output_text,
                time_to_first_token=time_to_first_token,
                time_to_last_token=time_to_last_token,
                num_tokens_input=num_tokens_input,
                num_tokens_output=num_tokens_output,
                retries=client_response.get("ResponseMetadata", {}).get(
                    "RetryAttempts"
                ),
            )
            return response

        except KeyError as e:
            logger.error(f"Missing required field in response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Missing required field: {e}",
            )
        except TypeError as e:
            logger.error(f"Unexpected type in response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Type error in response: {e}",
            )
        except Exception as e:
            logger.error(f"Error parsing response stream: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Stream parsing error: {e}",
            )
