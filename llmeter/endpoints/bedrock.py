# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)


class BedrockBase(Endpoint):
    """Base class for interacting with Amazon Bedrock endpoints.

    This class provides core functionality for making requests to Amazon Bedrock
    endpoints, handling configuration and client initialization.

    Args:
        model_id (str): The identifier for the model to use
        endpoint_name (str | None, optional): Name of the endpoint. Defaults to None.
        region (str | None, optional): AWS region to use. Defaults to None.
        inference_config (dict | None, optional): Configuration for inference. Defaults to None.
        bedrock_boto3_client (boto3.client, optional): Pre-configured boto3 client. Defaults to None.
        max_attempts (int, optional): Maximum number of retry attempts. Defaults to 3.
    """

    def __init__(
        self,
        model_id: str,
        endpoint_name: str | None = None,
        region: str | None = None,
        inference_config: dict | None = None,
        bedrock_boto3_client=None,
        max_attempts: int = 3,
    ):
        super().__init__(
            model_id=model_id, endpoint_name=endpoint_name or "", provider="bedrock"
        )

        self.endpoint_name = "amazon bedrock"

        self.region = region or boto3.session.Session().region_name
        logger.info(f"Using AWS region: {self.region}")

        self._bedrock_client = bedrock_boto3_client
        if self._bedrock_client is None:
            config = Config(retries={"max_attempts": max_attempts, "mode": "standard"})
            self._bedrock_client = boto3.client(
                "bedrock-runtime", region_name=self.region, config=config
            )
        self._inference_config = inference_config

    def _parse_payload(self, payload):
        """
        Parse the payload to extract text content.

        Args:
            payload (dict): The payload containing messages.

        Returns:
            str: Concatenated text content from the messages.

        Raises:
            TypeError: If payload is not a dictionary
            KeyError: If required fields are missing from payload
        """
        try:
            if not isinstance(payload, dict):
                raise TypeError("Payload must be a dictionary")

            messages = payload.get("messages", [])
            if not isinstance(messages, list):
                raise TypeError("Messages must be a list")

            texts = []
            for msg in messages:
                if not isinstance(msg, dict):
                    raise TypeError("Each message must be a dictionary")

                for content in msg.get("content", []):
                    if not isinstance(content, dict):
                        raise TypeError("Content must be a dictionary")

                    text_content = content.get("text")
                    if isinstance(text_content, list):
                        texts.extend(text_content)
                    else:
                        texts.append(text_content or "")

            return "\n".join(filter(None, texts))

        except (TypeError, KeyError) as e:
            logger.error(f"Error parsing payload: {e}")
            return ""

        except Exception as e:
            logger.error(f"Unexpected error parsing payload: {e}")
            return ""

    @staticmethod
    def create_payload(user_message: str | list[str], max_tokens: int = 256, **kwargs):
        """
        Create a payload for the Bedrock Converse API request.

        Args:
            user_message (str | Sequence[str]): The user's message or a sequence of messages.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 256.
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

        except Exception as e:
            logger.error(f"Error creating payload: {e}")
            raise RuntimeError(f"Failed to create payload: {str(e)}")


class BedrockConverse(BedrockBase):
    def _parse_converse_response(self, response: dict) -> InvocationResponse:
        """
        Parse the response from a Bedrock converse API call.

        Args:
            response (dict): Raw response from the Bedrock API containing output text and metadata

        Returns:
            InvocationResponse: Parsed response containing the generated text and metadata

        Raises:
            KeyError: If required fields are missing from the response
            TypeError: If response fields have unexpected types
        """
        try:
            # Direct dictionary access and single-level assignment for better performance
            output = response["output"]["message"]["content"][0]["text"]
            if not isinstance(output, str):
                raise TypeError("Expected string for output text")

            usage = response.get("usage", {})
            retries = response["ResponseMetadata"]["RetryAttempts"]

            return InvocationResponse(
                id=uuid4().hex,
                response_text=output,
                num_tokens_input=usage.get("inputTokens"),
                num_tokens_output=usage.get("outputTokens"),
                retries=retries,
            )

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
            logger.error(f"Error parsing converse response: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Response parsing error: {e}",
            )

    def invoke(self, payload: dict, **kwargs) -> InvocationResponse:
        """
        Invoke the Bedrock converse API with the given payload.

        Args:
            payload (dict): The payload containing the request parameters
            **kwargs: Additional keyword arguments to include in the payload

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
            payload = {**kwargs, **payload}
            if payload.get("inferenceConfig") is None:
                payload["inferenceConfig"] = self._inference_config or {}

            payload["modelId"] = self.model_id
            try:
                start_t = time.perf_counter()
                client_response = self._bedrock_client.converse(**payload)  # type: ignore
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

            response = self._parse_converse_response(client_response)  # type: ignore
            response.input_payload = payload
            response.input_prompt = self._parse_payload(payload)
            response.time_to_last_token = time_to_last_token
            return response

        except Exception as e:
            logger.error(f"Error in invoke method: {e}")
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )


class BedrockConverseStream(BedrockConverse):
    def invoke(self, payload: dict, **kwargs) -> InvocationResponse:
        payload = {**kwargs, **payload}
        if payload.get("inferenceConfig") is None:
            payload["inferenceConfig"] = self._inference_config or {}

        payload["modelId"] = self.model_id
        start_t = time.perf_counter()
        try:
            client_response = self._bedrock_client.converse_stream(**payload)  # type: ignore
        except (ClientError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(
                input_payload=payload, id=uuid4().hex, error=str(e)
            )
        response = self._parse_conversation_stream(client_response, start_t)
        response.input_payload = payload
        response.input_prompt = self._parse_payload(payload)
        return response

    def _parse_conversation_stream(
        self, client_response, start_t: float
    ) -> InvocationResponse:
        """
        Parse the streaming response from Bedrock conversation API.

        Args:
            client_response (dict): The raw response from the Bedrock API
            start_t (float): The timestamp when the request was initiated

        Returns:
            InvocationResponse: Parsed response containing the generated text and metadata

        Raises:
            KeyError: If required fields are missing from the response
            TypeError: If response fields have unexpected types
        """
        time_flag = True
        time_to_first_token = None
        time_to_last_token = None
        output_text = ""
        metadata = None

        try:
            for chunk in client_response["stream"]:
                if "contentBlockDelta" in chunk:
                    delta_text = chunk["contentBlockDelta"]["delta"].get("text", "")
                    if not isinstance(delta_text, str):
                        raise TypeError("Expected string for delta text")
                    output_text += delta_text or ""
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
                # The latency provided by Bedrock is at the service endpoint time, not client side
                # time_to_last_token = metadata.get("metrics", {}).get("latencyMs")
                try:
                    usage = metadata.get("usage", {})
                    response.num_tokens_input = usage.get("inputTokens")
                    response.num_tokens_output = usage.get("outputTokens")
                except Exception as e:
                    logger.error(f"Error parsing metadata: {e}")
                    return InvocationResponse.error_output(
                        id=uuid4().hex,
                        error=f"Metadata parsing error: {e}",
                    )

            # compute time per output token if the model supports it
            if (
                response.num_tokens_output
                and response.time_to_last_token
                and response.time_to_first_token
            ):
                response.time_per_output_token = (
                    response.time_to_last_token - response.time_to_first_token
                ) / (response.num_tokens_output - 1)

            response.retries = client_response["ResponseMetadata"]["RetryAttempts"]

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
            logger.error(f"Error parsing conversation stream: {e}")
            return InvocationResponse.error_output(
                id=uuid4().hex,
                error=f"Stream parsing error: {e}",
            )
