# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import time
import warnings
from uuid import uuid4

import boto3
import jmespath
from botocore.exceptions import ClientError

from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)


class SageMakerBase(Endpoint):
    def __init__(
        self,
        endpoint_name: str,
        model_id: str,
        generated_text_jmespath: str = "generated_text",
        input_text_jmespath: str = "inputs",
        token_count_jmespath: str | None = "details.generated_tokens",
        region: str | None = None,
        boto3_session: boto3.Session | None = None,
        **kwargs,
    ):
        super().__init__(
            endpoint_name=endpoint_name, model_id=model_id, provider="sagemaker"
        )
        self.generated_text_jmespath = generated_text_jmespath
        self.input_text_jmespath = input_text_jmespath
        self.token_count_jmespath = token_count_jmespath
        self.kwargs = kwargs

        # Get the current AWS region if not provided
        _session = boto3_session or boto3.session.Session()
        self.region = region or _session.region_name
        logger.info(f"Using AWS region: {self.region}")

        self._sagemaker_runtime = _session.client(
            "sagemaker-runtime", region_name=self.region
        )

    def _parse_input(self, payload: dict) -> str | None:
        try:
            return jmespath.search(self.input_text_jmespath, payload)
        except Exception:
            return None

    @staticmethod
    def create_payload(
        input_text: str | list[str],
        max_tokens: int = 256,
        inference_parameters: dict = {},
        **kwargs,
    ):
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": max_tokens,
                "details": True,
            },
        }
        if inference_parameters:
            payload["parameters"].update(inference_parameters)
        payload.update(kwargs)
        return payload


class SageMakerEndpoint(SageMakerBase):
    """
    A class for handling invocations to a SageMaker endpoint.

    This class extends SageMakerBase to provide functionality for invoking
    a SageMaker endpoint and parsing its response.
    """

    def _parse_client_response(self, client_response: dict | None) -> dict | None:
        """
        Parse the response from the SageMaker endpoint.

        This method processes the raw response from the SageMaker endpoint,
        extracting the generated text and token count if available.

        Args:
            client_response (Dict | None): The raw response from the SageMaker endpoint.

        Returns:
            Dict | None: A dictionary containing the parsed response, or None if the input is None.
        """

        if client_response is None:
            return None
        client_response_body_json = client_response["Body"].read().decode("utf-8")
        client_response_body = json.loads(client_response_body_json)
        ret_dict = {
            "output_text": jmespath.search(
                self.generated_text_jmespath, client_response_body
            )
        }
        if self.token_count_jmespath:
            ret_dict["num_tokens_output"] = jmespath.search(
                self.token_count_jmespath, client_response_body
            )
        return ret_dict

    def invoke(self, payload: dict) -> InvocationResponse:
        """
        Invoke the SageMaker endpoint with the given payload.

        This method sends a request to the SageMaker endpoint, processes the response,
        and returns an InvocationResponse object with the results.

        Args:
            payload (Dict): The input payload for the model.

        Returns:
            InvocationResponse: An object containing the model's response and associated metrics.

        Raises:
            ClientError: If there's an error during the invocation of the SageMaker endpoint.
            Exception: If there's any other error during the invocation or parsing of the response.
        """

        json_payload = json.dumps(payload)
        input_prompt = self._parse_input(payload)

        start_t = time.perf_counter()
        try:
            client_response = self._sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=bytes(json_payload, "utf-8"),
            )
        except (ClientError, Exception) as e:
            logger.error(e)
            return InvocationResponse.error_output(
                id=uuid4().hex, error=str(e), input_prompt=input_prompt
            )

        time_to_last_token = time.perf_counter() - start_t
        parsed_response = self._parse_client_response(client_response)
        if parsed_response:
            response_text = parsed_response.get("output_text", "")
            num_tokens_output = parsed_response.get("num_tokens_output", None)

        return InvocationResponse(
            id=uuid4().hex,
            response_text=response_text,
            time_to_last_token=time_to_last_token,
            input_prompt=input_prompt,
            num_tokens_output=num_tokens_output if num_tokens_output else None,
            time_per_output_token=(
                (time_to_last_token / (num_tokens_output - 1))
                if num_tokens_output
                else None
            ),
        )


class SageMakerStreamEndpoint(SageMakerBase):
    """
    A class for handling streaming invocations to a SageMaker endpoint.

    This class extends SageMakerBase to provide functionality specific to
    streaming responses from a SageMaker endpoint.
    """

    def _parse_client_response(
        self, client_response, start_t: float
    ) -> InvocationResponse:
        """
        Parse the streaming response from the SageMaker endpoint.

        This method processes the streaming response, extracting tokens and
        calculating various timing metrics.

        Args:
            client_response (dict): The raw response from the SageMaker endpoint.
            start_t (float): The timestamp when the invocation started.

        Returns:
            InvocationResponse: An object containing the parsed response and metrics.
        """
        token_iterator = TokenIterator(client_response["Body"])
        first_token = next(token_iterator)
        time_to_first_token = time.perf_counter() - start_t

        response_text_tokens = [first_token] + [k for k in token_iterator]
        time_to_last_token = time.perf_counter() - start_t
        num_tokens_output = len(response_text_tokens)

        return InvocationResponse(
            id=uuid4().hex,
            response_text="".join(response_text_tokens),
            time_to_first_token=time_to_first_token,
            time_to_last_token=time_to_last_token,
            num_tokens_output=num_tokens_output,
            time_per_output_token=(time_to_last_token - time_to_first_token)
            / (num_tokens_output - 1),
        )

    def invoke(self, payload: dict) -> InvocationResponse:
        """
        Invoke a SageMaker endpoint with the given payload.

        This method sends a request to the SageMaker endpoint and handles
        the streaming response.

        Args:
            payload (Dict): The input payload for the model.

        Returns:
            InvocationResponse: An object containing the model's response and metrics.

        Raises:
            Exception: If there's an error during the invocation or parsing of the response.
        """

        _payload = payload
        if "parameters" in _payload:
            _payload["parameters"].pop("decoder_input_details", None)
        if "stream" not in _payload:
            warnings.warn("stream not specified in payload, defaulting to True")
            _payload["stream"] = True

        json_payload = json.dumps(_payload)
        input_prompt = self._parse_input(_payload)

        start_t = time.perf_counter()
        try:
            client_response = (
                self._sagemaker_runtime.invoke_endpoint_with_response_stream(
                    EndpointName=self.endpoint_name,
                    Body=json_payload,
                    ContentType="application/json",
                )
            )
        except Exception as e:
            logger.error(e)
            return InvocationResponse.error_output(
                error=str(e), input_prompt=input_prompt
            )

        try:
            response = self._parse_client_response(client_response, start_t)
            response.input_prompt = input_prompt
            return response
        except Exception as e:
            return InvocationResponse.error_output(
                error=str(e), input_prompt=input_prompt
            )

    @staticmethod
    def create_payload(
        input_text: str | list[str],
        max_tokens: int = 256,
        inference_parameters: dict = {},
        **kwargs,
    ):
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": max_tokens,
                "details": True,
            },
            "stream": True,
        }
        if inference_parameters:
            payload["parameters"].update(inference_parameters)
        payload.update(kwargs)
        return payload


class TokenIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0
        self.details = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                full_line = line[:-1].decode("utf-8")
                try:
                    line_data = json.loads(full_line.lstrip("data:").rstrip("/n"))
                except json.JSONDecodeError:
                    continue
                if line_data.get("error"):
                    raise Exception(line_data["error"])
                    break
                self.details = line_data.get("details")
                return line_data["token"]["text"]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])
