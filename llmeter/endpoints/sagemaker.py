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
from ..prompt_utils import (
    AudioContent,
    ContentItem,
    DocumentContent,
    ImageContent,
    MediaContent,
    VideoContent,
)

logger = logging.getLogger(__name__)


def _mime_to_sagemaker_format(mime_type: str) -> str | None:
    """Convert MIME type to SageMaker format string."""
    mime_map = {
        "image/jpeg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "application/pdf": "pdf",
        "video/mp4": "mp4",
        "video/quicktime": "mov",
        "video/x-msvideo": "avi",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/ogg": "ogg",
    }
    return mime_map.get(mime_type)


# Mapping from MediaContent subclass to SageMaker content-block key
_SM_MEDIA_TYPE_KEY: dict[type, str] = {
    ImageContent: "image",
    AudioContent: "audio",
    VideoContent: "video",
    DocumentContent: "document",
}


def _build_content_blocks_sagemaker(items: list[ContentItem]) -> list[dict]:
    """Convert an ordered list of content items to SageMaker content blocks."""
    blocks: list[dict] = []
    for item in items:
        if isinstance(item, str):
            blocks.append({"text": item})
        elif isinstance(item, MediaContent):
            key = _SM_MEDIA_TYPE_KEY.get(type(item))
            if key is None:
                raise TypeError(f"Unsupported content type: {type(item).__name__}")
            fmt = _mime_to_sagemaker_format(item.mime_type)
            if fmt is None:
                raise ValueError(f"Unsupported MIME type '{item.mime_type}' for {key}")
            blocks.append({key: {"format": fmt, "source": {"bytes": item.data}}})
        else:
            raise TypeError(
                f"Content items must be str or MediaContent, got {type(item).__name__}"
            )
    return blocks


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
        input_text: str | list[ContentItem],
        max_tokens: int = 256,
        inference_parameters: dict = {},
        **kwargs,
    ):
        """Create a payload for the SageMaker API request.

        Args:
            input_text: A single text string, or an ordered list mixing strings
                and :class:`~llmeter.prompt_utils.MediaContent` objects.
            max_tokens: Maximum tokens to generate. Defaults to 256.
            inference_parameters: Additional inference parameters.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The formatted payload for the SageMaker API request.
        """
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        if isinstance(input_text, str):
            items: list[ContentItem] = [input_text]
        elif isinstance(input_text, list):
            items = input_text
        else:
            raise TypeError(
                "input_text must be a str or list of str/MediaContent, "
                f"got {type(input_text).__name__}"
            )

        if not items:
            raise ValueError("input_text must not be empty")

        # Text-only shortcut
        if len(items) == 1 and isinstance(items[0], str):
            payload = {
                "inputs": items[0],
                "parameters": {"max_new_tokens": max_tokens, "details": True},
            }
            if inference_parameters:
                payload["parameters"].update(inference_parameters)
            payload.update(kwargs)
            return payload

        content_blocks = _build_content_blocks_sagemaker(items)
        payload = {
            "inputs": content_blocks,
            "parameters": {"max_new_tokens": max_tokens, "details": True},
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
                input_payload=payload,
                id=uuid4().hex,
                error=str(e),
            )

        time_to_last_token = time.perf_counter() - start_t
        parsed_response = self._parse_client_response(client_response)
        if parsed_response:
            response_text = parsed_response.get("output_text", "")
            num_tokens_output = parsed_response.get("num_tokens_output", None)

        return InvocationResponse(
            input_payload=payload,
            id=uuid4().hex,
            response_text=response_text,
            time_to_last_token=time_to_last_token,
            input_prompt=input_prompt,
            num_tokens_output=num_tokens_output if num_tokens_output else None,
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
            return InvocationResponse.error_output(input_payload=payload, error=str(e))

        try:
            response = self._parse_client_response(client_response, start_t)
            response.input_payload = payload
            response.input_prompt = input_prompt
            return response
        except Exception as e:
            return InvocationResponse.error_output(input_payload=payload, error=str(e))

    @staticmethod
    def create_payload(
        input_text: str | list[ContentItem],
        max_tokens: int = 256,
        inference_parameters: dict = {},
        **kwargs,
    ):
        """Create a payload for the SageMaker streaming API request.

        Args:
            input_text: A single text string, or an ordered list mixing strings
                and :class:`~llmeter.prompt_utils.MediaContent` objects.
            max_tokens: Maximum tokens to generate. Defaults to 256.
            inference_parameters: Additional inference parameters.
            **kwargs: Additional keyword arguments to include in the payload.

        Returns:
            dict: The formatted payload for the SageMaker streaming API request.
        """
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        if isinstance(input_text, str):
            items: list[ContentItem] = [input_text]
        elif isinstance(input_text, list):
            items = input_text
        else:
            raise TypeError(
                "input_text must be a str or list of str/MediaContent, "
                f"got {type(input_text).__name__}"
            )

        if not items:
            raise ValueError("input_text must not be empty")

        # Text-only shortcut
        if len(items) == 1 and isinstance(items[0], str):
            payload = {
                "inputs": items[0],
                "parameters": {"max_new_tokens": max_tokens, "details": True},
                "stream": True,
            }
            if inference_parameters:
                payload["parameters"].update(inference_parameters)
            payload.update(kwargs)
            return payload

        content_blocks = _build_content_blocks_sagemaker(items)
        payload = {
            "inputs": content_blocks,
            "parameters": {"max_new_tokens": max_tokens, "details": True},
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
                    raise RuntimeError(line_data["error"])
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
