# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import time
from typing import Generic, TypeVar
import warnings

import boto3
import jmespath

try:
    from mypy_boto3_sagemaker_runtime.type_defs import (
        InvokeEndpointOutputTypeDef,
        InvokeEndpointWithResponseStreamOutputTypeDef,
    )
except ImportError:
    InvokeEndpointOutputTypeDef = TypeVar("InvokeEndpointOutputTypeDef")
    InvokeEndpointWithResponseStreamOutputTypeDef = TypeVar(
        "InvokeEndpointWithResponseStreamOutputTypeDef"
    )


from ..prompt_utils import (
    AudioContent,
    ContentItem,
    DocumentContent,
    ImageContent,
    MediaContent,
    VideoContent,
)
from .base import Endpoint, InvocationResponse

logger = logging.getLogger(__name__)


TSageMakerResponseBase = TypeVar(
    "TSageMakerResponseBase",
    bound=InvokeEndpointOutputTypeDef | InvokeEndpointWithResponseStreamOutputTypeDef,
)


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


class SageMakerBase(Endpoint[TSageMakerResponseBase], Generic[TSageMakerResponseBase]):
    def __init__(
        self,
        endpoint_name: str,
        model_id: str,
        # TODO: generated & token count jmespaths not actually used by streaming yet
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


class SageMakerEndpoint(SageMakerBase[InvokeEndpointOutputTypeDef]):
    """
    A class for handling invocations to a SageMaker endpoint.

    This class extends SageMakerBase to provide functionality for invoking
    a SageMaker endpoint and parsing its response.
    """

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

    @SageMakerBase.llmeter_invoke
    def invoke(self, payload: dict) -> InvokeEndpointOutputTypeDef:
        """Invoke the SageMaker endpoint with the given payload."""
        json_payload = json.dumps(payload)

        client_response = self._sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=bytes(json_payload, "utf-8"),
        )
        return client_response

    def process_raw_response(
        self,
        raw_response: InvokeEndpointOutputTypeDef,
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        response.time_to_last_token = time.perf_counter()

        if raw_response is None:
            response.error = "Null response from SageMaker endpoint"
            return

        raw_response_body_json = raw_response["Body"].read().decode("utf-8")
        raw_response_body = json.loads(raw_response_body_json)
        raw_response_meta = raw_response_body.get("ResponseMetadata", {})

        response.id = raw_response_meta.get("RequestId")
        response.retries = raw_response_meta.get("RetryAttempts")
        response.response_text = jmespath.search(
            self.generated_text_jmespath, raw_response_body
        )
        if self.token_count_jmespath:
            response.num_tokens_output = jmespath.search(
                self.token_count_jmespath, raw_response_body
            )


class SageMakerStreamEndpoint(
    SageMakerBase[InvokeEndpointWithResponseStreamOutputTypeDef]
):
    """
    A class for handling streaming invocations to a SageMaker endpoint.

    This class extends SageMakerBase to provide functionality specific to
    streaming responses from a SageMaker endpoint.
    """

    @SageMakerBase.llmeter_invoke
    def invoke(self, payload: dict) -> InvokeEndpointWithResponseStreamOutputTypeDef:
        """Invoke a SageMaker streaming endpoint with the given payload."""
        json_payload = json.dumps(payload)

        client_response = self._sagemaker_runtime.invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name,
            Body=json_payload,
            ContentType="application/json",
        )
        return client_response

    def prepare_payload(self, payload):
        if "parameters" in payload:
            payload["parameters"].pop("decoder_input_details", None)
        if "stream" not in payload:
            warnings.warn("stream not specified in payload, defaulting to True")
            payload["stream"] = True
        return payload

    def process_raw_response(
        self,
        raw_response: InvokeEndpointWithResponseStreamOutputTypeDef,
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        response_meta = raw_response.get("ResponseMetadata", {})
        response.id = response_meta.get("RequestId")
        response.retries = response_meta.get("RetryAttempts")

        token_iterator = TokenIterator(raw_response["Body"])
        first_token = next(token_iterator)
        response.time_to_first_token = time.perf_counter() - start_t

        response_text_tokens = [first_token] + [k for k in token_iterator]
        response.time_to_last_token = time.perf_counter() - start_t
        response.response_text = "".join(response_text_tokens)

        if token_iterator.details:
            response.num_tokens_output = token_iterator.details.get(
                "generated_tokens", 0
            )
        if response.num_tokens_output is None:
            response.num_tokens_output = len(response_text_tokens)

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
