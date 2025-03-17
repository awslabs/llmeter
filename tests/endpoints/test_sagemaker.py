# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict
from unittest.mock import patch
from uuid import UUID

import pytest
import requests
from moto import mock_aws

from llmeter.endpoints.sagemaker import (
    InvocationResponse,
    SageMakerBase,
    SageMakerEndpoint,
    SageMakerStreamEndpoint,
    TokenIterator,
)


class ConcreteClass(SageMakerBase):
    def invoke(self, payload: Dict) -> InvocationResponse:
        return InvocationResponse(response_text="test response")


@pytest.fixture
def sagemaker_base():
    with mock_aws():
        return ConcreteClass(endpoint_name="test-endpoint", model_id="test-model")


@pytest.fixture
def sagemaker_endpoint():
    with mock_aws():
        return SageMakerEndpoint(endpoint_name="test-endpoint", model_id="test-model")


@pytest.fixture
def sagemaker_stream_endpoint():
    with mock_aws():
        return SageMakerStreamEndpoint(
            endpoint_name="test-endpoint", model_id="test-model"
        )


def test_sagemaker_base_init(sagemaker_base: ConcreteClass):
    assert sagemaker_base.endpoint_name == "test-endpoint"
    assert sagemaker_base.model_id == "test-model"
    assert sagemaker_base.provider == "sagemaker"
    assert sagemaker_base.generated_text_jmespath == "generated_text"
    assert sagemaker_base.input_text_jmespath == "inputs"
    assert sagemaker_base.token_count_jmespath == "details.generated_tokens"


def test_sagemaker_base_parse_input(sagemaker_base: ConcreteClass):
    payload = {"inputs": "Test input"}
    assert sagemaker_base._parse_input(payload) == "Test input"


def test_sagemaker_base_create_payload():
    payload = SageMakerBase.create_payload("Test input", max_tokens=100)
    assert payload == {
        "inputs": "Test input",
        "parameters": {
            "max_new_tokens": 100,
            "details": True,
        },
    }


def test_sagemaker_stream_create_payload():
    payload = SageMakerStreamEndpoint.create_payload("Test input", max_tokens=100)
    assert payload == {
        "inputs": "Test input",
        "parameters": {
            "max_new_tokens": 100,
            "details": True,
        },
        "stream": True,
    }


# @patch("boto3.client")
def test_sagemaker_endpoint_invoke(sagemaker_endpoint: SageMakerEndpoint):
    expected_results = {
        "region": sagemaker_endpoint.region,
        "results": [
            {
                "Body": f"""{json.dumps(
                        {
                            "generated_text": "Test output",
                            "details": {"generated_tokens": 10},
                        }
                    )
                }""",
                "ContentType": "application/json",
                # "InvokedProductionVariant": "prod",
                # "CustomAttributes": "my_attr",
            },
        ],
    }
    with mock_aws():
        requests.post(
            "https://motoapi.amazonaws.com/moto-api/static/sagemaker/endpoint-results",
            json=expected_results,
        )

        payload = {"inputs": "Test input"}
        response = sagemaker_endpoint.invoke(payload)

    assert isinstance(response, InvocationResponse)
    assert response.response_text == "Test output"
    assert response.num_tokens_output == 10
    assert isinstance(response.id, str)
    assert UUID(response.id, version=4)


# @patch("boto3.client")
# def test_sagemaker_stream_endpoint_invoke(mock_boto3_client, sagemaker_stream_endpoint: SageMakerStreamEndpoint):
#     mock_client = Mock()
#     mock_boto3_client.return_value = mock_client

#     class MockStream:
#         def __init__(self):
#             self.content = [
#                 b'data: {"token": {"text": "Hello"}}\n',
#                 b'data: {"token": {"text": " World"}}\n',
#                 b'data: {"token": {"text": "!"}}\n',
#             ]
#             self.index = 0

#         def __iter__(self):
#             return self

#         def __next__(self):
#             if self.index < len(self.content):
#                 result = self.content[self.index]
#                 self.index += 1
#                 return result
#             raise StopIteration

#     mock_response = {"Body": MockStream()}
#     mock_client.invoke_endpoint_with_response_stream.return_value = mock_response

#     payload = {"inputs": "Test input"}
#     response = sagemaker_stream_endpoint.invoke(payload)

#     assert isinstance(response, InvocationResponse)
#     assert response.response_text == "Hello World!"
#     assert response.num_tokens_output == 3
#     assert isinstance(response.id, str)
#     assert UUID(response.id, version=4)


def test_sagemaker_endpoint_error_handling(sagemaker_endpoint: SageMakerEndpoint):
    with patch.object(sagemaker_endpoint, "_sagemaker_runtime") as mock_runtime:
        mock_runtime.invoke_endpoint.side_effect = Exception("Test error")

        payload = {"inputs": "Test input"}
        response = sagemaker_endpoint.invoke(payload)

        assert isinstance(response, InvocationResponse)
        assert response.error == "Test error"


def test_sagemaker_stream_endpoint_error_handling(
    sagemaker_stream_endpoint: SageMakerStreamEndpoint,
):
    with patch.object(sagemaker_stream_endpoint, "_sagemaker_runtime") as mock_runtime:
        mock_runtime.invoke_endpoint_with_response_stream.side_effect = Exception(
            "Test error"
        )

        payload = {"inputs": "Test input", "stream": True}
        response = sagemaker_stream_endpoint.invoke(payload)

        assert isinstance(response, InvocationResponse)
        assert response.error == "Test error"


@pytest.fixture
def mock_stream():
    stream = [
        {"token": {"text": k}}
        for k in "sample text to validate the TokenIterator".split()
    ]
    stream[-1] = {**stream[-1], "details": {"generated_tokens": len(stream)}}
    return [
        {"PayloadPart": {"Bytes": json.dumps(s).encode("utf-8") + b"\n"}}
        for s in stream
    ]


def test_token_iterator_next(mock_stream):
    iterator = TokenIterator(mock_stream)
    assert next(iterator) == "sample"
    assert next(iterator) == "text"
    assert next(iterator) == "to"
    # The last item contains details, not a token

    # with pytest.raises(StopIteration):
    [_ for _ in iterator]

    assert iterator.details == {"generated_tokens": 6}


def test_token_iterator_error_handling():
    error_stream = [{"PayloadPart": {"Bytes": b'data: {"error": "Test error"}\n'}}]
    iterator = TokenIterator(error_stream)

    with pytest.raises(Exception) as exc_info:
        next(iterator)

    assert str(exc_info.value) == "Test error"
