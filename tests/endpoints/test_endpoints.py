# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import llmeter
import llmeter.endpoints
from llmeter.endpoints.base import InvocationResponse, Endpoint

# Tests for InvocationResponse


def test_invocation_response_initialization():
    response = InvocationResponse(
        id="test_id",
        response_text="Hello, world!",
        input_prompt="Say hello",
        time_to_first_token=0.1,
        time_to_last_token=0.5,
        num_tokens_input=3,
        num_tokens_output=2,
        time_per_output_token=0.2,
    )
    assert response.id == "test_id"
    assert response.response_text == "Hello, world!"
    assert response.input_prompt == "Say hello"
    assert response.time_to_first_token == 0.1
    assert response.time_to_last_token == 0.5
    assert response.num_tokens_input == 3
    assert response.num_tokens_output == 2
    assert response.time_per_output_token == 0.2
    assert response.error is None


def test_invocation_response_to_json():
    response = InvocationResponse(
        id="test_id", response_text="Hello, world!", input_prompt="Say hello"
    )
    json_str = response.to_json()
    assert "test_id" in json_str
    assert "Hello, world!" in json_str
    assert "Say hello" in json_str


def test_invocation_response_error_output():
    error_response = InvocationResponse.error_output(
        input_payload={"input": "Test prompt"}, error="Test error"
    )
    assert error_response.response_text is None
    assert error_response.input_payload == {"input": "Test prompt"}
    assert error_response.error == "Test error"
    assert error_response.id is not None


def test_invocation_response_repr_and_str():
    response = InvocationResponse(
        id="test_id", response_text="Hello, world!", input_prompt="Say hello"
    )
    repr_str = repr(response)
    str_str = str(response)
    assert "test_id" in repr_str
    assert "Hello, world!" in repr_str
    assert "Say hello" in str_str
    assert repr_str != str_str  # str should be indented


# Tests for BaseEndpoint


class ConcreteEndpoint(Endpoint):
    def __init__(self, endpoint_name: str, model_id: str, provider: str):
        super().__init__(
            endpoint_name=endpoint_name, model_id=model_id, provider=provider
        )

    def invoke(self, payload: dict) -> InvocationResponse:
        return InvocationResponse(
            id="test_id",
            response_text=f"Invoked with payload: {payload}",
            input_prompt=payload.get("prompt", ""),
        )

    @classmethod
    def create_payload(cls, prompt: str):
        return {"prompt": prompt}


llmeter.endpoints.ConcreteEndpoint = ConcreteEndpoint  # type: ignore


@pytest.fixture
def concrete_endpoint():
    return ConcreteEndpoint("test_endpoint", "test_model", "test_provider")


def test_base_endpoint_initialization(concrete_endpoint):
    assert concrete_endpoint.endpoint_name == "test_endpoint"
    assert concrete_endpoint.model_id == "test_model"


def test_base_endpoint_invoke(concrete_endpoint):
    payload = {"prompt": "Hello"}
    response = concrete_endpoint.invoke(payload)
    assert isinstance(response, InvocationResponse)
    assert response.id == "test_id"
    assert response.response_text == "Invoked with payload: {'prompt': 'Hello'}"
    assert response.input_prompt == "Hello"


def test_base_endpoint_create_payload():
    payload = ConcreteEndpoint.create_payload("Test prompt")
    assert payload == {"prompt": "Test prompt"}


def test_endpoint_abstract_methods():
    with pytest.raises(TypeError):
        Endpoint("test", "test", "test")  # type: ignore


def test_endpoint_to_dict(concrete_endpoint):
    endpoint_dict = concrete_endpoint.to_dict()
    assert endpoint_dict == {
        "endpoint_name": "test_endpoint",
        "model_id": "test_model",
        "provider": "test_provider",
        "endpoint_type": "ConcreteEndpoint",
    }


def test_endpoint_save_and_load(concrete_endpoint, tmp_path):
    save_path = tmp_path / "test_endpoint.json"
    concrete_endpoint.save(save_path)

    loaded_endpoint = ConcreteEndpoint.load_from_file(save_path)
    assert loaded_endpoint.endpoint_name == concrete_endpoint.endpoint_name
    assert loaded_endpoint.model_id == concrete_endpoint.model_id
    assert loaded_endpoint.provider == concrete_endpoint.provider


def test_endpoint_load_from_dict():
    config = {
        "endpoint_name": "test_endpoint",
        "model_id": "test_model",
        "provider": "test_provider",
        "endpoint_type": "ConcreteEndpoint",
    }
    loaded_endpoint = ConcreteEndpoint.load(config)
    assert loaded_endpoint.endpoint_name == "test_endpoint"
    assert loaded_endpoint.model_id == "test_model"
    assert loaded_endpoint.provider == "test_provider"


def test_invocation_response_to_dict():
    response = InvocationResponse(
        id="test_id",
        response_text="Hello, world!",
        input_prompt="Say hello",
        time_to_first_token=0.1,
        time_to_last_token=0.5,
        num_tokens_input=3,
        num_tokens_output=2,
        time_per_output_token=0.2,
    )
    response_dict = response.to_dict()
    assert response_dict["id"] == "test_id"
    assert response_dict["response_text"] == "Hello, world!"
    assert response_dict["input_prompt"] == "Say hello"
    assert response_dict["time_to_first_token"] == 0.1
    assert response_dict["time_to_last_token"] == 0.5
    assert response_dict["num_tokens_input"] == 3
    assert response_dict["num_tokens_output"] == 2
    assert response_dict["time_per_output_token"] == 0.2


def test_endpoint_subclasshook():
    class ValidEndpoint(Endpoint):
        def invoke(self, payload):
            pass

        @staticmethod
        def create_payload():
            pass

    class InvalidEndpoint:
        pass

    assert issubclass(ValidEndpoint, Endpoint)
    assert not issubclass(InvalidEndpoint, Endpoint)


def test_endpoint_load_from_file_error(tmp_path):
    invalid_path = tmp_path / "nonexistent_file.json"
    with pytest.raises(FileNotFoundError):
        ConcreteEndpoint.load_from_file(invalid_path)


def test_endpoint_load_error():
    invalid_config = {
        "endpoint_name": "test_endpoint",
        "model_id": "test_model",
        # Missing "provider" key
    }
    with pytest.raises(KeyError):
        ConcreteEndpoint.load(invalid_config)
