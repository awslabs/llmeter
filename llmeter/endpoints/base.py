# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across the different LLM endpoint types offered by LLMeter

You can also use these classes to implement your own custom `Endpoint` types.
"""

import importlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from upath import UPath as Path


# @dataclass(slots=True)
@dataclass
class InvocationResponse:
    """
    A class representing a invocation result.

    Attributes:
        response_text (str): The invocation output.
        id (str): A unique identifier for the invocation.
        time_to_last_token (float): The time taken to generate the response in seconds.
        time_to_first_token (float): The time taken to receive the first token of the response in seconds.
        num_tokens_output (Optional[int]): The number of tokens in the response.
        num_tokens_input (Optional[int]): The number of tokens in the invocation payload.
        input_prompt (str): The input prompt used in the invocation.
        time_per_output_token (float): The average time taken to generate each token in the response.
        error (str): Any error that occurred during invocation.
    """

    response_text: str | None
    input_payload: dict | None = None
    id: str | None = None
    input_prompt: str | dict | None = None
    time_to_first_token: float | None = None
    time_to_last_token: float | None = None
    num_tokens_input: int | None = None
    num_tokens_output: int | None = None
    time_per_output_token: float | None = None
    error: str | None = None
    retries: int | None = None

    def to_json(self, **kwargs) -> str:
        """
        Convert InvocationResponse to JSON string with binary content support.
        
        This method serializes the InvocationResponse object to a JSON string, with
        automatic handling of binary content (bytes objects) in the input_payload field.
        Binary data is converted to base64-encoded strings wrapped in marker objects,
        enabling JSON serialization while preserving the ability to restore the original
        bytes during deserialization.
        
        Binary Content Handling:
            When the input_payload contains bytes objects (e.g., images, video), they are
            automatically converted to base64-encoded strings and wrapped in marker objects
            with the key "__llmeter_bytes__". This approach enables JSON serialization of
            multimodal payloads while maintaining round-trip integrity.
            
            The marker object format is: {"__llmeter_bytes__": "<base64-string>"}
            
            For non-serializable types other than bytes, the encoder falls back to str()
            representation to ensure the response can always be serialized.

        Args:
            **kwargs: Additional arguments passed to json.dumps (e.g., indent, sort_keys)

        Returns:
            str: JSON representation of the response

        Examples:
            Serialize a response with binary content in the payload:
            
            >>> # Create a response with binary image data in the payload
            >>> with open("image.jpg", "rb") as f:
            ...     image_bytes = f.read()
            >>> response = InvocationResponse(
            ...     response_text="The image shows a cat.",
            ...     input_payload={
            ...         "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            ...         "messages": [{
            ...             "role": "user",
            ...             "content": [
            ...                 {"text": "What is in this image?"},
            ...                 {
            ...                     "image": {
            ...                         "format": "jpeg",
            ...                         "source": {"bytes": image_bytes}
            ...                     }
            ...                 }
            ...             ]
            ...         }]
            ...     },
            ...     time_to_last_token=1.23,
            ...     num_tokens_output=15
            ... )
            >>> json_str = response.to_json()
            >>> # The JSON string contains marker objects for binary data
            >>> "__llmeter_bytes__" in json_str
            True
            
            Serialize with pretty printing:
            
            >>> json_str = response.to_json(indent=2)
            >>> print(json_str)
            {
              "response_text": "The image shows a cat.",
              "input_payload": {
                "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                "messages": [
                  {
                    "role": "user",
                    "content": [
                      {"text": "What is in this image?"},
                      {
                        "image": {
                          "format": "jpeg",
                          "source": {
                            "bytes": {"__llmeter_bytes__": "/9j/4AAQSkZJRg..."}
                          }
                        }
                      }
                    ]
                  }
                ]
              },
              "time_to_last_token": 1.23,
              "num_tokens_output": 15,
              ...
            }
            
            Round-trip serialization with binary preservation:
            
            >>> # Serialize to JSON
            >>> json_str = response.to_json()
            >>> # Parse back to dict
            >>> import json
            >>> from llmeter.prompt_utils import llmeter_bytes_decoder
            >>> response_dict = json.loads(json_str, object_hook=llmeter_bytes_decoder)
            >>> # Binary data is preserved
            >>> original_bytes = response.input_payload["messages"][0]["content"][1]["image"]["source"]["bytes"]
            >>> restored_bytes = response_dict["input_payload"]["messages"][0]["content"][1]["image"]["source"]["bytes"]
            >>> original_bytes == restored_bytes
            True
        """
        from llmeter.results import InvocationResponseEncoder
        
        return json.dumps(asdict(self), cls=InvocationResponseEncoder, **kwargs)

    @staticmethod
    def error_output(
        input_payload: dict | None = None, error=None, id: str | None = None
    ) -> "InvocationResponse":
        return InvocationResponse(
            id=id or uuid4().hex,
            response_text=None,
            input_payload=input_payload,
            time_to_last_token=None,
            error="invocation failed" if error is None else str(error),
        )

    def __repr__(self):
        return self.to_json(
            # default=str,
        )

    def __str__(self):
        return self.to_json(
            indent=4,
            # default=str
        )

    def to_dict(self):
        return asdict(self)


class Endpoint(ABC):
    """
    An abstract base class for endpoint implementations.

    This class defines the basic structure and interface for all endpoint classes.
    It provides abstract methods that must be implemented by subclasses.
    """

    @abstractmethod
    def __init__(
        self,
        endpoint_name: str,
        model_id: str,
        provider: str,
    ):
        """
        Initialize the BaseEndpoint.

        Args:
            endpoint_name (str): The name of the endpoint.
            model_id (str): The identifier of the model associated with this endpoint.
            provider (str): The provider of the endpoint.
        """
        self.endpoint_name = endpoint_name
        self.model_id = model_id
        self.provider = provider

    @abstractmethod
    def invoke(self, payload: dict) -> InvocationResponse:
        """
        Invoke the endpoint with the given payload.

        This method must be implemented by subclasses to define how the endpoint
        is invoked and how the response is processed.

        Args:
            payload (Dict): The input payload for the model.

        Returns:
            InvocationResponse: An object containing the model's response and associated metrics.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @staticmethod
    def create_payload(*args: Any, **kwargs: Any) -> Any:
        """
        Create a payload for the endpoint invocation.

        This static method should be implemented by subclasses to define
        how the payload is created based on the given arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            NotImplemented: This method returns NotImplemented in the base class.
        """
        return NotImplemented

    @classmethod
    def __subclasshook__(cls, C: type) -> bool:
        """
        Determine if a class is considered a subclass of BaseEndpoint.

        This method is used to implement a custom subclass check. A class
        is considered a subclass of BaseEndpoint if it has an 'invoke' method.

        Args:
            C: The class to check.

        Returns:
            bool or NotImplemented: True if the class is a subclass, False if it isn't,
                                    or NotImplemented if the check is inconclusive.
        """
        if cls is Endpoint:
            if any("invoke" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented

    def save(self, output_path: os.PathLike) -> os.PathLike:
        """
        Save the endpoint configuration to a JSON file.

        This method serializes the endpoint's configuration (excluding private attributes)
        to a JSON file at the specified path.

        Args:
            output_path (str | UPath): The path where the configuration file will be saved.

        Returns:
            None
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            endpoint_conf = self.to_dict()

            def _default_serializer(obj):
                if isinstance(obj, os.PathLike):
                    return Path(obj).as_posix()
                return str(obj)

            json.dump(endpoint_conf, f, indent=4, default=_default_serializer)
        return output_path

    def to_dict(self) -> dict:
        """
        Convert the endpoint configuration to a dictionary.

        Returns:
            Dict: A dictionary representation of the endpoint configuration.
        """
        endpoint_conf = {}
        for k, v in vars(self).items():
            if k.startswith("_"):
                continue
            if isinstance(v, os.PathLike):
                v = Path(v).as_posix()
            endpoint_conf[k] = v
        endpoint_conf["endpoint_type"] = self.__class__.__name__
        return endpoint_conf

    @classmethod
    def load_from_file(cls, input_path: os.PathLike) -> "Endpoint":
        """
        Load an endpoint configuration from a JSON file.

        This class method reads a JSON file containing an endpoint configuration,
        determines the appropriate endpoint class, and instantiates it with the
        loaded configuration.

        Args:
            input_path (str|UPath): The path to the JSON configuration file.

        Returns:
            Endpoint: An instance of the appropriate endpoint class, initialized
                      with the configuration from the file.
        """

        input_path = Path(input_path)
        with input_path.open("r") as f:
            data = json.load(f)
        endpoint_type = data.pop("endpoint_type")
        endpoint_module = importlib.import_module("llmeter.endpoints")
        endpoint_class = getattr(endpoint_module, endpoint_type)
        return endpoint_class(**data)

    @classmethod
    def load(cls, endpoint_config: dict) -> "Endpoint":  # type: ignore
        """
        Load an endpoint configuration from a dictionary.

        This class method reads a dictionary containing an endpoint configuration,
        determines the appropriate endpoint class, and instantiates it with the
        loaded configuration.

        Args:
            endpoint_config (Dict): A dictionary containing the endpoint configuration.

        Returns:
            Endpoint: An instance of the appropriate endpoint class, initialized
                      with the configuration from the dictionary.
        """
        endpoint_type = endpoint_config.pop("endpoint_type")
        endpoint_module = importlib.import_module("llmeter.endpoints")
        endpoint_class = getattr(endpoint_module, endpoint_type)
        return endpoint_class(**endpoint_config)
