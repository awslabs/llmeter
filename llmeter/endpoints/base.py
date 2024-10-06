# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, TypeVar
from uuid import uuid4

from upath import UPath as Path

Self = TypeVar(
    "Self", bound="Endpoint"
)  # for python >= 3.11 can be replaced with direct import of `Self`


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
    id: str | None = None
    input_prompt: str | None = None
    time_to_first_token: float | None = None
    time_to_last_token: float | None = None
    num_tokens_input: int | None = None
    num_tokens_output: int | None = None
    time_per_output_token: float | None = None
    error: str | None = None

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.__dict__, **kwargs)

    @staticmethod
    def error_output(
        input_prompt: str | None = None, error=None, id: str | None = None
    ):
        return InvocationResponse(
            id=id or uuid4().hex,
            response_text=None,
            input_prompt=input_prompt,
            time_to_last_token=None,
            error="invocation failed" if error is None else str(error),
        )

    def __repr__(self):
        return self.to_json(default=str)

    def __str__(self):
        return self.to_json(indent=4, default=str)

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

        Returns:
            None
        """
        self.endpoint_name = endpoint_name
        self.model_id = model_id
        self.provider = provider

    @abstractmethod
    def invoke(self, payload: Dict) -> InvocationResponse:
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
    def create_payload(*args, **kwargs):
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
    def __subclasshook__(cls, C):
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
            json.dump(endpoint_conf, f, indent=4, default=str)
        return output_path

    def to_dict(self) -> Dict:
        """
        Convert the endpoint configuration to a dictionary.

        Returns:
            Dict: A dictionary representation of the endpoint configuration.
        """
        endpoint_conf = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        endpoint_conf["endpoint_type"] = self.__class__.__name__
        return endpoint_conf

    @classmethod
    def load_from_file(cls, input_path: os.PathLike) -> Self:
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
    def load(cls, endpoint_config: Dict) -> Self:  # type: ignore
        """
        Load an endpoint configuration from a dictionary.

        This class method reads a dictionary containing an endpoint configuration,
        determines the appropriate endpoint class, and instantiates it with the
        loaded configuration.

        Args:
            data (Dict): A dictionary containing the endpoint configuration.

        Returns:
            Endpoint: An instance of the appropriate endpoint class, initialized
                      with the configuration from the dictionary.
        """
        endpoint_type = endpoint_config.pop("endpoint_type")
        endpoint_module = importlib.import_module("llmeter.endpoints")
        endpoint_class = getattr(endpoint_module, endpoint_type)
        return endpoint_class(**endpoint_config)
