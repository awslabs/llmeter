# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across the different LLM endpoint types offered by LLMeter

You can also use these classes to implement your own custom `Endpoint` types.
"""

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from upath import UPath as Path
from upath.types import ReadablePathLike, WritablePathLike

from ..json_utils import llmeter_default_serializer
from ..utils import ensure_path


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

    def to_json(self, default=llmeter_default_serializer, **kwargs) -> str:
        """Serialize this response to a JSON string.

        Uses :func:`~llmeter.json_utils.llmeter_default_serializer` by default,
        which handles ``bytes``, ``datetime``, ``PathLike``, and other common
        non-serializable types.

        Args:
            default: Fallback serializer passed to :func:`json.dumps`.
                Defaults to :func:`~llmeter.json_utils.llmeter_default_serializer`.
            **kwargs: Additional arguments passed to :func:`json.dumps`
                (e.g., ``indent``, ``sort_keys``).

        Returns:
            str: JSON representation of the response.
        """
        return json.dumps(asdict(self), default=default, **kwargs)

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

    def save(self, output_path: WritablePathLike) -> Path:
        """
        Save the endpoint configuration to a JSON file.

        This method serializes the endpoint's configuration (excluding private attributes)
        to a JSON file at the specified path.

        Args:
            output_path (str | UPath): The path where the configuration file will be saved.

        Returns:
            None
        """
        output_path = ensure_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(self, f, indent=4, default=llmeter_default_serializer)
        return output_path

    def to_dict(self) -> dict:
        """
        Convert the endpoint configuration to a dictionary.

        Returns:
            Dict: A dictionary representation of the endpoint configuration.
        """
        endpoint_conf = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        endpoint_conf["endpoint_type"] = self.__class__.__name__
        return endpoint_conf

    @classmethod
    def load_from_file(cls, input_path: ReadablePathLike) -> "Endpoint":
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

        input_path = ensure_path(input_path)
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
