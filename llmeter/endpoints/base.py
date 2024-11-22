# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, TypeVar
from uuid import uuid4

from llmeter.serde import JSONableBase

Self = TypeVar(
    "Self", bound="Endpoint"
)  # for python >= 3.11 can be replaced with direct import of `Self`


@dataclass
class InvocationResponse(JSONableBase):
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


class Endpoint(JSONableBase, ABC):
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

    @classmethod
    def from_dict(
        cls: Self, raw: Dict, alt_classes: Dict[str, Self] = {}, **kwargs
    ) -> Self:
        """Load any built-in Endpoint type (or custom ones) from a plain JSON dictionary

        Args:
            raw: A plain Endpoint config dictionary, as created with `to_dict()`, `to_json`, etc.
            alt_classes (Dict[str, type[Endpoint]]): A dictionary mapping additional custom type
                names (beyond those in `llmeter.endpoints`, which are included automatically), to
                corresponding classes for loading custom endpoint types.
            **kwargs: Optional extra keyword arguments to pass to the constructor

        Returns:
            endpoint: An instance of the appropriate endpoint class, initialized with the
                configuration from the file.
        """
        builtin_endpoint_types = importlib.import_module("llmeter.endpoints")
        class_map = {
            **builtin_endpoint_types,
            **alt_classes,
        }
        return super().from_dict(raw, alt_classes=class_map, **kwargs)
