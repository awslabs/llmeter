# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across the different LLM endpoint types offered by LLMeter

You can also use these classes to implement your own custom `Endpoint` types.

The :func:`llmeter_invoke` decorator wraps a concrete ``invoke`` method with
payload preparation, timing, error handling, and metadata back-fill.  Apply it
to every concrete ``invoke`` in an :class:`Endpoint` subclass.
"""

import functools
import importlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from upath import UPath as Path
from upath.types import ReadablePathLike, WritablePathLike

from ..json_utils import llmeter_default_serializer
from ..utils import ensure_path

logger = logging.getLogger(__name__)


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
        num_tokens_input_cached (Optional[int]): The number of input tokens served from cache (prompt caching).
        input_prompt (str): The input prompt used in the invocation.
        time_per_output_token (float): The average time taken to generate each token in the response.
        error (str): Any error that occurred during invocation.
        request_time (datetime): The wall-clock time when the request was sent.
    """

    response_text: str | None
    input_payload: dict | None = None
    id: str | None = None
    input_prompt: str | dict | None = None
    time_to_first_token: float | None = None
    time_to_last_token: float | None = None
    num_tokens_input: int | None = None
    num_tokens_output: int | None = None
    num_tokens_input_cached: int | None = None
    time_per_output_token: float | None = None
    error: str | None = None
    retries: int | None = None
    request_time: datetime | None = None

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
        input_payload: dict | None = None,
        error=None,
        id: str | None = None,
        request_time: datetime | None = None,
    ) -> "InvocationResponse":
        return InvocationResponse(
            id=id or uuid4().hex,
            response_text=None,
            input_payload=input_payload,
            time_to_last_token=None,
            error="invocation failed" if error is None else str(error),
            request_time=request_time,
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


def llmeter_invoke(fn):
    """Decorator that wraps an :meth:`Endpoint.invoke` implementation with
    payload preparation, timing, error handling, and metadata back-fill.

    Apply this to every concrete ``invoke`` method in an :class:`Endpoint`
    subclass::

        class MyEndpoint(Endpoint):
            @llmeter_invoke
            def invoke(self, payload: dict) -> InvocationResponse:
                raw = self._client.call(**payload)
                return self.parse_response(raw, self._start_t)

    The wrapper performs the following steps around the decorated function:

    1. Calls :meth:`~Endpoint.prepare_payload` to merge ``**kwargs`` and
       inject provider-specific fields.
    2. Records ``self._start_t`` via :func:`time.perf_counter`.
    3. Calls the inner ``invoke``.
    4. On exception, converts it to an error :class:`InvocationResponse`.
    5. Back-fills ``time_to_last_token``, ``input_payload``,
       ``input_prompt``, ``id``, and ``request_time`` if the subclass
       didn't set them.
    """

    @functools.wraps(fn)
    def wrapper(self: "Endpoint", payload: dict, **kw: Any) -> InvocationResponse:
        prepared = self.prepare_payload(payload, **kw)
        self._last_payload = prepared
        request_time = datetime.now(timezone.utc)
        self._start_t = time.perf_counter()
        try:
            response = fn(self, prepared)
        except Exception as e:
            logger.exception("Endpoint invocation failed: %s", e)
            response = InvocationResponse.error_output(
                input_payload=prepared,
                id=uuid4().hex,
                error=str(e),
                request_time=request_time,
            )

        if response.time_to_last_token is None and response.error is None:
            response.time_to_last_token = time.perf_counter() - self._start_t

        if response.input_payload is None:
            response.input_payload = prepared
        if response.input_prompt is None:
            try:
                response.input_prompt = self._parse_payload(prepared)
            except Exception:
                logger.debug("_parse_payload failed; leaving input_prompt as None")
        if response.id is None:
            response.id = uuid4().hex
        if response.request_time is None:
            response.request_time = request_time

        return response

    wrapper._is_llmeter_invoke = True
    return wrapper


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
        """Invoke the endpoint with the given payload.

        Subclasses implement this with their provider-specific API call,
        passing the raw response to :meth:`parse_response`.  Decorate the
        concrete implementation with :func:`llmeter_invoke` to get automatic
        payload preparation, timing, error handling, and metadata back-fill::

            @llmeter_invoke
            def invoke(self, payload: dict) -> InvocationResponse:
                raw = self._client.call(**payload)
                return self.parse_response(raw, self._start_t)

        The :func:`llmeter_invoke` wrapper provides:

        * **Payload preparation** — calls :meth:`prepare_payload` before the
          inner function, so ``**kwargs`` are merged and provider-specific
          fields (``model``, ``modelId``, etc.) are set.
        * **Timing** — ``self._start_t`` is set immediately before the call
          and ``time_to_last_token`` is back-filled on the response if the
          subclass didn't set it (streaming endpoints typically set it during
          stream consumption).
        * **Error handling** — unhandled exceptions are caught, logged, and
          converted to an error :class:`InvocationResponse`.
        * **Metadata back-fill** — ``input_payload`` and ``input_prompt`` are
          guaranteed to be set on the returned response (success or error),
          using the prepared payload.

        Args:
            payload: The prepared input payload for the model.

        Returns:
            InvocationResponse: An object containing the model's response and
                associated metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, raw_response: Any, start_t: float) -> InvocationResponse:
        """Parse the raw API response into an :class:`InvocationResponse`.

        Subclasses implement this to extract the generated text, token counts,
        timing information, and any other provider-specific metadata from the
        raw object returned by the API client.

        This method is called from :meth:`invoke` after the API call.
        Exceptions raised here will be caught by the base-class ``invoke``
        wrapper and converted to an error response automatically.

        Non-streaming endpoints can ignore ``start_t`` — the wrapper
        back-fills ``time_to_last_token`` automatically.  Streaming endpoints
        use it to compute ``time_to_first_token`` and ``time_to_last_token``
        during stream consumption.

        Args:
            raw_response: The raw response object from the provider's API
                client.  The type varies by provider (e.g. ``ChatCompletion``,
                ``dict``, a streaming iterator).
            start_t: The :func:`time.perf_counter` timestamp captured
                immediately before the API call (also available as
                ``self._start_t``).

        Returns:
            InvocationResponse: Parsed response with at least
                ``response_text`` populated.
        """
        raise NotImplementedError

    def prepare_payload(self, payload: dict, **kwargs: Any) -> dict:
        """Prepare the payload before sending it to the API.

        This method is called by the ``invoke`` wrapper before the actual
        invocation.  Subclasses can override it to merge ``**kwargs``, inject
        provider-specific fields (``model``, ``modelId``, streaming options,
        etc.), or apply any other transformation.

        The default implementation returns *payload* unchanged (ignoring
        ``**kwargs``).

        Args:
            payload: The raw input payload from the caller.
            **kwargs: Additional keyword arguments from the caller.

        Returns:
            dict: The final payload to send to the API.
        """
        return payload

    def _parse_payload(self, payload: dict) -> str | dict | None:
        """Extract the user-facing input text from an API request payload.

        The ``invoke`` wrapper calls this automatically to populate
        :pyattr:`InvocationResponse.input_prompt`.  That field serves two
        purposes:

        * **Observability** — it records *what* was sent to the model in a
          human-readable form, separate from the full API payload (which may
          contain binary media, inference config, etc.).
        * **Token counting fallback** — when the API response does not include
          an input-token count, the :class:`~llmeter.runner.Runner` tokenizes
          ``input_prompt`` to estimate it.

        Subclasses should override this to navigate their provider-specific
        payload structure and return the concatenated message text.  The
        default implementation returns ``None`` (no prompt extracted).

        Args:
            payload: The prepared request payload (after :meth:`prepare_payload`).

        Returns:
            The extracted prompt text, or ``None`` if extraction is not
            possible.
        """
        return None

    @staticmethod
    def create_payload(*args: Any, **kwargs: Any) -> Any:
        """
        Create a payload for the endpoint invocation.

        This static method should be implemented by subclasses to define
        how the payload is created based on the given arguments. Ideally,
        subclasses should conform to the conventions of existing endpoint types
        (for example taking a `user_message: str | list[ContentItem]` param),
        but this is not strictly enforced at the typing level.

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
