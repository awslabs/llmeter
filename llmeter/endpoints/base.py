# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across the different LLM endpoint types offered by LLMeter

You can also use these classes to implement your own custom `Endpoint` integrations.
"""

import copy
import functools
import importlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar
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
        num_tokens_input_cached: The number of input tokens served from cache (prompt caching).
        num_tokens_output_reasoning: The number of output tokens used for internal reasoning
            (included in `num_tokens_output`). Populated when the provider reports a separate
            reasoning/thinking token count (e.g. OpenAI `reasoning_tokens`). `None` when the
            provider does not break out reasoning tokens — for example, Anthropic includes
            thinking tokens in `output_tokens` without a separate count.
        input_prompt (str): The input prompt used in the invocation.
        time_per_output_token (float): The average time taken to generate each token in the response.
        error (str): Any error that occurred during invocation.
        request_time: The wall-clock time when the request was sent.
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
    num_tokens_output_reasoning: int | None = None
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
        """Return a dictionary representation of this response.

        Returns a plain ``dict`` produced by :func:`dataclasses.asdict`,
        preserving native Python types (e.g. ``datetime`` for
        ``request_time``).  This is suitable for programmatic access —
        for example :class:`~llmeter.utils.RunningStats` consumes this
        output and relies on ``datetime`` comparisons and arithmetic.

        For JSON output, use :meth:`to_json` which delegates to
        :func:`~llmeter.json_utils.llmeter_default_serializer` for
        non-serializable types, or pass the dict through
        ``json.dumps(response.to_dict(), default=llmeter_default_serializer)``.

        Returns:
            dict: A dictionary of response fields with native Python types.
        """
        return asdict(self)


TRawResponse = TypeVar("TRawResponse", bound=Any)


class Endpoint(ABC, Generic[TRawResponse]):
    """
    An abstract base class for endpoint implementations.

    We strongly recommend using the
    [`llmeter_invoke`][llmeter.endpoints.base.Endpoint.llmeter_invoke] decorator to implement
    custom endpoints as shown below - which wraps payload pre-processing, response parsing, and
    error handling around a core invoke function you provide.

    Example:
        ```python
        class MyCustomEndpoint(Endpoint[MyAISDKRawReturnType]):
            @Endpoint.llmeter_invoke
            def invoke(self, payload: dict) -> MyAISDKRawReturnType:
                # Just the raw AI / SDK call goes here:
                raw: MyAISDKRawReturnType = self._my_cool_api_client.call(**payload)
                return raw

            def process_raw_response(
                self,
                raw_response: MyAISDKRawReturnType,
                start_t: float,
                response: InvocationResponse
            ):
                # llmeter_invoke wrapper automatically calls process_raw_response,
                # in which you should parse the outputs onto `response`
                response.id = raw_response["ResponseId"]
                ...
        ```

    See [`llmeter_invoke`][llmeter.endpoints.base.Endpoint.llmeter_invoke] and
    [`process_raw_response`][llmeter.endpoints.base.Endpoint.process_raw_response]for more info.

    You can also implement:

    - [`create_payload`][llmeter.endpoints.base.Endpoint.create_payload] convenience method to
        simplify building payload objects for your endpoint - for example converting a simple input
        prompt to a full request object with other required parameters.
    - [`prepare_payload`][llmeter.endpoints.base.Endpoint.prepare_payload] in case you need to do
        any request payload pre-processing **outside** the timer that measures response speed
    """

    @classmethod
    def llmeter_invoke(
        cls,
        call_endpoint: Callable[..., TRawResponse],
    ) -> Callable[..., InvocationResponse]:
        """Wrap a raw model API call with pre+postprocessing and error handling

        This decorator wraps around a function that *only* does the core model call, to add the
        full range of steps that LLMeter Endpoints are expected to handle as part of `invoke`:

        1. **Before** starting the response timer, calls your class'
            [`prepare_payload`](llmeter.endpoints.base.Endpoint.prepare_payload) method to
            transform the input payload, if required
        2. Initialises an [`InvocationResponse`](llmeter.endpoints.base.InvocationResponse) with
            the timestamp of the request.
        3. Calls the wrapped function to fetch the raw API response
        4. Calls your class'
            [`process_raw_response`](llmeter.endpoints.base.Endpoint.process_raw_response) method
            to incrementally parse fields from the raw response to the target `InvocationResponse`
        5. In case of any unhandled errors during API call or response processing, logs and sets
            `response.error`
        6. Automatically backfills the following fields on the parsed response, if missing:
            - `id` (as a generated UUID)
            - `input_payload` (the final payload sent to the API)
            - `input_prompt` (via
                [`_parse_payload`](llmeter.endpoints.base.Endpoint._parse_payload) method)
            - `time_to_last_token`

        Args:
            call_endpoint: The function to wrap. Should be a method that takes a `payload: dict`
                and returns a `raw_response` object for input to `process_raw_response`

        Returns:
            A wrapped function that implements the full `invoke` logic.
        """

        @functools.wraps(call_endpoint)
        def wrapper(self: "Endpoint", payload: dict) -> InvocationResponse:
            prepared = self.prepare_payload(payload)
            # Snapshot before the API call for _parse_payload, which runs after
            # the inner invoke — by which point the client may have mutated the dict.
            saved_payload = copy.deepcopy(prepared)
            default_response_id = uuid4().hex
            response = InvocationResponse(
                id=default_response_id,
                request_time=datetime.now(timezone.utc),
                response_text=None,
            )
            start_t = time.perf_counter()
            try:
                raw_response: TRawResponse = call_endpoint(self, prepared)
                self.process_raw_response(raw_response, start_t, response)
                default_end_t = time.perf_counter()
            except Exception as e:
                default_end_t = time.perf_counter()
                logger.exception("Endpoint invocation failed: %s", response.error or e)
                if not response.error:
                    response.error = str(e)

            if response.id is None:
                # Just in case user's parsing logic accidentally cleared the default ID provided:
                response.id = default_response_id

            if response.time_to_last_token is None and response.error is None:
                response.time_to_last_token = default_end_t - start_t

            if response.input_payload is None:
                response.input_payload = prepared
            if response.input_prompt is None:
                try:
                    response.input_prompt = self._parse_payload(saved_payload)
                except Exception:
                    logger.debug("_parse_payload failed; leaving input_prompt as None")

            return response

        # Add a private marker to indicate that the wrapping happened:
        # (We don't currently use this for anything except unit tests)
        wrapper._is_llmeter_invoke = True  # type: ignore
        return wrapper

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
        """Call a model and return a full parsed response with error handling

        !!! info
            We strongly encourage to use the
            [`llmeter_invoke`](llmeter.endpoints.base.Endpoint.llmeter_invoke) decorator to implement
            your invoke method with proper orchestration and error handling.

        `Endpoint.invoke` should:

        1. Call `prepare_payload` to transform the input payload
        2. Invoke your actual target endpoint
        3. Parse the results onto an
            [`InvocationResponse`](llmeter.endpoints.base.InvocationResponse) object (preferably
            via [`process_raw_response`](llmeter.endpoints.base.Endpoint.process_raw_response))
        4. Populate `.error` and as many other response fields as possible, in the event that an
            error occurs during model calling or response processing

        The `llmeter_invoke` decorator handles this flow for you - so you'll need to re-implement
        the steps if you choose not to use it.

        Args:
            payload: The input payload for the model.

        Returns:
            response: The final `InvocationResponse`, including all the information that could be
                parsed from the API response - even in case of an error (when the ``error`` field
                should also be set)
        """
        raise NotImplementedError

    def prepare_payload(self, payload: dict) -> dict:
        """Transform the payload before sending it to the API.

        You can use it to enforce any transformations you need between the input dataset/payload
        and what actually gets sent to the model, that should not be counted in the response time
        measurement. For example: Setting fixed parameters required by the endpoint e.g.
        `streaming: False`.

        This method is called by the
        [`llmeter_invoke`](llmeter.endpoints.base.Endpoint.llmeter_invoke) wrapper *before*
        starting the timer that measures response latency.

        !!! warning
            If you made a custom :meth:`invoke` implementation **without** using the
            :meth:`llmeter_invoke` decorator - check whether your implementation actually calls
            this `prepare_payload` method or not!

        The default implementation returns ``payload`` unchanged

        Args:
            payload: The raw input payload from the caller.

        Returns:
            dict: The final payload to send to the API.
        """
        return payload

    @abstractmethod
    def process_raw_response(
        self,
        raw_response: TRawResponse,
        start_t: float,
        response: InvocationResponse,
    ) -> None:
        """Parse a raw API response onto `InvocationResponse` fields

        Subclasses implement this to extract LLMeter data points (such as time to first and last
        token, output text, number of input/output tokens, etc.) from raw model responses.

        !!! warning
            If you made a custom :meth:`invoke` implementation **without** using the
            :meth:`llmeter_invoke` decorator - check whether your implementation actually calls
            this `process_raw_response` method or not!

        This function does not return a value, but is instead expected to incrementally populate
        properties on the provided draft ``response`` object.

        In this way, partial data will be stored even if an error occurs later during processing.
        For example if a stream times out, or a guardrail intervenes - we might still be able to
        capture a unique ID initially pulled from the response header.

        See [`llmeter_invoke`](llmeter.endpoints.base.Endpoint.llmeter_invoke) for more details
        about which fields of `InvocationResponse` are automatically populated for you.

        Args:
            raw_response: The raw response object (returned by your `invoke` method before it's
                wrapped with `llmeter_invoke`)
            start_t: `time.perf_counter` timestamp captured immediately before the API call.
                Use this to calculate and populate `response.time_to_last_token` and (if in
                streaming mode) `response.time_to_first_token`.
            response: The LLMeter response object to be populated **in-place**.

        Raises:
            Exception: If something goes wrong during response streaming or parsing,
                implementations can just raise an error. The :meth:`llmeter_invoke` wrapper will
                populate ``response.error`` and ``response.time_to_last_token`` if they're not set
                already.
        """
        raise NotImplementedError

    def _parse_payload(self, payload: dict) -> str | dict | None:
        """Extract the user-facing input text from an API request payload.

        The `invoke` wrapper calls this automatically to populate
        `InvocationResponse.input_prompt`.  That field serves two purposes:

        * **Observability** — it records *what* was sent to the model in a
          human-readable form, separate from the full API payload (which may
          contain binary media, inference config, etc.).
        * **Token counting fallback** — when the API response does not include
          an input-token count, the :class:`~llmeter.runner.Runner` tokenizes
          ``input_prompt`` to estimate it.

        Subclasses should override this to navigate their provider-specific
        payload structure and return the concatenated message text.  The
        default implementation returns `None` (no prompt extracted).

        Args:
            payload: The prepared request payload (after `prepare_payload`).

        Returns:
            The extracted prompt text, or `None` if extraction is not
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
