# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across the different LLM endpoint types offered by LLMeter

You can also use these classes to implement your own custom `Endpoint` integrations.
"""

import copy
import functools
import importlib
import inspect
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Generic, Literal, TypeVar
from uuid import uuid4

from upath import UPath as Path
from upath.types import ReadablePathLike, WritablePathLike

from ..serialization import (
    Serializable,
    bytes_decoder,
    json_default,
    load_object,
    restore_dataclass_types,
)
from ..utils import ensure_path

logger = logging.getLogger(__name__)

ReasoningType = Literal["verbatim", "summary", "redacted", "unknown"]
"""
How internal reasoning was disclosed in a model's response, when the model reasoned at all.

What matters for timing (and `time_per_output_token` calculation) is whether the reasoning reached
us *as it was generated*, or after the fact, or not at all.

* `"verbatim"`: The raw reasoning tokens were streamed as plain text, so their decode time is
    inside the window between `time_to_first_token` and `time_to_last_token`.
* `"summary"`: Only a *summary* of the reasoning was streamed. A summary is shorter than the
    reasoning it describes and cannot be produced until some reasoning already exists, so at least
    part of the reasoning generation falls outside the measured TTFT-TTLT window.
* `"redacted"`: The reasoning content was withheld. This covers both content that is encrypted
    but still delivered (Bedrock Converse `reasoningContent.redactedContent` deltas, Anthropic
    `redacted_thinking` blocks) and content that is not delivered at all until it is complete
    (Anthropic `thinking.display: "omitted"`, where a trailing `signature_delta` is the only
    signal). Whether a given API delivers redacted reasoning as it is generated is not documented
    and appears to vary, so LLMeter does not distinguish the two and conservatively refuses to
    use either for TPOT calculation.
* `"unknown"`: Reasoning demonstrably happened, but the endpoint could not establish how it was
    disclosed. Distinct from `None`, which means no reasoning was observed at all. See also
    [`backfill_reasoning_type_from_token_counts`][llmeter.endpoints.base.backfill_reasoning_type_from_token_counts].

Only `"verbatim"` - and `None`, where there is no reasoning to account for - permit calculating the
`time_per_output_token` from the whole measured TTFT-TTLT window and full output token count. See
[`_Run._compute_time_per_output_token`][llmeter.runner._Run._compute_time_per_output_token].
"""


# @dataclass(slots=True)
@dataclass
class InvocationResponse:
    """
    A class representing a invocation result.

    Attributes:
        response_text (str): The invocation output.
        id (str): A unique identifier for the invocation.
        time_to_last_token (float): The time taken to generate the response in seconds.
        time_to_first_token: Seconds until the first output token of **any** kind arrived,
            including internal reasoning/thinking tokens. `None` for non-streaming endpoints.

            Note this is the first output *received*, which is not always the first token
            *generated*: where a model withholds its reasoning, the earliest observable signal can
            arrive after reasoning finished.
            [`reasoning_type`][llmeter.endpoints.base.ReasoningType] records when that applies, and
            is what makes such values recognisable as not comparable with a model that streams its
            reasoning.
        time_to_first_content_token: Seconds until the first **visible** (non-reasoning) output
            token arrived. In applications where users see streaming output but internal "thinking"
            is hidden, this will correspond closely to user-perceived latency. Equal to
            `time_to_first_token` when the model emits no reasoning tokens. `None` for
            non-streaming endpoints.
        num_tokens_output (Optional[int]): The number of tokens in the response.
        num_tokens_input (Optional[int]): The number of tokens in the invocation payload.
        num_tokens_input_cached: The number of input tokens served from cache (prompt caching).
        num_tokens_output_reasoning: The number of output tokens used for internal reasoning
            (included in `num_tokens_output`). Populated when the provider reports a separate
            reasoning/thinking token count — for example OpenAI `reasoning_tokens`, or Anthropic
            `output_tokens_details.thinking_tokens`. `None` when the provider does not provide a
            separate count for this.
        input_prompt (str): The input prompt used in the invocation.
        reasoning_type: How the model's internal reasoning was disclosed, or `None` if the model
            does not appear to have reasoned at all. See
            [`ReasoningType`][llmeter.endpoints.base.ReasoningType]. This governs whether
            `time_per_output_token` can be derived.
        time_per_output_token (float): The average time taken to generate each token in the
            response, **excluding** initial prompt processing/prefill. Computed by the `Runner`
            from whichever pairing of first-token metric and token count is internally consistent;
            `None` when no consistent pairing is available.
        error (str): Any error that occurred during invocation.
        request_time: The wall-clock time when the request was sent.
        annotations (dict): Free-form extra data attached to this response, for example by
            callbacks. This is the **preferred** place for a `Callback` to store additional
            per-response fields (rather than setting arbitrary attributes on the response), because
            `annotations` is a declared field and therefore round-trips through
            `to_json`/`from_json` and disk persistence. On load, any *unrecognized* top-level keys
            (e.g. from older files or other LLMeter versions) are collected into `annotations` too.
    """

    response_text: str | None
    input_payload: dict | None = None
    id: str | None = None
    input_prompt: str | dict | None = None
    time_to_first_token: float | None = None
    time_to_first_content_token: float | None = None
    time_to_last_token: float | None = None
    num_tokens_input: int | None = None
    num_tokens_output: int | None = None
    num_tokens_input_cached: int | None = None
    num_tokens_output_reasoning: int | None = None
    reasoning_type: ReasoningType | None = None
    time_per_output_token: float | None = None
    error: str | None = None
    retries: int | None = None
    request_time: datetime | None = None
    annotations: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_str: str) -> "InvocationResponse":
        """Deserialize a JSON string into an `InvocationResponse`.

        This is the inverse of [`to_json`][llmeter.endpoints.base.InvocationResponse.to_json]. It
        correctly restores types that the default JSON round-trip would leave as strings or marker
        objects:

        * `datetime`-annotated fields are parsed from ISO-8601 strings back to Python `datetime`
        * `bytes`-typed fields and `__llmeter_bytes__` markers in nested payloads are restored

        For legacy compatability, any top-level keys that are *not* recognized fields are currently
        collected into [`annotations`][llmeter.endpoints.base.InvocationResponse] rather than being
        dropped or raising an error. This supports older files where CostModel callbacks wrote
        extra fields (such as `cost_*`) directly onto the response - but may be dropped in a future
        version.

        Args:
            json_str: A JSON string representation of an InvocationResponse (produced by `to_json`
                or similar).

        Returns:
            InvocationResponse: The deserialized response.

        Example:
            A round-trip can be run as follows:
            ```python
            original = InvocationResponse(response_text="hi", ...)
            restored = InvocationResponse.from_json(original.to_json())
            ```
        """
        data = json.loads(json_str, object_hook=bytes_decoder)
        restore_dataclass_types(cls, data)
        # Route any unrecognized top-level keys into `annotations` rather than failing. This keeps
        # forward/backward compatibility: e.g. older files where callbacks wrote extra fields (like
        # `cost_*`) directly onto the response, or fields written by a different LLMeter version.
        known = {f.name for f in fields(cls)}
        extras = {k: data.pop(k) for k in list(data) if k not in known}
        if extras:
            data["annotations"] = {**extras, **(data.get("annotations") or {})}
            logger.debug(
                "Loaded %d unrecognized InvocationResponse field(s) into `annotations`: %s",
                len(extras),
                ", ".join(sorted(extras)),
            )
        return cls(**data)

    def to_json(
        self, default: Callable[[Any], Any] | None = json_default, **kwargs: Any
    ) -> str:
        """Serialize this response to a JSON string.

        Uses [`json_default`][llmeter.serialization.json_default] by
        default, which handles `bytes`, `datetime`, `PathLike`, and other common non-serializable
        types.

        Args:
            default: Fallback serializer passed to `json.dumps`.
            **kwargs: Additional arguments passed to `json.dumps` (e.g., `indent`, `sort_keys`).

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

    def to_dict(self) -> dict:
        """Return a dictionary representation of this response.

        Returns a plain `dict` produced by `dataclasses.asdict`, preserving native Python types
        (e.g. `datetime` for `request_time`).  This is suitable for programmatic access — for
        example [`RunningStats`][llmeter.utils.RunningStats] consumes this output and relies on
        `datetime` comparisons and arithmetic.

        For JSON output, use [`to_json`][llmeter.endpoints.base.InvocationResponse.to_json], (which
        delegates to [`json_default`][llmeter.serialization.json_default]
        by default, for non-JSON-serializable data types).

        Returns:
            dict: A dictionary of response fields with native Python types.
        """
        return asdict(self)


_TTFT_VISIBLE_TOKENS_ONLY_DEPRECATION = (
    "`ttft_visible_tokens_only` is deprecated and no longer has any effect. LLMeter now always "
    "records both `InvocationResponse.time_to_first_token` (the first output token of any kind, "
    "including reasoning) and `InvocationResponse.time_to_first_content_token` (the first visible "
    "output token). Read `time_to_first_content_token` for the behaviour previously selected by "
    "`ttft_visible_tokens_only=True`. This parameter will be removed in a future release."
)


def warn_if_ttft_visible_tokens_only_set(value: bool | None) -> None:
    """Warn if a caller passed the retired `ttft_visible_tokens_only` endpoint argument.

    Streaming endpoints that support reasoning models now unconditionally record both
    [`time_to_first_token`][llmeter.endpoints.base.InvocationResponse] and
    [`time_to_first_content_token`][llmeter.endpoints.base.InvocationResponse], so the flag that
    used to choose between the two is redundant. It is still accepted (and ignored) so that
    endpoint configurations saved by earlier LLMeter versions keep loading.

    Args:
        value: The value the caller passed, or `None` if the argument was omitted. Only a
            non-`None` value triggers the warning.
    """
    if value is None:
        return
    warnings.warn(
        _TTFT_VISIBLE_TOKENS_ONLY_DEPRECATION, DeprecationWarning, stacklevel=3
    )


def _get_delta_field(delta: Any, name: str) -> Any:
    """Read a named field from a streaming delta, whether it's a mapping or an object.

    Args:
        delta: A mapping (e.g. parsed JSON dict) or an object with attributes (e.g. an SDK model).
        name: The field name to read.

    Returns:
        The field value, or `None` if absent.
    """
    if isinstance(delta, Mapping):
        return delta.get(name)
    return getattr(delta, name, None)


def delta_has_reasoning_content(delta: Any) -> bool:
    """Check whether an OpenAI-style streaming `delta` carries reasoning/thinking content.

    The OpenAI Chat Completions schema has no standard field for reasoning tokens, so providers
    and proxies each added their own. This checks the field names in common use:

    * `reasoning_content` - DeepSeek, vLLM, SGLang, Bedrock Mantle (`gpt-oss`), and LiteLLM's
      normalized form
    * `reasoning` - OpenRouter and several OpenAI-compatible gateways
    * `thinking_blocks` - LiteLLM's structured form for Anthropic-style thinking blocks

    Only presence is reported, not the content itself: LLMeter uses reasoning deltas purely to
    time [`time_to_first_token`][llmeter.endpoints.base.InvocationResponse] and never adds them to
    `response_text`.

    Values are type-checked (non-empty `str`, or non-empty `list`/`tuple` for `thinking_blocks`)
    rather than merely tested for truthiness, so that sentinel or placeholder attribute values are
    not mistaken for real reasoning output.

    Args:
        delta: The `delta` from a streaming chunk choice. Both shapes work, so this is usable from
            custom endpoints that parse raw SSE JSON as well as from SDK-backed ones:

            * a **mapping**, read with `.get()` - e.g. `{"reasoning_content": "..."}`
            * an **object**, read with `getattr()` - plain objects, pydantic models, and models
              carrying the fields as permitted "extras" (both the `openai` and `litellm` delta
              types work, including fields their SDK version does not declare)

    Returns:
        `True` if the delta carries any recognized reasoning content.
    """
    for name in ("reasoning_content", "reasoning"):
        value = _get_delta_field(delta, name)
        if isinstance(value, str) and value:
            return True
    blocks = _get_delta_field(delta, "thinking_blocks")
    if isinstance(blocks, (list, tuple)) and blocks:
        return True
    return False


def infer_reasoning_visibility_from_model_id(model_id: str) -> ReasoningType | None:
    """Guess how a model discloses its reasoning, from provider naming in its ID.

    Several streaming schemas look identical whether the model streams its reasoning verbatim or
    only a summary of it, in which cases this cannot be detected from the responses alone. This
    helper attempts to infer a default from a model identifier - since usually reasoning behaviour
    is by provider, and model IDs are usually namespaced by provider such as -
    `anthropic.claude-opus-4-7`, `us.anthropic.claude-...`, `bedrock/anthropic.claude-...`,
    `openai.gpt-oss-120b-1:0`.

    * **Anthropic models** return *summarized* thinking on Claude 4 and later, so `"summary"` is
      assumed. This is wrong for Claude 3.7 Sonnet, which returns its full thinking output; declare
      `default_reasoning_visibility="verbatim"` explicitly for that model (or older ones).
    * **Everything else** (`gpt-oss`, Qwen, DeepSeek, ...) streams the reasoning tokens themselves,
      so `"verbatim"` is assumed.

    Matching is on whole `/`- and `.`-delimited segments, so a lookalike such as
    `acme.anthropic-compatible-v1` is not treated as Anthropic.

    Args:
        model_id: The model identifier, with or without provider/region prefixes.

    Returns:
        The assumed disclosure level, or `None` if `model_id` is not a string.
    """
    if not isinstance(model_id, str):
        return None
    segments = model_id.replace("/", ".").split(".")
    return "summary" if "anthropic" in segments else "verbatim"


def backfill_reasoning_type_from_token_counts(
    response: InvocationResponse,
    silent_reasoning_type: ReasoningType = "unknown",
) -> None:
    """Infer that reasoning happened from the *token accounting*, when no content revealed it.

    Detecting reasoning from streamed content only works if the provider streams some. Several
    prominent APIs bill for reasoning tokens while emitting nothing that identifies them:

    * **OpenAI Chat Completions** never returns reasoning content for the o-series or GPT-5
      reasoning models - only a `reasoning_tokens` figure in `usage`.
    * **OpenAI Responses** emits no reasoning events unless `reasoning.summary` was requested.
    * Any gateway that strips reasoning fields (or renames them past
      [`delta_has_reasoning_content`][llmeter.endpoints.base.delta_has_reasoning_content])
      while passing usage through.

    In cases where `num_tokens_output_reasoning` is positive, reasoning demonstrably happened - so
    this function checks if `reasoning_type=None` and if so backfills it to the given
    `silent_reasoning_type`: Preventing incorrect `time_per_output_token` calculations.

    Applied automatically by the
    [`Endpoint.llmeter_invoke`][llmeter.endpoints.base.Endpoint.llmeter_invoke] decorator after
    `process_raw_response`, using the endpoint's
    [`silent_reasoning_type`][llmeter.endpoints.base.Endpoint.silent_reasoning_type], so it covers
    custom endpoints too. Custom endpoints that bypass the decorator can call it directly.

    Args:
        response: The response to update in-place. Only modified when `reasoning_type` is unset
            *and* a positive reasoning-token count was reported; any value the endpoint established
            from the response content wins.
        silent_reasoning_type: What to record for this situation. `"redacted"` and `"unknown"` route
            TPOT identically, so the choice is purely about how confidently the outcome can be
            described - see
            [`Endpoint.silent_reasoning_type`][llmeter.endpoints.base.Endpoint.silent_reasoning_type].
    """
    if response.reasoning_type is not None:
        return
    reasoning_tokens = response.num_tokens_output_reasoning
    # Type-checked rather than truthiness-tested, so that a placeholder/mock attribute value cannot
    # be mistaken for a real count. `bool` is excluded because it is an `int` subclass.
    if isinstance(reasoning_tokens, bool) or not isinstance(reasoning_tokens, int):
        return
    if reasoning_tokens > 0:
        response.reasoning_type = silent_reasoning_type


TRawResponse = TypeVar("TRawResponse", bound=Any)


class Endpoint(Serializable, ABC, Generic[TRawResponse]):
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

    silent_reasoning_type: ReasoningType = "unknown"
    """What to record as `reasoning_type` when reasoning was billed but never disclosed.

    Read by [`llmeter_invoke`][llmeter.endpoints.base.Endpoint.llmeter_invoke] when it calls
    [`backfill_reasoning_type_from_token_counts`][llmeter.endpoints.base.backfill_reasoning_type_from_token_counts],
    for the case where the provider reported `num_tokens_output_reasoning > 0` but nothing in the
    response identified the reasoning. A **class**-level constant, not user configuration: it
    describes how much the connector's API guarantees, so it is not serialized with the endpoint.

    `"redacted"` and `"unknown"` route
    [`time_per_output_token`][llmeter.endpoints.base.InvocationResponse] identically (both use the
    answer-only pairing), so this choice only affects how the situation is *described* - and the
    right description depends on how completely the connector can parse its API:

    * **`"redacted"`** where silence is *documented, expected* behaviour of a self-describing API,
        so the reasoning was demonstrably withheld rather than merely unrecognized. Set on the
        OpenAI Responses endpoints, whose schema states the disclosure level outright (distinct
        event types when streaming, distinct item fields when not) - if none of those appear, the
        API withheld the reasoning. This also keeps them self-consistent, since a reasoning *item*
        that discloses neither content nor summary already yields `"redacted"`.
    * **`"unknown"`** (the default) where silence is *ambiguous*, because the connector cannot be
        sure it would have recognized the reasoning had it been sent. This is the honest answer for:

        - **OpenAI-compatible Chat Completions**, whose schema has no reasoning field at all;
          `reasoning_content` / `reasoning` are vendor extensions, so an unrecognized field name is
          indistinguishable from real withholding.
        - **LiteLLM**, which normalizes ~100 providers and may not normalize a given one.
        - **Anthropic Messages**, where every *documented* thinking mode leaves a structural trace,
          so reaching this fallback at all means something undocumented happened (a gateway
          stripping signature deltas, say) - which is worth surfacing as "go and check", not
          reporting as though it were expected.

    Custom endpoints should leave this as `"unknown"` unless their API structurally guarantees that
    reasoning content would have been identifiable.
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
            - `reasoning_type` (via
                [`backfill_reasoning_type_from_token_counts`][llmeter.endpoints.base.backfill_reasoning_type_from_token_counts],
                which catches providers that bill for reasoning tokens without streaming any
                reasoning content, using this class'
                [`silent_reasoning_type`][llmeter.endpoints.base.Endpoint.silent_reasoning_type])

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

            # Last, because it depends on `num_tokens_output_reasoning` being fully parsed:
            backfill_reasoning_type_from_token_counts(
                response, self.silent_reasoning_type
            )

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
        """Save the endpoint configuration to a JSON file.

        .. deprecated::
            Use :meth:`~llmeter.serialization.Serializable.save_to_file` instead, which
            provides the same behavior with a consistent name across all serializable
            LLMeter objects. This alias will be removed in a future major version.

        Args:
            output_path (str | UPath): The path where the configuration file will be saved.

        Returns:
            Path: The path the file was written to.
        """
        warnings.warn(
            "Endpoint.save() is deprecated and will be removed in a future version; "
            "use Endpoint.save_to_file() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.save_to_file(output_path)

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
    def load_from_file(cls, path: ReadablePathLike) -> "Endpoint":
        """Load an endpoint configuration from a JSON file.

        This class method reads a JSON file containing an endpoint configuration,
        determines the appropriate endpoint class, and instantiates it with the
        loaded configuration.

        Args:
            path (str | UPath): The path to the JSON configuration file.

        Returns:
            Endpoint: An instance of the appropriate endpoint class, initialized
                      with the configuration from the file.
        """
        path = ensure_path(path)
        with path.open("r") as f:
            data = json.load(f)
        if "__llmeter_class__" in data:
            return load_object(data)
        endpoint_type = data.pop("endpoint_type")
        endpoint_module = importlib.import_module("llmeter.endpoints")
        endpoint_class = getattr(endpoint_module, endpoint_type)
        return endpoint_class(**_filter_legacy_ctor_kwargs(endpoint_class, data))

    @classmethod
    def load(cls, endpoint_config: dict) -> "Endpoint":  # type: ignore
        """Load an endpoint configuration from a dictionary.

        This class method reads a dictionary containing an endpoint configuration,
        determines the appropriate endpoint class, and instantiates it with the
        loaded configuration.

        !!! warning "Deprecated"
            This supports the legacy `{"endpoint_type": ...}` format. New code should
            use [`load_object`][llmeter.serialization.load_object] with dicts produced by
            [`dump_object`][llmeter.serialization.dump_object].

        Args:
            endpoint_config (dict): A dictionary containing the endpoint configuration.
                Must include at minimum an `endpoint_type` key.

        Returns:
            Endpoint: An instance of the appropriate endpoint class, initialized
                      with the configuration from the dictionary.
        """
        endpoint_type = endpoint_config.pop("endpoint_type")
        endpoint_module = importlib.import_module("llmeter.endpoints")
        endpoint_class = getattr(endpoint_module, endpoint_type)
        return endpoint_class(
            **_filter_legacy_ctor_kwargs(endpoint_class, endpoint_config)
        )


def _filter_legacy_ctor_kwargs(endpoint_class: type, config: dict) -> dict:
    """Drop legacy config keys that the target endpoint constructor won't accept.

    Older LLMeter configs persisted derived/read-only attributes (notably `provider`, which
    endpoints now set internally) alongside the real constructor arguments. Passing those through
    to a modern `__init__` raises `TypeError`, so we filter the dict down to the parameters the
    constructor actually declares.

    If the constructor accepts `**kwargs` the dict is passed through unchanged.

    Args:
        endpoint_class: The endpoint class about to be instantiated.
        config: The legacy configuration dict (already stripped of ``endpoint_type``).

    Returns:
        A copy of `config` containing only keys the constructor accepts.
    """
    params = inspect.signature(endpoint_class.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return config
    accepted = {name for name in params if name != "self"}
    dropped = set(config) - accepted
    if dropped:
        logger.debug(
            "Ignoring legacy config field(s) not accepted by %s.__init__: %s",
            endpoint_class.__name__,
            ", ".join(sorted(dropped)),
        )
    return {k: v for k, v in config.items() if k in accepted}
