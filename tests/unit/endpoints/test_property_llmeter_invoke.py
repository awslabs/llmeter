# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for the llmeter_invoke decorator.

Covers the in-place process_raw_response contract introduced by the refactor
in 09f080aa ("refactor(endpoint): process_raw_response"):

- Partial data survives when process_raw_response raises mid-stream
- The wrapper correctly backfills missing fields (id, request_time, ttlt, etc.)
- prepare_payload transforms are reflected in input_payload without mutating
  the caller's original dict
"""

from datetime import datetime, timezone

from hypothesis import given
from hypothesis import strategies as st

from llmeter.endpoints.base import Endpoint

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

non_negative_ints = st.integers(min_value=0, max_value=100_000)
optional_non_negative_ints = st.one_of(st.none(), non_negative_ints)
optional_positive_floats = st.one_of(
    st.none(), st.floats(min_value=0.0, max_value=120.0, allow_nan=False)
)


# ---------------------------------------------------------------------------
# Stub endpoint for testing the wrapper in isolation
# ---------------------------------------------------------------------------


class _StubEndpoint(Endpoint):
    """Minimal endpoint whose behavior is fully controllable via callbacks."""

    def __init__(self, invoke_fn=None, process_fn=None, prepare_fn=None):
        super().__init__(endpoint_name="stub", model_id="stub-model", provider="stub")
        self._invoke_fn = invoke_fn
        self._process_fn = process_fn
        self._prepare_fn = prepare_fn

    @Endpoint.llmeter_invoke
    def invoke(self, payload: dict):
        if self._invoke_fn:
            return self._invoke_fn(self, payload)
        return {"text": "ok"}

    def process_raw_response(self, raw_response, start_t, response):
        if self._process_fn:
            self._process_fn(raw_response, start_t, response)
            return
        if isinstance(raw_response, dict):
            response.response_text = raw_response.get("text")

    def prepare_payload(self, payload):
        if self._prepare_fn:
            return self._prepare_fn(payload)
        return payload


# ===================================================================
# Partial data preservation on mid-stream errors
# ===================================================================


class TestPartialDataPreservation:
    """The core motivation of the refactor: fields set *before* an error
    in process_raw_response must survive on the returned InvocationResponse."""

    @given(
        partial_id=st.text(min_size=1, max_size=40),
        partial_text=st.text(min_size=0, max_size=200),
        partial_ttft=optional_positive_floats,
        partial_input_tokens=optional_non_negative_ints,
    )
    def test_fields_set_before_error_are_preserved(
        self, partial_id, partial_text, partial_ttft, partial_input_tokens
    ):
        """When process_raw_response sets some fields then raises, those
        fields must still be present on the final response."""

        def process_then_explode(_raw, _start_t, response):
            response.id = partial_id
            response.response_text = partial_text
            response.time_to_first_token = partial_ttft
            response.num_tokens_input = partial_input_tokens
            raise RuntimeError("stream interrupted")

        endpoint = _StubEndpoint(process_fn=process_then_explode)
        result = endpoint.invoke({"prompt": "hello"})

        assert result.error is not None
        assert "stream interrupted" in result.error
        # Partial fields survive:
        assert result.id == partial_id
        assert result.response_text == partial_text
        assert result.time_to_first_token == partial_ttft
        assert result.num_tokens_input == partial_input_tokens

    @given(
        partial_id=st.text(min_size=1, max_size=40),
        error_msg=st.text(min_size=1, max_size=100),
    )
    def test_process_raw_response_can_set_error_before_raising(
        self, partial_id, error_msg
    ):
        """If process_raw_response sets response.error *and* raises, the
        wrapper should keep the explicitly-set error (not overwrite it)."""

        def set_error_then_raise(_raw, _start_t, response):
            response.id = partial_id
            response.error = error_msg
            raise RuntimeError("this should be ignored")

        endpoint = _StubEndpoint(process_fn=set_error_then_raise)
        result = endpoint.invoke({"prompt": "hello"})

        assert result.error == error_msg  # Not "this should be ignored"
        assert result.id == partial_id


# ===================================================================
# llmeter_invoke wrapper backfill properties
# ===================================================================


class TestLlmeterInvokeBackfill:
    """Property tests for the automatic field backfill in the wrapper."""

    @given(payload=st.fixed_dictionaries({"prompt": st.text(min_size=1, max_size=200)}))
    def test_id_always_present(self, payload):
        """Every response must have a non-empty id, even if process_raw_response
        doesn't set one."""

        def no_id_process(_raw, _start_t, response):
            response.response_text = "ok"

        endpoint = _StubEndpoint(process_fn=no_id_process)
        result = endpoint.invoke(payload)
        assert result.id is not None
        assert len(result.id) > 0

    @given(payload=st.fixed_dictionaries({"prompt": st.text(min_size=1, max_size=200)}))
    def test_request_time_always_utc(self, payload):
        """request_time must always be a timezone-aware UTC datetime."""
        endpoint = _StubEndpoint()
        result = endpoint.invoke(payload)
        assert isinstance(result.request_time, datetime)
        assert result.request_time.tzinfo == timezone.utc

    @given(payload=st.fixed_dictionaries({"prompt": st.text(min_size=1, max_size=200)}))
    def test_ttlt_backfilled_on_success(self, payload):
        """On success, time_to_last_token is backfilled if not set by
        process_raw_response."""

        def no_timing_process(_raw, _start_t, response):
            response.response_text = "ok"

        endpoint = _StubEndpoint(process_fn=no_timing_process)
        result = endpoint.invoke(payload)
        assert result.error is None
        assert result.time_to_last_token is not None
        assert result.time_to_last_token > 0

    @given(
        ttlt=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    )
    def test_ttlt_not_overwritten_when_set(self, ttlt):
        """If process_raw_response sets time_to_last_token, the wrapper
        must not overwrite it."""

        def set_ttlt(_raw, _start_t, response):
            response.response_text = "ok"
            response.time_to_last_token = ttlt

        endpoint = _StubEndpoint(process_fn=set_ttlt)
        result = endpoint.invoke({"prompt": "hello"})
        assert result.time_to_last_token == ttlt

    @given(payload=st.fixed_dictionaries({"prompt": st.text(min_size=1, max_size=200)}))
    def test_ttlt_not_set_on_error(self, payload):
        """On error, time_to_last_token should remain None (unless
        process_raw_response set it before the error)."""

        def explode(_self, _payload):
            raise RuntimeError("boom")

        endpoint = _StubEndpoint(invoke_fn=explode)
        result = endpoint.invoke(payload)
        assert result.error is not None
        assert result.time_to_last_token is None

    @given(payload=st.fixed_dictionaries({"prompt": st.text(min_size=1, max_size=200)}))
    def test_input_payload_backfilled(self, payload):
        """input_payload should be the prepared payload when not set by
        process_raw_response."""
        endpoint = _StubEndpoint()
        result = endpoint.invoke(payload)
        assert result.input_payload is not None
        assert result.input_payload == payload

    @given(
        custom_id=st.text(min_size=1, max_size=40),
    )
    def test_id_cleared_by_process_is_restored(self, custom_id):
        """If process_raw_response accidentally sets id to None, the wrapper
        restores the default UUID."""

        def clear_id(_raw, _start_t, response):
            response.response_text = "ok"
            response.id = None  # Oops

        endpoint = _StubEndpoint(process_fn=clear_id)
        result = endpoint.invoke({"prompt": "hello"})
        # Wrapper restores the default UUID
        assert result.id is not None
        assert len(result.id) == 32  # UUID hex


# ===================================================================
# prepare_payload contract (no **kwargs)
# ===================================================================


class TestPreparePayloadContract:
    """The refactored prepare_payload accepts only (self, payload) — no **kwargs.

    Verify that the wrapper calls prepare_payload with the payload only, and
    that the prepared payload is what gets sent to invoke and saved.
    """

    @given(
        payload=st.fixed_dictionaries(
            {
                "prompt": st.text(min_size=1, max_size=200),
                "max_tokens": st.integers(min_value=1, max_value=4096),
            }
        ),
        injected_key=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(
                whitelist_categories=("L",),
            ),
        ),
        injected_val=st.text(min_size=1, max_size=20),
    )
    def test_prepare_payload_transforms_are_reflected(
        self, payload, injected_key, injected_val
    ):
        """Fields added by prepare_payload must appear in input_payload."""

        def add_field(p):
            return {**p, injected_key: injected_val}

        endpoint = _StubEndpoint(prepare_fn=add_field)
        result = endpoint.invoke(payload)

        assert result.input_payload[injected_key] == injected_val
        # Original fields still present
        assert result.input_payload["prompt"] == payload["prompt"]

    @given(
        payload=st.fixed_dictionaries(
            {
                "prompt": st.text(min_size=1, max_size=200),
            }
        ),
    )
    def test_original_payload_not_mutated(self, payload):
        """The caller's original dict must not be modified."""
        original_copy = payload.copy()

        def mutating_prepare(p):
            return {**p, "model": "injected"}

        endpoint = _StubEndpoint(prepare_fn=mutating_prepare)
        endpoint.invoke(payload)

        assert payload == original_copy
