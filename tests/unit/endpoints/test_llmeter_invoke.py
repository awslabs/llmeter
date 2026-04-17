# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the @llmeter_invoke decorator.

Validates the decorator's contract around payload handling, timing,
error handling, metadata back-fill, and invocation isolation.
"""

from datetime import datetime, timezone


from llmeter.endpoints.base import Endpoint, InvocationResponse, llmeter_invoke


# ---------------------------------------------------------------------------
# Minimal concrete endpoint for testing the decorator in isolation
# ---------------------------------------------------------------------------


class StubEndpoint(Endpoint):
    """Endpoint whose invoke body and side-effects are fully controllable."""

    def __init__(self, invoke_fn=None, parse_payload_fn=None):
        super().__init__(
            endpoint_name="stub",
            model_id="stub-model",
            provider="stub",
        )
        self._invoke_fn = invoke_fn
        self._parse_payload_fn = parse_payload_fn

    @llmeter_invoke
    def invoke(self, payload: dict) -> InvocationResponse:
        if self._invoke_fn:
            return self._invoke_fn(self, payload)
        return InvocationResponse(response_text="ok")

    def parse_response(self, raw_response, start_t):
        return InvocationResponse(response_text=str(raw_response))

    def _parse_payload(self, payload):
        if self._parse_payload_fn:
            return self._parse_payload_fn(payload)
        return super()._parse_payload(payload)


class StubWithPrepare(StubEndpoint):
    """Endpoint that injects a model field in prepare_payload."""

    def prepare_payload(self, payload, **kwargs):
        return {**kwargs, **payload, "model": self.model_id}


# ---------------------------------------------------------------------------
# Payload mutation tests
# ---------------------------------------------------------------------------


class TestPayloadMutation:
    def test_input_payload_reflects_mutations_by_api_client(self):
        """input_payload on the response should contain the dict as mutated
        by the API client, since that's what was actually sent."""

        def mutating_invoke(self, payload):
            # Simulate what boto3 does: pop keys, add metadata
            payload.pop("extra_field", None)
            payload["_injected_by_client"] = True
            return InvocationResponse(response_text="ok")

        endpoint = StubWithPrepare(invoke_fn=mutating_invoke)
        response = endpoint.invoke({"prompt": "hello", "extra_field": "will be popped"})

        assert response.error is None
        # The mutated payload is what gets saved
        assert "_injected_by_client" in response.input_payload
        assert "extra_field" not in response.input_payload

    def test_input_prompt_uses_pre_mutation_snapshot(self):
        """_parse_payload should receive the payload as it was *before* the
        API client mutated it, so prompt extraction is reliable."""

        parse_payload_received = {}

        def capture_parse_payload(payload):
            parse_payload_received.update(payload)
            return payload.get("prompt", "")

        def mutating_invoke(self, payload):
            # Wipe the prompt field — simulates a destructive client
            payload.pop("prompt", None)
            return InvocationResponse(response_text="ok")

        endpoint = StubWithPrepare(
            invoke_fn=mutating_invoke,
            parse_payload_fn=capture_parse_payload,
        )
        response = endpoint.invoke({"prompt": "hello"})

        assert response.error is None
        # _parse_payload saw the original prompt
        assert parse_payload_received["prompt"] == "hello"
        assert response.input_prompt == "hello"
        # But input_payload has it removed (mutated)
        assert "prompt" not in response.input_payload

    def test_original_caller_payload_is_not_mutated(self):
        """The caller's original dict must not be modified by prepare_payload
        or the API client."""

        def mutating_invoke(self, payload):
            payload["injected"] = True
            return InvocationResponse(response_text="ok")

        endpoint = StubWithPrepare(invoke_fn=mutating_invoke)
        original = {"prompt": "hello"}
        original_copy = original.copy()

        endpoint.invoke(original)

        # Caller's dict is unchanged (prepare_payload creates a new dict)
        assert original == original_copy


# ---------------------------------------------------------------------------
# Invocation isolation tests
# ---------------------------------------------------------------------------


class TestInvocationIsolation:
    def test_no_start_t_on_self(self):
        """The decorator must not store _start_t on the endpoint instance."""
        endpoint = StubEndpoint()
        endpoint.invoke({"prompt": "hello"})

        assert not hasattr(endpoint, "_start_t")

    def test_no_last_payload_on_self(self):
        """The decorator must not store _last_payload on the endpoint instance."""
        endpoint = StubEndpoint()
        endpoint.invoke({"prompt": "hello"})

        assert not hasattr(endpoint, "_last_payload")

    def test_consecutive_invocations_are_independent(self):
        """Each invocation should produce its own response with independent
        metadata — no state leaks between calls."""
        call_count = 0

        def counting_invoke(self, payload):
            nonlocal call_count
            call_count += 1
            return InvocationResponse(
                response_text=f"response-{call_count}",
                id=f"id-{call_count}",
            )

        endpoint = StubWithPrepare(invoke_fn=counting_invoke)

        r1 = endpoint.invoke({"prompt": "first"})
        r2 = endpoint.invoke({"prompt": "second"})

        assert r1.response_text == "response-1"
        assert r2.response_text == "response-2"
        assert r1.id == "id-1"
        assert r2.id == "id-2"
        assert r1.input_payload["prompt"] == "first"
        assert r2.input_payload["prompt"] == "second"
        # Each has its own request_time
        assert r1.request_time != r2.request_time or r1.request_time is not r2.request_time


# ---------------------------------------------------------------------------
# Timing tests
# ---------------------------------------------------------------------------


class TestTiming:
    def test_time_to_last_token_backfilled_for_non_streaming(self):
        """For non-streaming endpoints that don't set time_to_last_token,
        the decorator should back-fill it."""
        endpoint = StubEndpoint()
        response = endpoint.invoke({"prompt": "hello"})

        assert response.error is None
        assert response.time_to_last_token is not None
        assert response.time_to_last_token > 0

    def test_time_to_last_token_not_overwritten_if_set(self):
        """If the inner invoke sets time_to_last_token (streaming), the
        decorator should not overwrite it."""

        def streaming_invoke(self, payload):
            return InvocationResponse(
                response_text="streamed",
                time_to_first_token=0.1,
                time_to_last_token=0.5,
            )

        endpoint = StubEndpoint(invoke_fn=streaming_invoke)
        response = endpoint.invoke({"prompt": "hello"})

        assert response.time_to_last_token == 0.5
        assert response.time_to_first_token == 0.1

    def test_time_to_last_token_not_set_on_error(self):
        """On error responses, time_to_last_token should remain None."""

        def failing_invoke(self, payload):
            raise RuntimeError("boom")

        endpoint = StubEndpoint(invoke_fn=failing_invoke)
        response = endpoint.invoke({"prompt": "hello"})

        assert response.error is not None
        assert response.time_to_last_token is None


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_exception_converted_to_error_response(self):
        """Exceptions in invoke should be caught and converted to an error
        InvocationResponse."""

        def failing_invoke(self, payload):
            raise ValueError("bad input")

        endpoint = StubEndpoint(invoke_fn=failing_invoke)
        response = endpoint.invoke({"prompt": "hello"})

        assert response.error is not None
        assert "bad input" in response.error
        assert response.response_text is None
        assert response.input_payload is not None
        assert response.id is not None
        assert response.request_time is not None

    def test_parse_payload_failure_does_not_crash(self):
        """If _parse_payload raises, input_prompt should be None but the
        response should still be valid."""

        def broken_parse(payload):
            raise KeyError("missing field")

        endpoint = StubEndpoint(parse_payload_fn=broken_parse)
        response = endpoint.invoke({"prompt": "hello"})

        assert response.error is None
        assert response.input_prompt is None
        assert response.response_text == "ok"


# ---------------------------------------------------------------------------
# Metadata back-fill tests
# ---------------------------------------------------------------------------


class TestMetadataBackfill:
    def test_id_backfilled_when_not_set(self):
        endpoint = StubEndpoint()
        response = endpoint.invoke({"prompt": "hello"})

        assert response.id is not None
        assert len(response.id) > 0

    def test_id_not_overwritten_when_set(self):
        def invoke_with_id(self, payload):
            return InvocationResponse(response_text="ok", id="my-custom-id")

        endpoint = StubEndpoint(invoke_fn=invoke_with_id)
        response = endpoint.invoke({"prompt": "hello"})

        assert response.id == "my-custom-id"

    def test_request_time_is_utc_datetime(self):
        endpoint = StubEndpoint()
        response = endpoint.invoke({"prompt": "hello"})

        assert isinstance(response.request_time, datetime)
        assert response.request_time.tzinfo == timezone.utc

    def test_input_payload_backfilled(self):
        endpoint = StubWithPrepare()
        response = endpoint.invoke({"prompt": "hello"})

        assert response.input_payload is not None
        assert response.input_payload["model"] == "stub-model"
        assert response.input_payload["prompt"] == "hello"

    def test_input_payload_not_overwritten_when_set(self):
        custom_payload = {"custom": True}

        def invoke_with_payload(self, payload):
            return InvocationResponse(
                response_text="ok", input_payload=custom_payload
            )

        endpoint = StubEndpoint(invoke_fn=invoke_with_payload)
        response = endpoint.invoke({"prompt": "hello"})

        assert response.input_payload is custom_payload


# ---------------------------------------------------------------------------
# Decorator marker test
# ---------------------------------------------------------------------------


class TestDecoratorMarker:
    def test_decorated_function_has_marker(self):
        assert getattr(StubEndpoint.invoke, "_is_llmeter_invoke", False) is True

    def test_undecorated_function_has_no_marker(self):
        def plain_invoke(self, payload):
            pass

        assert not hasattr(plain_invoke, "_is_llmeter_invoke")
