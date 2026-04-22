# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for InvocationResponse.to_dict() JSON serializability.

The to_dict() method must return a plain dict that can be passed through
json.dumps() without a custom ``default`` handler.  This is critical for
Lambda responses and any other code path that does not use
llmeter_default_serializer.

See: https://github.com/awslabs/llmeter/issues/67
"""

import json
from datetime import datetime, timezone

from llmeter.endpoints.base import InvocationResponse


class TestInvocationResponseToDict:
    """Verify InvocationResponse.to_dict() produces JSON-serializable output."""

    def test_to_dict_is_json_serializable_with_request_time(self):
        """Regression: to_dict() with a datetime request_time must not raise
        'Object of type datetime is not JSON serializable'."""
        response = InvocationResponse(
            id="test-id",
            response_text="hello",
            request_time=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        data = response.to_dict()

        # This is the exact call that failed in Lambda marshaling
        serialized = json.dumps(data)
        assert isinstance(serialized, str)

        # Verify the datetime was converted to a string
        assert isinstance(data["request_time"], str)

    def test_to_dict_is_json_serializable_without_request_time(self):
        """to_dict() with request_time=None should also serialize cleanly."""
        response = InvocationResponse(
            id="test-id",
            response_text="hello",
            request_time=None,
        )

        data = response.to_dict()
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
        assert data["request_time"] is None

    def test_to_dict_preserves_datetime_value(self):
        """The serialized request_time should round-trip to the same instant."""
        dt = datetime(2025, 6, 15, 8, 30, 45, tzinfo=timezone.utc)
        response = InvocationResponse(
            id="test-id",
            response_text="hello",
            request_time=dt,
        )

        data = response.to_dict()
        parsed = datetime.fromisoformat(data["request_time"].replace("Z", "+00:00"))
        assert parsed == dt

    def test_to_dict_full_response_is_json_serializable(self):
        """A fully-populated InvocationResponse.to_dict() must be JSON-safe."""
        response = InvocationResponse(
            id="resp-001",
            response_text="The answer is 42.",
            input_payload={"messages": [{"role": "user", "content": "What is 6*7?"}]},
            input_prompt="What is 6*7?",
            time_to_first_token=0.15,
            time_to_last_token=0.85,
            num_tokens_input=12,
            num_tokens_output=5,
            num_tokens_input_cached=4,
            num_tokens_output_reasoning=2,
            time_per_output_token=0.14,
            error=None,
            retries=0,
            request_time=datetime.now(timezone.utc),
        )

        data = response.to_dict()
        serialized = json.dumps(data)
        assert isinstance(serialized, str)

    def test_to_dict_error_response_is_json_serializable(self):
        """Error responses created via error_output() must also serialize."""
        response = InvocationResponse.error_output(
            input_payload={"messages": []},
            error="Connection timeout",
            request_time=datetime.now(timezone.utc),
        )

        data = response.to_dict()
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
        assert isinstance(data["request_time"], str)
