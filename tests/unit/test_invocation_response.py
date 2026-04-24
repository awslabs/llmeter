# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for InvocationResponse"""

import json
from datetime import datetime, timedelta, timezone

from llmeter.endpoints.base import InvocationResponse
from llmeter.json_utils import llmeter_default_serializer


class TestToJson:
    def test_full_response_to_json(self):
        """A fully-populated InvocationResponse must serialize via to_json()."""
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
            request_time=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        serialized = response.to_json()
        assert isinstance(serialized, str)

        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)
        assert parsed["id"] == "resp-001"
        assert parsed["response_text"] == "The answer is 42."
        assert isinstance(parsed["input_payload"], dict)
        assert parsed["input_prompt"] == "What is 6*7?"
        assert parsed["time_to_first_token"] == 0.15
        assert parsed["time_to_last_token"] == 0.85
        assert parsed["num_tokens_input"] == 12
        assert parsed["num_tokens_output"] == 5
        assert parsed["num_tokens_input_cached"] == 4
        assert parsed["num_tokens_output_reasoning"] == 2
        assert parsed["time_per_output_token"] == 0.14
        assert parsed["error"] is None
        assert parsed["retries"] == 0
        # ISO with 'Z' suffix:
        assert parsed["request_time"] == "2026-01-01T12:00:00Z"

    def test_error_response_to_json(self):
        """Error responses with request_time must also serialize cleanly."""
        response = InvocationResponse.error_output(
            input_payload={"messages": []},
            error="Connection timeout",
            request_time=datetime.now(timezone.utc),
        )
        parsed = json.loads(response.to_json())
        assert isinstance(parsed, dict)
        assert parsed["error"] == "Connection timeout"

    def test_save_request_time_with_offset(self):
        """Maps local timestamps to UTC in saved files"""
        response = InvocationResponse(
            response_text="hi",
            request_time=datetime(
                2026, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=8))
            ),
        )
        parsed = json.loads(response.to_json())
        assert parsed["request_time"] == "2026-01-01T04:00:00Z"


class TestFromJson:
    def test_round_trip_minimal(self):
        original = InvocationResponse(response_text="hello")
        restored = InvocationResponse.from_json(original.to_json())
        assert restored.response_text == "hello"
        assert restored.request_time is None

    def test_round_trip_error_response(self):
        dt = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
        original = InvocationResponse.error_output(
            input_payload={"messages": []},
            error="Connection timeout",
            request_time=dt,
        )
        restored = InvocationResponse.from_json(original.to_json())
        assert restored.error == "Connection timeout"
        assert restored.response_text is None
        assert isinstance(restored.request_time, datetime)
        assert restored.request_time == dt

    def test_round_trip_all_fields(self):
        dt = datetime(2026, 6, 15, 8, 30, 45, tzinfo=timezone.utc)
        original = InvocationResponse(
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
            request_time=dt,
        )
        restored = InvocationResponse.from_json(original.to_json())

        assert restored.id == original.id
        assert restored.response_text == original.response_text
        assert restored.input_payload == original.input_payload
        assert restored.input_prompt == original.input_prompt
        assert restored.time_to_first_token == original.time_to_first_token
        assert restored.time_to_last_token == original.time_to_last_token
        assert restored.num_tokens_input == original.num_tokens_input
        assert restored.num_tokens_output == original.num_tokens_output
        assert restored.num_tokens_input_cached == original.num_tokens_input_cached
        assert (
            restored.num_tokens_output_reasoning == original.num_tokens_output_reasoning
        )
        assert restored.time_per_output_token == original.time_per_output_token
        assert restored.error == original.error
        assert restored.retries == original.retries
        assert restored.request_time == dt
        assert isinstance(restored.request_time, datetime)

    def test_load_request_time_with_offset(self):
        json_str = json.dumps(
            {"response_text": "hi", "request_time": "2026-01-01T14:00:00+02:00"},
            default=llmeter_default_serializer,
        )
        restored = InvocationResponse.from_json(json_str)
        assert isinstance(restored.request_time, datetime)
        assert restored.request_time == datetime(
            2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc
        )

    def test_round_trip_with_binary_payload(self):
        original = InvocationResponse(
            response_text="A cat",
            input_payload={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "What is this?"},
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {"bytes": b"\xff\xd8\xff\xe0"},
                                }
                            },
                        ],
                    }
                ]
            },
        )
        restored = InvocationResponse.from_json(original.to_json())
        source_bytes = restored.input_payload["messages"][0]["content"][1]["image"][
            "source"
        ]["bytes"]
        assert isinstance(source_bytes, bytes)
        assert source_bytes == b"\xff\xd8\xff\xe0"

    def test_no_bytes_marker_without_binary(self):
        original = InvocationResponse(
            response_text="hi",
            input_payload={"prompt": "hello"},
        )
        restored = InvocationResponse.from_json(original.to_json())
        assert restored.input_payload == {"prompt": "hello"}
