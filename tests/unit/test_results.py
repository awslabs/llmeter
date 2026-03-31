# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from upath import UPath

from llmeter.endpoints.base import InvocationResponse
from llmeter.results import Result, _get_run_stats, _get_stats_from_results

sample_responses_successful = [
    InvocationResponse(
        response_text="test response",
        time_to_first_token=k[0],
        time_to_last_token=k[1],
        num_tokens_input=k[2],
        num_tokens_output=k[3],
        input_prompt="test input prompt",
    )
    for k in [
        [132, 236, 106, 104],
        [89, 230, 122, 8],
        [184, 187, 256, 183],
        [51, 247, 269, 171],
        [13, 126, 293, 168],
        [33, 245, 164, 109],
        [41, 231, 266, 131],
        [71, 206, 1, 31],
        [124, 179, 134, 19],
        [218, 230, 239, 265],
    ]
]
response_error = InvocationResponse(error="this is an error", response_text="")

## testing for `_get_test_stats()`


def test_get_test_stats():
    # Test case 1: No errors, non-zero test time
    responses = sample_responses_successful
    result = Result(
        responses=responses,
        total_requests=10,
        clients=2,
        n_requests=5,
        total_test_time=100,
    )
    stats = _get_run_stats(result)

    assert stats["failed_requests"] == 0
    assert stats["failed_requests_rate"] == 0
    assert pytest.approx(stats["requests_per_minute"], 0.01) == 6

    # Test case 2: Some errors, non-zero test time
    responses = sample_responses_successful[:8] + [response_error] * 2
    result = Result(
        responses=responses,
        total_requests=10,
        clients=2,
        n_requests=5,
        total_test_time=100,
    )
    stats = _get_run_stats(result)

    assert stats["failed_requests"] == 2
    assert pytest.approx(stats["failed_requests_rate"], 0.01) == 0.2
    assert pytest.approx(stats["requests_per_minute"], 0.01) == 6

    # Test case 3: All errors, non-zero test time
    responses = [response_error] * 5
    result = Result(
        responses=responses,
        total_requests=5,
        clients=1,
        n_requests=5,
        total_test_time=10,
    )
    stats = _get_run_stats(result)

    assert stats["failed_requests"] == 5
    assert stats["failed_requests_rate"] == 1
    assert pytest.approx(stats["requests_per_minute"], 0.01) == 30

    # Test case 4: No errors, zero test time
    responses = sample_responses_successful[:3]
    result = Result(
        responses=responses,
        total_requests=3,
        clients=1,
        n_requests=3,
        total_test_time=0,
    )
    stats = _get_run_stats(result)

    assert stats["failed_requests"] == 0
    assert stats["failed_requests_rate"] == 0
    assert stats["requests_per_minute"] == 0  # Avoid division by zero


# ## testing `_get_stats_from_results()`
test_metrics = [
    "time_to_last_token",
    "time_to_first_token",
    "num_tokens_output",
    "num_tokens_input",
]


def test_get_stats_from_results_with_result_object():
    responses = sample_responses_successful
    result = Result(
        clients=5,
        n_requests=100,
        responses=responses,
        total_requests=5,
        total_test_time=10,
    )

    stats = _get_stats_from_results(result, metrics=test_metrics)

    assert "time_to_last_token" in stats
    assert "time_to_first_token" in stats
    assert "num_tokens_output" in stats
    assert "num_tokens_input" in stats


def test_get_stats_from_results_with_no_metrics():
    responses = sample_responses_successful
    result = Result(
        clients=5,
        n_requests=100,
        responses=responses,
        total_requests=5,
        total_test_time=10,
    )

    stats = _get_stats_from_results(result, metrics=[])
    assert stats == {}


@pytest.fixture
def sample_result():
    responses = [
        InvocationResponse(
            id=f"test_{i}",
            response_text=f"Response {i}",
            input_prompt=f"Prompt {i}",
            time_to_first_token=0.1 * i,
            time_to_last_token=0.2 * i,
            num_tokens_output=10 * i,
            num_tokens_input=5 * i,
        )
        for i in range(1, 6)
    ]
    return Result(
        responses=responses,
        total_requests=5,
        clients=1,
        n_requests=5,
        total_test_time=1,
        run_name="Test Run",
    )


def test_stats_property(sample_result: Result):
    stats = sample_result.stats

    # Test basic information
    assert stats["total_requests"] == 5

    # Test aggregated statistics
    assert "time_to_last_token-p50" in stats
    assert "time_to_first_token-average" in stats
    assert "num_tokens_output-p90" in stats
    assert "num_tokens_input-p99" in stats

    # Test specific values (you may need to adjust these based on your exact implementation)
    assert stats["time_to_last_token-average"] == pytest.approx(0.6)
    assert stats["time_to_first_token-p50"] == pytest.approx(0.3)
    assert stats["num_tokens_output-average"] == 30
    assert stats["num_tokens_input-average"] == 15

    # Test test-specific statistics
    assert "failed_requests" in stats
    assert "failed_requests_rate" in stats
    assert "requests_per_minute" in stats

    # Test that all keys from to_dict() are present
    for key in sample_result.to_dict().keys():
        assert key in stats

    # Test caching returns same object for built-in stats:
    assert sample_result._builtin_stats is sample_result._builtin_stats


def test_stats_property_empty_result():
    empty_result = Result(
        responses=[], total_requests=0, clients=0, n_requests=0, total_test_time=0
    )
    stats = empty_result.stats

    assert stats["total_requests"] == 0
    assert stats["failed_requests"] == 0
    assert stats["failed_requests_rate"] == 0
    assert stats["requests_per_minute"] == 0

    # Check that no errors are raised for empty data
    for metric in [
        "time_to_last_token",
        "time_to_first_token",
        "num_tokens_output",
        "num_tokens_input",
    ]:
        for stat in ["p50", "p90", "p99", "average"]:
            assert f"{metric}-{stat}" not in stats


def test_stats_json_serializable_with_datetimes():
    """stats dict should be directly JSON-serializable even with datetime fields."""
    from datetime import datetime, timezone

    result = Result(
        responses=sample_responses_successful[:2],
        total_requests=2,
        clients=1,
        n_requests=2,
        total_test_time=1.0,
        start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    stats = result.stats
    # Should not raise TypeError
    json_str = json.dumps(stats)
    parsed = json.loads(json_str)
    assert parsed["start_time"] == "2025-01-01T12:00:00Z"
    assert parsed["end_time"] == "2025-01-01T12:00:01Z"


@pytest.fixture
def temp_dir(tmp_path: Path):
    return UPath(tmp_path)


def test_save_method(sample_result: Result, temp_dir: UPath):
    output_path = temp_dir / "test_output"
    sample_result.save(output_path)

    # Check if files are created
    assert (output_path / "summary.json").exists()
    assert (output_path / "stats.json").exists()
    assert (output_path / "responses.jsonl").exists()

    # Check content of summary.json
    with (output_path / "summary.json").open() as f:
        summary = json.load(f)
        assert summary["total_requests"] == 5
        assert summary["run_name"] == "Test Run"

    # Check content of stats.json
    with (output_path / "stats.json").open() as f:
        stats = json.load(f)
        assert "total_requests" in stats
        assert "time_to_last_token-average" in stats

    # Check content of responses.jsonl
    with (output_path / "responses.jsonl").open() as f:
        responses = [json.loads(line) for line in f]
        assert len(responses) == 5
        assert all(isinstance(r["id"], str) for r in responses)


def test_load_method(sample_result: Result, temp_dir: UPath):
    output_path = temp_dir / "test_output"
    sample_result.save(output_path)

    # Load the saved result
    loaded_result = Result.load(output_path)

    # Check if loaded result matches the original
    assert loaded_result.total_requests == sample_result.total_requests
    assert loaded_result.run_name == sample_result.run_name
    assert len(loaded_result.responses) == len(sample_result.responses)

    # Check if responses are correctly loaded
    for orig, loaded in zip(sample_result.responses, loaded_result.responses):
        assert orig.id == loaded.id
        assert orig.response_text == loaded.response_text
        assert orig.input_prompt == loaded.input_prompt
        assert orig.time_to_first_token == loaded.time_to_first_token
        assert orig.time_to_last_token == loaded.time_to_last_token
        assert orig.num_tokens_output == loaded.num_tokens_output
        assert orig.num_tokens_input == loaded.num_tokens_input


def test_save_method_no_output_path(sample_result: Result):
    with pytest.raises(ValueError, match="No output path provided"):
        sample_result.save()


def test_load_method_missing_files(temp_dir: UPath):
    with pytest.raises(FileNotFoundError):
        Result.load(temp_dir / "non_existent_directory")


def test_save_and_load_with_string_path(sample_result: Result, temp_dir: UPath):
    output_path = str(temp_dir / "test_output")
    sample_result.save(output_path)
    loaded_result = Result.load(output_path)
    assert loaded_result.total_requests == sample_result.total_requests


def test_save_method_existing_responses(sample_result: Result, temp_dir: UPath):
    output_path = temp_dir / "test_output"
    sample_result.save(output_path)

    # Modify the responses file
    with (output_path / "responses.jsonl").open("a") as f:
        f.write(json.dumps({"id": "extra_response"}) + "\n")

    # Save again
    sample_result.save(output_path)

    # Check that the responses file wasn't overwritten
    with (output_path / "responses.jsonl").open() as f:
        responses = [json.loads(line) for line in f]
        assert len(responses) == 6  # 5 original + 1 extra
        assert responses[-1]["id"] == "extra_response"


# Tests for LLMeterEncoder
# Validates Requirements: 7.1, 7.2


def test_llmeter_encoder_handles_bytes():
    """Test that LLMeterEncoder handles bytes objects via base64 marker.
    
    Validates: Requirements 7.1, 7.2
    """
    from llmeter.json_utils import LLMeterEncoder
    
    # Test bytes object encoding
    payload = {"image": {"bytes": b"\xff\xd8\xff\xe0"}}
    encoded = json.dumps(payload, cls=LLMeterEncoder)
    
    # Verify marker object is present
    assert "__llmeter_bytes__" in encoded
    
    # Verify it can be decoded back
    decoded = json.loads(encoded, object_hook=lambda d: d)
    assert decoded["image"]["bytes"]["__llmeter_bytes__"] == "/9j/4A=="


def test_llmeter_encoder_str_fallback():
    """Test that LLMeterEncoder falls back to str() for custom objects.
    
    Validates: Requirements 7.1, 7.2
    """
    from llmeter.json_utils import LLMeterEncoder
    
    # Create a custom object with __str__ method
    class CustomObject:
        def __str__(self):
            return "custom_string_representation"
    
    payload = {"custom": CustomObject()}
    encoded = json.dumps(payload, cls=LLMeterEncoder)
    
    # Verify str() fallback was used
    decoded = json.loads(encoded)
    assert decoded["custom"] == "custom_string_representation"


def test_llmeter_encoder_none_on_str_failure():
    """Test that LLMeterEncoder returns None when str() conversion fails.
    
    Validates: Requirements 7.1, 7.2
    """
    from llmeter.json_utils import LLMeterEncoder
    
    # Create a custom object that raises exception in __str__
    class FailingObject:
        def __str__(self):
            raise RuntimeError("Cannot convert to string")
    
    payload = {"failing": FailingObject()}
    encoded = json.dumps(payload, cls=LLMeterEncoder)
    
    # Verify None was returned
    decoded = json.loads(encoded)
    assert decoded["failing"] is None


def test_llmeter_encoder_mixed_types():
    """Test that LLMeterEncoder handles mixed types correctly.
    
    Validates: Requirements 7.1, 7.2
    """
    from llmeter.json_utils import LLMeterEncoder
    
    class CustomObject:
        def __str__(self):
            return "custom"
    
    # Mix of bytes, custom objects, and standard types
    payload = {
        "bytes_field": b"\x00\x01\x02",
        "custom_field": CustomObject(),
        "string_field": "normal string",
        "int_field": 42,
        "nested": {
            "bytes": b"\xff\xfe",
            "custom": CustomObject()
        }
    }
    
    encoded = json.dumps(payload, cls=LLMeterEncoder)
    decoded = json.loads(encoded)
    
    # Verify bytes were encoded with marker
    assert "__llmeter_bytes__" in encoded
    
    # Verify custom objects were converted to strings
    assert decoded["custom_field"] == "custom"
    assert decoded["nested"]["custom"] == "custom"
    
    # Verify standard types remain unchanged
    assert decoded["string_field"] == "normal string"
    assert decoded["int_field"] == 42


# Tests for InvocationResponse.to_json
# Validates Requirements: 7.1, 7.2, 7.3


def test_invocation_response_to_json_with_binary_content():
    """Test InvocationResponse.to_json with binary content in input_payload.
    
    Validates: Requirements 7.1, 7.2
    """
    # Create InvocationResponse with binary content in input_payload
    response = InvocationResponse(
        response_text="This is an image of a cat",
        input_payload={
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "messages": [{
                "role": "user",
                "content": [
                    {"text": "What is in this image?"},
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": b"\xff\xd8\xff\xe0\x00\x10JFIF"}
                        }
                    }
                ]
            }]
        },
        id="test-123",
        time_to_first_token=0.5,
        time_to_last_token=1.2,
        num_tokens_input=15,
        num_tokens_output=8
    )
    
    # Serialize to JSON
    json_str = response.to_json()
    
    # Verify it's valid JSON
    parsed = json.loads(json_str)
    
    # Verify marker object is present for bytes
    assert "__llmeter_bytes__" in json_str
    
    # Verify structure is preserved
    assert parsed["response_text"] == "This is an image of a cat"
    assert parsed["id"] == "test-123"
    assert parsed["input_payload"]["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Verify bytes were encoded with marker
    bytes_marker = parsed["input_payload"]["messages"][0]["content"][1]["image"]["source"]["bytes"]
    assert "__llmeter_bytes__" in bytes_marker
    assert bytes_marker["__llmeter_bytes__"] == "/9j/4AAQSkZJRg=="


def test_invocation_response_to_json_valid_json_output():
    """Test that InvocationResponse.to_json produces valid JSON.
    
    Validates: Requirements 7.1, 7.2
    """
    response = InvocationResponse(
        response_text="Test response",
        input_payload={
            "image_data": b"\x89PNG\r\n\x1a\n",
            "text": "Analyze this image"
        },
        id="test-456"
    )
    
    # Serialize to JSON
    json_str = response.to_json()
    
    # Verify it's valid JSON (should not raise exception)
    parsed = json.loads(json_str)
    
    # Verify all fields are present
    assert "response_text" in parsed
    assert "input_payload" in parsed
    assert "id" in parsed
    
    # Verify bytes were properly encoded
    assert isinstance(parsed["input_payload"]["image_data"], dict)
    assert "__llmeter_bytes__" in parsed["input_payload"]["image_data"]


def test_invocation_response_to_json_round_trip():
    """Test round-trip serialization/deserialization with InvocationResponse.
    
    Validates: Requirements 7.1, 7.2, 7.3
    """
    from llmeter.json_utils import llmeter_bytes_decoder
    
    # Create original response with binary content
    original_payload = {
        "video": {
            "format": "mp4",
            "source": {"bytes": b"\x00\x00\x00\x18ftypmp42"}
        },
        "prompt": "Describe this video"
    }
    
    response = InvocationResponse(
        response_text="A video of a sunset",
        input_payload=original_payload,
        id="test-789",
        time_to_first_token=1.0,
        time_to_last_token=2.5
    )
    
    # Serialize to JSON
    json_str = response.to_json()
    
    # Deserialize back
    parsed = json.loads(json_str, object_hook=llmeter_bytes_decoder)
    
    # Verify input_payload was restored correctly
    assert parsed["input_payload"] == original_payload
    assert isinstance(parsed["input_payload"]["video"]["source"]["bytes"], bytes)
    assert parsed["input_payload"]["video"]["source"]["bytes"] == b"\x00\x00\x00\x18ftypmp42"
    
    # Verify other fields
    assert parsed["response_text"] == "A video of a sunset"
    assert parsed["id"] == "test-789"
    assert parsed["time_to_first_token"] == 1.0
    assert parsed["time_to_last_token"] == 2.5


def test_invocation_response_to_json_no_binary_content():
    """Test InvocationResponse.to_json with no binary content (backward compatibility).
    
    Validates: Requirements 7.1, 7.2
    """
    response = InvocationResponse(
        response_text="Simple text response",
        input_payload={
            "modelId": "test-model",
            "prompt": "Hello, world!"
        },
        id="test-no-binary"
    )
    
    # Serialize to JSON
    json_str = response.to_json()
    
    # Verify no marker objects are present
    assert "__llmeter_bytes__" not in json_str
    
    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert parsed["input_payload"]["prompt"] == "Hello, world!"


def test_invocation_response_to_json_with_kwargs():
    """Test that InvocationResponse.to_json passes through kwargs to json.dumps.
    
    Validates: Requirements 7.4
    """
    response = InvocationResponse(
        response_text="Test",
        input_payload={"data": b"\x01\x02\x03"},
        id="test-kwargs"
    )
    
    # Test with indent parameter
    json_str = response.to_json(indent=2)
    
    # Verify indentation is present
    assert "\n" in json_str
    assert "  " in json_str
    
    # Verify it's still valid JSON
    parsed = json.loads(json_str)
    assert parsed["id"] == "test-kwargs"
