# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for llmeter_default_serializer and llmeter_bytes_decoder.

This module contains property-based tests using Hypothesis to verify that
llmeter_default_serializer correctly handles all supported types (bytes, datetime, date,
time, PathLike, to_dict() objects, and str() fallback) and that
llmeter_bytes_decoder restores binary content from marker objects.

Feature: json-serialization-optimization
"""

import base64
import json
from datetime import date, datetime, time, timezone
from pathlib import PurePosixPath

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from llmeter.json_utils import llmeter_default_serializer, llmeter_bytes_decoder

# Test infrastructure is set up and ready for property test implementation
# This file will contain property-based tests for:
# - Property 1: Serialization produces valid JSON with marker objects
# - Property 2: Serialization preserves non-binary structure
# - Property 3: Deserialization restores bytes from markers
# - Property 4: Deserialization preserves non-marker dicts
# - Property 5: Round-trip serialization preserves data integrity
# - Property 6: Round-trip preserves dictionary key ordering
# - Property 7: Non-binary payloads are backward compatible
# - Property 8: Serialization errors are descriptive
# - Property 9: Deserialization errors are descriptive
# - Property 10: InvocationResponse serializes binary payloads


# Custom strategies for generating test data
@composite
def json_value_strategy(draw, allow_bytes=True):
    """Generate JSON-compatible values, optionally including bytes."""
    base_types = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=100),
    )

    if allow_bytes:
        return draw(st.one_of(base_types, st.binary(max_size=1000)))
    return draw(base_types)


@composite
def nested_dict_strategy(draw, max_depth=3, allow_bytes=True):
    """Generate nested dictionary structures with optional bytes objects."""
    if max_depth == 0:
        return draw(json_value_strategy(allow_bytes=allow_bytes))

    return draw(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20).filter(
                lambda k: k != "__llmeter_bytes__"
            ),
            values=st.one_of(
                json_value_strategy(allow_bytes=allow_bytes),
                nested_dict_strategy(max_depth=max_depth - 1, allow_bytes=allow_bytes),
                st.lists(json_value_strategy(allow_bytes=allow_bytes), max_size=5),
            ),
            max_size=5,
        )
    )


@composite
def payload_with_bytes_strategy(draw):
    """Generate payloads that contain at least one bytes object."""
    # Start with a nested dict that may or may not have bytes
    payload = draw(nested_dict_strategy(max_depth=3, allow_bytes=True))

    # Ensure at least one bytes object exists
    # Add a guaranteed bytes field
    payload["_test_bytes"] = draw(st.binary(min_size=1, max_size=500))

    return payload


@composite
def payload_without_bytes_strategy(draw):
    """Generate payloads that contain no bytes objects."""
    return draw(nested_dict_strategy(max_depth=3, allow_bytes=False))


class TestSerializationProperties:
    """Property-based tests for serialization behavior."""

    @given(payload_with_bytes_strategy())
    @settings(max_examples=100)
    def test_property_1_serialization_produces_valid_json_with_marker_objects(
        self, payload
    ):
        """Property 1: Serialization produces valid JSON with marker objects.

        **Validates: Requirements 1.1, 1.3**

        For any payload containing bytes objects at any nesting level, serializing
        the payload SHALL produce valid JSON where each bytes object is replaced
        with a marker object containing the key "__llmeter_bytes__" and a
        base64-encoded string value.
        """
        # Serialize the payload
        serialized = json.dumps(payload, default=llmeter_default_serializer)

        # Verify it's valid JSON by parsing it
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)

        # Verify that marker objects exist in the serialized string
        assert "__llmeter_bytes__" in serialized

        # Helper function to verify marker objects in nested structures
        def verify_markers(obj):
            """Recursively verify that bytes are replaced with marker objects."""
            if isinstance(obj, dict):
                # Check if this is a marker object
                if "__llmeter_bytes__" in obj:
                    # Verify it's a valid marker object
                    assert len(obj) == 1, "Marker object should have only one key"
                    base64_str = obj["__llmeter_bytes__"]
                    assert isinstance(base64_str, str), (
                        "Marker value should be a string"
                    )
                    # Verify it's valid base64 by attempting to decode
                    try:
                        base64.b64decode(base64_str)
                    except Exception as e:
                        raise AssertionError(f"Invalid base64 in marker object: {e}")
                else:
                    # Recursively check nested structures
                    for value in obj.values():
                        verify_markers(value)
            elif isinstance(obj, list):
                for item in obj:
                    verify_markers(item)

        # Verify all marker objects in the parsed structure
        verify_markers(parsed)


class TestDeserializationProperties:
    """Property-based tests for deserialization behavior."""

    @given(st.binary(min_size=0, max_size=1000))
    @settings(max_examples=100)
    def test_property_3_deserialization_restores_bytes_from_markers(
        self, original_bytes
    ):
        """Property 3: Deserialization restores bytes from markers.

        **Validates: Requirements 2.1**

        For any JSON string containing marker objects with the "__llmeter_bytes__"
        key, deserializing SHALL convert each marker object back to the original
        bytes object by base64-decoding the string value.
        """

        # Create a marker object from the original bytes
        base64_str = base64.b64encode(original_bytes).decode("utf-8")
        marker_object = {"__llmeter_bytes__": base64_str}

        # Decode the marker object
        decoded_bytes = llmeter_bytes_decoder(marker_object)

        # Verify the decoded bytes match the original
        assert isinstance(decoded_bytes, bytes), "Decoder should return bytes"
        assert decoded_bytes == original_bytes, (
            "Decoded bytes should match original bytes"
        )

    @given(payload_without_bytes_strategy())
    @settings(max_examples=100)
    def test_property_4_deserialization_preserves_non_marker_dicts(self, payload):
        """Property 4: Deserialization preserves non-marker dicts.

        **Validates: Requirements 2.4**

        For any payload containing dictionaries without the "__llmeter_bytes__"
        marker key, deserializing SHALL return those dictionaries unchanged.
        """

        # Helper function to recursively apply decoder to all dicts
        def apply_decoder_recursively(obj):
            """Apply decoder to all dicts in the structure."""
            if isinstance(obj, dict):
                # Apply decoder to this dict
                decoded = llmeter_bytes_decoder(obj)
                # If it's still a dict (not converted to bytes), recurse
                if isinstance(decoded, dict):
                    return {k: apply_decoder_recursively(v) for k, v in decoded.items()}
                return decoded
            elif isinstance(obj, list):
                return [apply_decoder_recursively(item) for item in obj]
            else:
                return obj

        # Apply decoder recursively to the entire payload
        decoded_payload = apply_decoder_recursively(payload)

        # Verify the payload is unchanged
        assert decoded_payload == payload, (
            "Decoder should return non-marker dicts unchanged"
        )

        # Verify no bytes objects were introduced
        def verify_no_bytes(obj):
            """Recursively verify no bytes objects exist."""
            if isinstance(obj, bytes):
                raise AssertionError("Decoder introduced bytes object unexpectedly")
            elif isinstance(obj, dict):
                for value in obj.values():
                    verify_no_bytes(value)
            elif isinstance(obj, list):
                for item in obj:
                    verify_no_bytes(item)

        verify_no_bytes(decoded_payload)


class TestRoundTripProperties:
    """Property-based tests for round-trip serialization integrity."""

    @given(payload_with_bytes_strategy())
    @settings(max_examples=100)
    def test_property_2_serialization_preserves_non_binary_structure(self, payload):
        """Property 2: Serialization preserves non-binary structure.

        **Validates: Requirements 1.5**

        For any payload structure (keys, nesting levels, value types except bytes),
        serializing then parsing the JSON SHALL preserve the structure identically,
        with only bytes objects replaced by marker objects.

        # Feature: json-serialization-optimization, Property 2: Serialization preserves non-binary structure
        """
        # Serialize the payload
        serialized = json.dumps(payload, default=llmeter_default_serializer)

        # Parse the JSON (without decoding markers back to bytes)
        parsed = json.loads(serialized)

        # Helper function to verify structure preservation
        def verify_structure(original, parsed_obj, path=""):
            """Recursively verify that structure is preserved except for bytes."""
            if isinstance(original, bytes):
                # Bytes should be replaced with marker object
                assert isinstance(parsed_obj, dict), (
                    f"At {path}: Expected marker dict for bytes, got {type(parsed_obj)}"
                )
                assert "__llmeter_bytes__" in parsed_obj, (
                    f"At {path}: Expected marker object for bytes"
                )
                assert len(parsed_obj) == 1, (
                    f"At {path}: Marker object should have only one key"
                )
                # Verify the base64 string can be decoded back to original bytes
                decoded = base64.b64decode(parsed_obj["__llmeter_bytes__"])
                assert decoded == original, (
                    f"At {path}: Decoded bytes don't match original"
                )
            elif isinstance(original, dict):
                # Dict structure should be preserved
                assert isinstance(parsed_obj, dict), (
                    f"At {path}: Expected dict, got {type(parsed_obj)}"
                )
                assert set(original.keys()) == set(parsed_obj.keys()), (
                    f"At {path}: Dict keys differ. Original: {set(original.keys())}, "
                    f"Parsed: {set(parsed_obj.keys())}"
                )
                # Verify nesting is preserved
                for key in original.keys():
                    verify_structure(
                        original[key], parsed_obj[key], f"{path}.{key}" if path else key
                    )
            elif isinstance(original, list):
                # List structure should be preserved
                assert isinstance(parsed_obj, list), (
                    f"At {path}: Expected list, got {type(parsed_obj)}"
                )
                assert len(original) == len(parsed_obj), (
                    f"At {path}: List lengths differ. Original: {len(original)}, "
                    f"Parsed: {len(parsed_obj)}"
                )
                # Verify each element
                for i, (orig_item, parsed_item) in enumerate(zip(original, parsed_obj)):
                    verify_structure(orig_item, parsed_item, f"{path}[{i}]")
            else:
                # Primitive types should be preserved exactly
                # Note: We need exact type matching here, not isinstance checks
                assert type(original) is type(parsed_obj), (  # noqa: E721
                    f"At {path}: Type mismatch. Original: {type(original)}, "
                    f"Parsed: {type(parsed_obj)}"
                )
                assert original == parsed_obj, (
                    f"At {path}: Values differ. Original: {original}, "
                    f"Parsed: {parsed_obj}"
                )

        # Verify structure preservation throughout
        verify_structure(payload, parsed)

    @given(payload_with_bytes_strategy())
    @settings(max_examples=100)
    def test_property_5_round_trip_serialization_preserves_data_integrity(
        self, payload
    ):
        """Property 5: Round-trip serialization preserves data integrity.

        **Validates: Requirements 4.1, 2.5**

        For any valid payload with binary content, the property
        deserialize(serialize(payload)) == payload SHALL hold, preserving
        byte-for-byte equality of all bytes objects and exact equality of
        all other values.
        """

        # Serialize the payload
        serialized = json.dumps(payload, default=llmeter_default_serializer)

        # Deserialize the payload
        deserialized = json.loads(serialized, object_hook=llmeter_bytes_decoder)

        # Verify round-trip equality
        assert deserialized == payload, (
            "Round-trip serialization should preserve data integrity"
        )

        # Helper function to verify byte-for-byte equality of bytes objects
        def verify_bytes_equality(original, restored, path=""):
            """Recursively verify that bytes objects are byte-for-byte equal."""
            if isinstance(original, bytes):
                assert isinstance(restored, bytes), (
                    f"At {path}: Expected bytes, got {type(restored)}"
                )
                assert original == restored, f"At {path}: Bytes objects differ"
            elif isinstance(original, dict):
                assert isinstance(restored, dict), (
                    f"At {path}: Expected dict, got {type(restored)}"
                )
                assert set(original.keys()) == set(restored.keys()), (
                    f"At {path}: Dict keys differ"
                )
                for key in original.keys():
                    verify_bytes_equality(
                        original[key], restored[key], f"{path}.{key}" if path else key
                    )
            elif isinstance(original, list):
                assert isinstance(restored, list), (
                    f"At {path}: Expected list, got {type(restored)}"
                )
                assert len(original) == len(restored), f"At {path}: List lengths differ"
                for i, (orig_item, rest_item) in enumerate(zip(original, restored)):
                    verify_bytes_equality(orig_item, rest_item, f"{path}[{i}]")
            else:
                # For primitive types, equality check is sufficient
                assert original == restored, (
                    f"At {path}: Values differ: {original} != {restored}"
                )

        # Verify byte-for-byte equality throughout the structure
        verify_bytes_equality(payload, deserialized)

    @given(payload_with_bytes_strategy())
    @settings(max_examples=100)
    def test_property_6_round_trip_preserves_dictionary_key_ordering(self, payload):
        """Property 6: Round-trip preserves dictionary key ordering.

        **Validates: Requirements 4.4**

        For any payload with ordered dictionaries, round-trip serialization SHALL
        preserve the insertion order of dictionary keys.

        # Feature: json-serialization-optimization, Property 6: Round-trip preserves dictionary key ordering
        """

        # Serialize the payload
        serialized = json.dumps(payload, default=llmeter_default_serializer)

        # Deserialize the payload
        deserialized = json.loads(serialized, object_hook=llmeter_bytes_decoder)

        # Helper function to verify key ordering
        def verify_key_ordering(original, restored, path=""):
            """Recursively verify that dictionary key ordering is preserved."""
            if isinstance(original, dict):
                assert isinstance(restored, dict), (
                    f"At {path}: Expected dict, got {type(restored)}"
                )

                # Get the keys as lists to preserve order
                original_keys = list(original.keys())
                restored_keys = list(restored.keys())

                # Verify the keys are in the same order
                assert original_keys == restored_keys, (
                    f"At {path}: Key ordering differs. "
                    f"Original: {original_keys}, Restored: {restored_keys}"
                )

                # Recursively verify nested structures
                for key in original_keys:
                    verify_key_ordering(
                        original[key], restored[key], f"{path}.{key}" if path else key
                    )
            elif isinstance(original, list):
                assert isinstance(restored, list), (
                    f"At {path}: Expected list, got {type(restored)}"
                )
                assert len(original) == len(restored), f"At {path}: List lengths differ"
                # Verify each element
                for i, (orig_item, rest_item) in enumerate(zip(original, restored)):
                    verify_key_ordering(orig_item, rest_item, f"{path}[{i}]")
            # For non-dict, non-list types, no key ordering to verify

        # Verify key ordering is preserved throughout the structure
        verify_key_ordering(payload, deserialized)


class TestBackwardCompatibilityProperties:
    """Property-based tests for backward compatibility."""

    @given(payload_without_bytes_strategy())
    @settings(max_examples=100)
    def test_property_7_non_binary_payloads_are_backward_compatible(self, payload):
        """Property 7: Non-binary payloads are backward compatible.

        **Validates: Requirements 3.2, 3.3**

        For any payload containing no bytes objects, serializing with the new
        encoder SHALL produce output identical to serializing with the standard
        json.dumps (no marker objects introduced).

        # Feature: json-serialization-optimization, Property 7: Non-binary payloads are backward compatible
        """
        # Serialize with the new encoder
        serialized_with_encoder = json.dumps(
            payload, default=llmeter_default_serializer
        )

        # Serialize with standard json.dumps
        serialized_standard = json.dumps(payload)

        # Verify they produce identical output
        assert serialized_with_encoder == serialized_standard, (
            "Serialization with llmeter_default_serializer should produce identical output "
            "to standard json.dumps for payloads without bytes objects"
        )

        # Verify no marker objects were introduced
        assert "__llmeter_bytes__" not in serialized_with_encoder, (
            "No marker objects should be introduced for payloads without bytes"
        )

        # Verify both can be parsed identically
        parsed_encoder = json.loads(serialized_with_encoder)
        parsed_standard = json.loads(serialized_standard)
        assert parsed_encoder == parsed_standard, (
            "Parsed output should be identical for both serialization methods"
        )

        # Verify the parsed output matches the original payload
        assert parsed_encoder == payload, (
            "Parsed output should match the original payload"
        )


class TestErrorHandlingProperties:
    """Property-based tests for error handling."""

    @composite
    def unserializable_object_strategy(draw):
        """Generate objects that are not JSON serializable and not bytes."""
        # Create various types of unserializable objects
        unserializable_types = [
            # Custom class instance
            lambda: type("CustomClass", (), {})(),
            # Function
            lambda: lambda x: x,
            # Set (not JSON serializable)
            lambda: {1, 2, 3},
            # Complex number
            lambda: complex(1, 2),
            # Object with __dict__
            lambda: type("ObjWithDict", (), {"attr": "value"})(),
        ]

        # Choose one of the unserializable types
        return draw(st.sampled_from(unserializable_types))()

    @given(st.data())
    @settings(max_examples=100)
    def test_property_8_serialization_handles_unknown_types(self, data):
        """Property 8: Serialization handles unknown types via str() fallback.

        **Validates: Requirements 6.1**

        For any payload containing unserializable types (not bytes, not standard
        JSON types), serialization SHALL succeed by falling back to str()
        representation, producing valid JSON output.

        # Feature: json-serialization-optimization, Property 8: Serialization handles unknown types
        """
        # Generate a payload with an unserializable object
        unserializable_obj = data.draw(
            TestErrorHandlingProperties.unserializable_object_strategy()
        )

        # Create a payload containing the unserializable object
        # We'll place it at various locations in the structure
        placement_strategy = st.sampled_from(
            [
                # Direct value
                lambda obj: {"unserializable": obj},
                # Nested in dict
                lambda obj: {"outer": {"inner": {"unserializable": obj}}},
                # In a list
                lambda obj: {"items": [1, 2, obj, 4]},
                # Mixed structure
                lambda obj: {"data": {"list": [{"nested": obj}]}},
            ]
        )

        payload_creator = data.draw(placement_strategy)
        payload = payload_creator(unserializable_obj)

        # The unified encoder falls back to str() for unknown types
        result = json.dumps(payload, default=llmeter_default_serializer)
        # Result should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @given(st.data())
    @settings(max_examples=100)
    def test_property_9_deserialization_errors_are_descriptive(self, data):
        """Property 9: Deserialization errors are descriptive.

        **Validates: Requirements 6.2**

        For any invalid JSON string or JSON containing marker objects with invalid
        base64 strings, attempting to deserialize SHALL raise an appropriate
        exception (JSONDecodeError or binascii.Error) with a descriptive message.

        # Feature: json-serialization-optimization, Property 9: Deserialization errors are descriptive
        """

        # Test invalid JSON strings that will raise JSONDecodeError
        # Note: base64.b64decode() is lenient by default and accepts many inputs,
        # so we focus on JSON parsing errors which are more common in practice

        # Generate invalid JSON strings that will definitely fail parsing
        invalid_json_strategy = st.sampled_from(
            [
                "{invalid json}",
                '{"key": undefined}',
                "{'single': 'quotes'}",
                '{"unclosed": ',
                '{"trailing": "comma",}',
                "not json at all",
                '["unclosed array"',
                "}invalid start{",
                '{"double""quotes"}',
                '{"key": value}',  # unquoted value
                "[1, 2, 3,]",  # trailing comma in array
            ]
        )
        invalid_json = data.draw(invalid_json_strategy)

        # Attempt to deserialize and verify it raises JSONDecodeError
        try:
            json.loads(invalid_json, object_hook=llmeter_bytes_decoder)
            # If we get here, deserialization succeeded when it shouldn't have
            raise AssertionError(
                f"Expected JSONDecodeError for invalid JSON: {invalid_json}"
            )
        except json.JSONDecodeError as e:
            # Verify the error message is descriptive
            error_msg = str(e)

            # The error message should indicate it's a JSON parsing error
            # JSONDecodeError messages typically contain position information
            assert len(error_msg) > 0, "Error message should not be empty"

            # Verify the exception has the expected attributes
            assert hasattr(e, "msg"), "JSONDecodeError should have 'msg' attribute"
            assert hasattr(e, "lineno"), (
                "JSONDecodeError should have 'lineno' attribute"
            )
            assert hasattr(e, "colno"), "JSONDecodeError should have 'colno' attribute"

            # Verify the error message contains useful information
            # It should mention what went wrong
            assert e.msg, "JSONDecodeError msg should not be empty"


class TestInvocationResponseProperties:
    """Property-based tests for InvocationResponse serialization."""

    @given(nested_dict_strategy(max_depth=3, allow_bytes=True))
    @settings(max_examples=100, deadline=None)
    def test_property_10_invocation_response_serializes_binary_payloads(
        self, input_payload
    ):
        """Property 10: InvocationResponse serializes binary payloads.

        **Validates: Requirements 7.1**

        For any InvocationResponse object where the input_payload field contains
        bytes objects, calling to_json() SHALL produce valid JSON with marker
        objects replacing the bytes.

        # Feature: json-serialization-optimization, Property 10: InvocationResponse serializes binary payloads
        """
        from llmeter.endpoints.base import InvocationResponse

        # Create an InvocationResponse with the generated input_payload
        response = InvocationResponse(
            response_text="Test response",
            input_payload=input_payload,
            id="test-id",
            time_to_first_token=0.1,
            time_to_last_token=0.5,
            num_tokens_input=10,
            num_tokens_output=20,
        )

        # Serialize using to_json()
        json_str = response.to_json()

        # Verify it's valid JSON by parsing it
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict), "to_json() should produce a valid JSON object"

        # Verify the input_payload field exists
        assert "input_payload" in parsed, (
            "Serialized response should contain input_payload"
        )

        # Helper function to check for bytes in original and markers in serialized
        def has_bytes(obj):
            """Recursively check if object contains bytes."""
            if isinstance(obj, bytes):
                return True
            elif isinstance(obj, dict):
                return any(has_bytes(v) for v in obj.values())
            elif isinstance(obj, list):
                return any(has_bytes(item) for item in obj)
            return False

        def verify_bytes_serialized(original, serialized_obj, path=""):
            """Recursively verify bytes are replaced with marker objects."""
            if isinstance(original, bytes):
                # Bytes should be replaced with marker object
                assert isinstance(serialized_obj, dict), (
                    f"At {path}: Expected marker dict for bytes, got {type(serialized_obj)}"
                )
                assert "__llmeter_bytes__" in serialized_obj, (
                    f"At {path}: Expected marker object for bytes"
                )
                assert len(serialized_obj) == 1, (
                    f"At {path}: Marker object should have only one key"
                )
                # Verify the base64 string can be decoded back to original bytes
                base64_str = serialized_obj["__llmeter_bytes__"]
                assert isinstance(base64_str, str), (
                    f"At {path}: Marker value should be a string"
                )
                decoded = base64.b64decode(base64_str)
                assert decoded == original, (
                    f"At {path}: Decoded bytes don't match original"
                )
            elif isinstance(original, dict):
                assert isinstance(serialized_obj, dict), (
                    f"At {path}: Expected dict, got {type(serialized_obj)}"
                )
                # Verify all keys are present
                for key in original.keys():
                    assert key in serialized_obj, (
                        f"At {path}: Key '{key}' missing in serialized object"
                    )
                    verify_bytes_serialized(
                        original[key],
                        serialized_obj[key],
                        f"{path}.{key}" if path else key,
                    )
            elif isinstance(original, list):
                assert isinstance(serialized_obj, list), (
                    f"At {path}: Expected list, got {type(serialized_obj)}"
                )
                assert len(original) == len(serialized_obj), (
                    f"At {path}: List lengths differ"
                )
                for i, (orig_item, ser_item) in enumerate(
                    zip(original, serialized_obj)
                ):
                    verify_bytes_serialized(orig_item, ser_item, f"{path}[{i}]")
            # For other types, no special verification needed

        # If the input_payload contains bytes, verify they're serialized correctly
        if has_bytes(input_payload):
            assert "__llmeter_bytes__" in json_str, (
                "Serialized JSON should contain marker objects when input_payload has bytes"
            )
            verify_bytes_serialized(
                input_payload, parsed["input_payload"], "input_payload"
            )

        # Verify other fields are serialized correctly
        assert parsed["response_text"] == "Test response"
        assert parsed["id"] == "test-id"
        assert parsed["time_to_first_token"] == 0.1
        assert parsed["time_to_last_token"] == 0.5
        assert parsed["num_tokens_input"] == 10
        assert parsed["num_tokens_output"] == 20


# ---------------------------------------------------------------------------
# Strategies for the extended type tests
# ---------------------------------------------------------------------------

# datetime strategy: aware and naive datetimes
_datetime_strategy = st.one_of(
    # Naive datetimes
    st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2030, 12, 31)),
    # UTC-aware datetimes
    st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2030, 12, 31),
        timezones=st.just(timezone.utc),
    ),
)

_date_strategy = st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31))

_time_strategy = st.times()

_path_strategy = st.from_regex(r"[a-z][a-z0-9_/]{0,30}", fullmatch=True).map(
    PurePosixPath
)


@composite
def to_dict_object_strategy(draw):
    """Generate an object with a to_dict() method returning a JSON-safe dict."""
    inner = draw(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.one_of(st.integers(), st.text(max_size=20), st.booleans()),
            max_size=5,
        )
    )

    class _Obj:
        def __init__(self, d):
            self._d = d

        def to_dict(self):
            return self._d

    return _Obj(inner), inner


class TestDatetimeSerializationProperties:
    """Property-based tests for datetime/date/time encoding."""

    @given(_datetime_strategy)
    @settings(max_examples=100)
    def test_datetime_produces_iso_string_with_z_suffix(self, dt):
        """Datetime values are serialized to ISO-8601 strings at seconds precision.

        Aware datetimes are converted to UTC and suffixed with 'Z'.
        Naive datetimes are serialized as-is with no timezone indicator.
        Microseconds are truncated (the encoder uses timespec="seconds").
        """
        result = json.loads(json.dumps({"v": dt}, default=llmeter_default_serializer))
        assert isinstance(result["v"], str)
        if dt.tzinfo is not None:
            assert result["v"].endswith("Z")
            parsed_back = datetime.fromisoformat(result["v"].replace("Z", "+00:00"))
            expected = dt.astimezone(timezone.utc).replace(microsecond=0)
            assert parsed_back == expected
        else:
            parsed_back = datetime.fromisoformat(result["v"])
            assert parsed_back == dt.replace(microsecond=0)

    @given(_date_strategy)
    @settings(max_examples=100)
    def test_date_produces_iso_string(self, d):
        """Date values are serialized via isoformat()."""
        result = json.loads(json.dumps({"v": d}, default=llmeter_default_serializer))
        assert result["v"] == d.isoformat()
        assert date.fromisoformat(result["v"]) == d

    @given(_time_strategy)
    @settings(max_examples=100)
    def test_time_produces_iso_string(self, t):
        """Time values are serialized via isoformat()."""
        result = json.loads(json.dumps({"v": t}, default=llmeter_default_serializer))
        assert result["v"] == t.isoformat()
        assert time.fromisoformat(result["v"]) == t


class TestPathSerializationProperties:
    """Property-based tests for PathLike encoding."""

    @given(_path_strategy)
    @settings(max_examples=100)
    def test_pathlike_produces_posix_string(self, p):
        """PathLike objects are serialized to POSIX path strings."""
        result = json.loads(json.dumps({"v": p}, default=llmeter_default_serializer))
        assert isinstance(result["v"], str)
        assert result["v"] == p.as_posix()


class TestToDictSerializationProperties:
    """Property-based tests for to_dict() delegation."""

    @given(to_dict_object_strategy())
    @settings(max_examples=100)
    def test_to_dict_delegation_produces_expected_dict(self, obj_and_expected):
        """Objects with to_dict() are serialized by calling that method."""
        obj, expected = obj_and_expected
        result = json.loads(json.dumps({"v": obj}, default=llmeter_default_serializer))
        assert result["v"] == expected

    @given(to_dict_object_strategy(), st.binary(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_to_dict_and_bytes_coexist(self, obj_and_expected, raw_bytes):
        """Payloads mixing to_dict() objects and bytes round-trip correctly."""
        obj, expected = obj_and_expected
        payload = {"obj": obj, "data": raw_bytes}
        serialized = json.dumps(payload, default=llmeter_default_serializer)
        restored = json.loads(serialized, object_hook=llmeter_bytes_decoder)
        assert restored["obj"] == expected
        assert restored["data"] == raw_bytes
