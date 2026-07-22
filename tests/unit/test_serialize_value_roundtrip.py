# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that _serialize_value/_deserialize_value correctly round-trip datetime and bytes."""

import base64
from datetime import datetime, timezone

import pytest

from llmeter.serialization import (
    Serializable,
    _deserialize_value,
    _serialize_value,
)


class TestBytesRoundTrip:
    """_serialize_value must produce the __llmeter_bytes__ marker for bytes,
    and _deserialize_value must restore it."""

    def test_bytes_serializes_to_marker(self):
        val = b"\x00\x01\x02\xff"
        result = _serialize_value(val)
        assert isinstance(result, dict)
        assert "__llmeter_bytes__" in result
        assert len(result) == 1
        assert base64.b64decode(result["__llmeter_bytes__"]) == val

    def test_bytes_roundtrip(self):
        val = b"hello world"
        assert _deserialize_value(_serialize_value(val)) == val

    def test_empty_bytes_roundtrip(self):
        val = b""
        assert _deserialize_value(_serialize_value(val)) == val

    def test_bytes_nested_in_dict(self):
        val = {"key": b"\xde\xad\xbe\xef", "other": "text"}
        serialized = _serialize_value(val)
        assert serialized["key"] == {
            "__llmeter_bytes__": base64.b64encode(b"\xde\xad\xbe\xef").decode()
        }
        assert serialized["other"] == "text"
        assert _deserialize_value(serialized) == val

    def test_bytes_nested_in_list(self):
        val = [b"first", "middle", b"last"]
        serialized = _serialize_value(val)
        restored = _deserialize_value(serialized)
        assert restored == val

    def test_bytes_deeply_nested(self):
        val = {"a": {"b": [{"c": b"\x01\x02\x03"}]}}
        assert _deserialize_value(_serialize_value(val)) == val


class TestDatetimeRoundTrip:
    """_serialize_value produces ISO-8601 Z-suffixed strings for datetime,
    and _deserialize_value recognizes the format and restores datetime objects."""

    def test_datetime_serializes_to_iso_string(self):
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = _serialize_value(dt)
        assert isinstance(result, str)
        assert result == "2024-06-15T10:30:00Z"

    def test_datetime_utc_roundtrip(self):
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        restored = _deserialize_value(_serialize_value(dt))
        assert restored == dt
        assert isinstance(restored, datetime)

    def test_datetime_naive_roundtrip(self):
        dt = datetime(2024, 3, 15, 8, 45, 30)
        serialized = _serialize_value(dt)
        # Naive datetimes produce no Z suffix, so they stay as plain strings
        assert serialized == "2024-03-15T08:45:30"

    def test_datetime_is_not_silently_lost(self):
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = _serialize_value(dt)
        # Must be the canonical format, not repr/str() of the datetime object
        assert "datetime" not in result
        assert result.endswith("Z")

    def test_datetime_nested_in_dict(self):
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        val = {"timestamp": dt, "label": "test"}
        serialized = _serialize_value(val)
        assert serialized["timestamp"] == "2024-06-15T10:30:00Z"
        restored = _deserialize_value(serialized)
        assert restored["timestamp"] == dt
        assert restored["label"] == "test"

    def test_datetime_nested_in_list(self):
        dt1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dt2 = datetime(2024, 12, 31, tzinfo=timezone.utc)
        val = [dt1, "gap", dt2]
        restored = _deserialize_value(_serialize_value(val))
        assert restored[0] == dt1
        assert restored[1] == "gap"
        assert restored[2] == dt2

    def test_plain_string_not_confused_for_datetime(self):
        val = "hello world"
        assert _deserialize_value(val) == "hello world"

    def test_partial_iso_string_stays_as_string(self):
        val = "2024-06-15"
        assert _deserialize_value(val) == "2024-06-15"


class TestPathSerialization:
    """PathLike objects serialize to POSIX path strings."""

    def test_path_serializes_to_string(self):
        from pathlib import PurePosixPath

        p = PurePosixPath("/tmp/foo/bar")
        result = _serialize_value(p)
        assert isinstance(result, str)
        assert result == "/tmp/foo/bar"


class TestSerializableWithDatetimeAndBytes:
    """End-to-end: a Serializable class with datetime/bytes fields round-trips
    via _get_llmeter_state/_set_llmeter_state."""

    def test_serializable_with_datetime_field(self):
        class MyObj(Serializable):
            def __init__(self, created_at: datetime, name: str = "test"):
                self.created_at = created_at
                self.name = name

        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        obj = MyObj(created_at=dt, name="hello")
        state = obj._get_llmeter_state()

        restored = MyObj.__new__(MyObj)
        restored._set_llmeter_state(state)

        assert restored.created_at == dt
        assert isinstance(restored.created_at, datetime)
        assert restored.name == "hello"

    def test_serializable_with_bytes_field(self):
        class MyObj(Serializable):
            def __init__(self, payload: bytes, label: str = "x"):
                self.payload = payload
                self.label = label

        obj = MyObj(payload=b"\xde\xad\xbe\xef", label="binary")
        state = obj._get_llmeter_state()

        restored = MyObj.__new__(MyObj)
        restored._set_llmeter_state(state)

        assert restored.payload == b"\xde\xad\xbe\xef"
        assert isinstance(restored.payload, bytes)
        assert restored.label == "binary"

    def test_serializable_with_mixed_fields(self):
        class MyObj(Serializable):
            def __init__(self, ts: datetime, data: bytes, info: dict):
                self.ts = ts
                self.data = data
                self.info = info

        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        obj = MyObj(ts=dt, data=b"raw", info={"nested_bytes": b"\x01", "nested_ts": dt})
        state = obj._get_llmeter_state()

        restored = MyObj.__new__(MyObj)
        restored._set_llmeter_state(state)

        assert restored.ts == dt
        assert restored.data == b"raw"
        assert restored.info["nested_bytes"] == b"\x01"
        assert restored.info["nested_ts"] == dt


class TestUnserializableTypeRaises:
    """_serialize_value must raise TypeError for unsupported types,
    never silently convert to str."""

    def test_set_raises(self):
        with pytest.raises(TypeError, match="Cannot serialize"):
            _serialize_value({1, 2, 3})

    def test_custom_object_without_getstate_raises(self):
        class Opaque:
            pass

        with pytest.raises(TypeError, match="Cannot serialize"):
            _serialize_value(Opaque())
