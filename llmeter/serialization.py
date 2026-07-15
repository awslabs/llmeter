# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified serialization for all LLMeter objects.

This module is the single source of truth for serialization in LLMeter.

Key components:

- :class:`Serializable` — Mixin that gives any class automatic
  ``__getstate__``/``__setstate__`` by introspecting ``__init__``.

- :func:`dump_object` / :func:`load_object` — Full round-trip persistence
  using a ``{"__llmeter_class__": ..., "__llmeter_state__": ...}`` envelope.

- :func:`json_default` — A ``json.dumps``-compatible *default* handler for
  ``bytes``, ``datetime``, ``PathLike``.

- :func:`bytes_decoder` — A ``json.loads``-compatible *object_hook* that
  restores binary content encoded by :func:`json_default`.

- :func:`datetime_to_str` / :func:`str_to_datetime` — Standardized
  datetime ↔ string conversion (UTC ISO-8601 with ``Z`` suffix).

.. warning:: Security

    :func:`load_object` imports and instantiates whatever class path is in the
    ``__llmeter_class__`` field. Do not load configs from untrusted sources.
"""

import base64
import importlib
import inspect
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timezone
from typing import Any

from upath import UPath as Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def datetime_to_str(dt: datetime) -> str:
    """Convert a datetime to a UTC ISO-8601 string with ``Z`` suffix.

    Timezone-aware datetimes are converted to UTC first. Naive datetimes are
    serialized as-is (assumed UTC).
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def str_to_datetime(s: str) -> datetime:
    """Parse an ISO-8601 string (with optional ``Z`` suffix) to a datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def json_default(obj: Any) -> Any:
    """Fallback serializer for :func:`json.dumps`.

    Handles ``bytes`` (→ base64 marker), ``datetime``, ``date``/``time``,
    ``os.PathLike``, and falls back to ``str()``.
    """
    if isinstance(obj, bytes):
        return {"__llmeter_bytes__": base64.b64encode(obj).decode("utf-8")}
    if isinstance(obj, datetime):
        return datetime_to_str(obj)
    if isinstance(obj, (date, time)):
        return obj.isoformat()
    if isinstance(obj, (os.PathLike, Path)):
        return Path(obj).as_posix()
    try:
        return str(obj)
    except Exception:
        return None


def bytes_decoder(dct: dict) -> dict | bytes:
    """Object hook for :func:`json.loads` that restores ``__llmeter_bytes__`` markers."""
    if "__llmeter_bytes__" in dct and len(dct) == 1:
        return base64.b64decode(dct["__llmeter_bytes__"])
    return dct


# ---------------------------------------------------------------------------
# Serializable mixin
# ---------------------------------------------------------------------------


class Serializable:
    """Mixin providing automatic ``__getstate__``/``__setstate__`` via __init__ introspection.

    Works with plain classes, ``@dataclass``, and any class whose ``__init__``
    parameters correspond to instance attributes (``self.x`` or ``self._x``).

    Nested :class:`Serializable` objects are recursively persisted via
    :func:`dump_object` / :func:`load_object`.
    """

    def __getstate__(self) -> dict:
        sig = inspect.signature(self.__init__)
        state = {}
        for name, param in sig.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if hasattr(self, name):
                state[name] = _serialize_value(getattr(self, name))
            elif hasattr(self, f"_{name}"):
                state[name] = _serialize_value(getattr(self, f"_{name}"))
        return state

    def __setstate__(self, state: dict) -> None:
        deserialized = {k: _deserialize_value(v) for k, v in state.items()}
        self.__init__(**deserialized)


# ---------------------------------------------------------------------------
# Object serialization API
# ---------------------------------------------------------------------------


def dump_object(obj: Any) -> dict:
    """Serialize an object to a type-tagged dict for round-trip persistence.

    Returns ``{"__llmeter_class__": "module.Class", "__llmeter_state__": {...}}``.
    """
    class_path = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
    if (
        hasattr(obj, "__getstate__")
        and type(obj).__getstate__ is not object.__getstate__
    ):
        state = obj.__getstate__()
    elif is_dataclass(obj) and not isinstance(obj, type):
        state = asdict(obj)
    elif hasattr(obj, "__dict__"):
        state = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    else:
        state = {}
    return {"__llmeter_class__": class_path, "__llmeter_state__": state}


def load_object(data: dict) -> Any:
    """Restore an object from a type-tagged dict produced by :func:`dump_object`.

    .. warning:: Do not call on data from untrusted sources.
    """
    class_path = data["__llmeter_class__"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    obj = cls.__new__(cls)
    obj.__setstate__(data["__llmeter_state__"])
    return obj


# ---------------------------------------------------------------------------
# Internal helpers for recursive serialization
# ---------------------------------------------------------------------------


def _serialize_value(val: Any) -> Any:
    """Recursively prepare a value for JSON persistence."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if hasattr(val, "__getstate__") and type(val).__getstate__ is not object.__getstate__:
        return dump_object(val)
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(item) for item in val]
    return str(val)


def _deserialize_value(val: Any) -> Any:
    """Recursively restore a value from JSON persistence."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, dict):
        if "__llmeter_class__" in val and "__llmeter_state__" in val:
            return load_object(val)
        return {k: _deserialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_deserialize_value(item) for item in val]
    return val
