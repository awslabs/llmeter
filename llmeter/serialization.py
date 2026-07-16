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
import json
import logging
import os
import re
import types as _types
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime, time, timezone
from typing import Any

from upath import UPath as Path
from upath.types import ReadablePathLike, WritablePathLike

from .utils import ensure_path

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
    """Serialize a single non-JSON-serializable object.

    Intended for use as the ``default`` argument to :func:`json.dumps` or
    :func:`json.dump`.

    Type handling (checked in order):

    * ``bytes`` — wrapped in a ``{"__llmeter_bytes__": "<base64>"}`` marker so
      that :func:`bytes_decoder` can restore them on the way back.
    * ``datetime`` — converted to a UTC ISO-8601 string with a ``Z`` suffix.
    * ``date`` / ``time`` — converted via ``.isoformat()``.
    * ``os.PathLike`` — converted to a POSIX path string.
    * Anything else — ``str()`` fallback (returns ``None`` if that also fails).

    Args:
        obj: The object that the default encoder could not handle.

    Returns:
        A JSON-serializable representation of *obj*.
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
    """Decode ``__llmeter_bytes__`` marker objects back to ``bytes``.

    Intended for use as the ``object_hook`` argument to :func:`json.loads` or
    :func:`json.load`. Marker objects produced by :func:`json_default` are detected
    and converted back to ``bytes``; all other dicts pass through unchanged.

    Args:
        dct: A dictionary produced by the JSON parser.

    Returns:
        The original ``bytes`` if *dct* is a marker object, otherwise *dct* unchanged.
    """
    if "__llmeter_bytes__" in dct and len(dct) == 1:
        return base64.b64decode(dct["__llmeter_bytes__"])
    return dct


def _get_type_args(tp) -> tuple:
    """Return the members of a union type (e.g. ``datetime | None`` -> (datetime, NoneType))."""
    if isinstance(tp, _types.UnionType):
        return tp.__args__
    origin = getattr(tp, "__origin__", None)
    if origin is _types.UnionType:
        return tp.__args__
    return (tp,) if isinstance(tp, type) else ()


def restore_dataclass_types(cls, data: dict) -> None:
    """Restore typed fields in a dict destined for a dataclass constructor.

    Introspects ``cls`` (a dataclass) and converts JSON-native values back to their
    annotated Python types. Currently handles:

    * ``datetime`` fields — parses ISO-8601 strings via :func:`str_to_datetime`.
    * ``bytes`` fields — decodes ``__llmeter_bytes__`` markers via base64.

    Only fields declared on ``cls`` are touched — nested user payloads (e.g.
    ``input_payload``) are left unchanged. Mutates *data* in place.

    Args:
        cls: A dataclass type to introspect for field type annotations.
        data: A dictionary of field values (e.g. from :func:`json.load`) to coerce.
    """
    for f in fields(cls):
        val = data.get(f.name)
        if val is None:
            continue
        type_args = _get_type_args(f.type)
        match val:
            case str() if datetime in type_args:
                try:
                    data[f.name] = str_to_datetime(val)
                except ValueError:
                    pass
            case {"__llmeter_bytes__": b64} if bytes in type_args and len(val) == 1:
                data[f.name] = base64.b64decode(b64)


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

    def save_to_file(self, path: WritablePathLike) -> None:
        """Save this object to a JSON file.

        Uses the ``__getstate__`` protocol. Override ``__getstate__`` (not this method)
        if custom serialization is needed.

        Args:
            path: (Local or Cloud) path where the object will be saved.
        """
        path = ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dump_object(self)
        with path.open("w") as f:
            json.dump(data, f, indent=4, default=json_default)

    @classmethod
    def load_from_file(cls, path: ReadablePathLike) -> "Serializable":
        """Load an object from a JSON file.

        Detects the type from the ``__llmeter_class__`` field and reconstructs it.

        Args:
            path: (Local or Cloud) path where the object was saved.
        Returns:
            The loaded instance.
        """
        path = ensure_path(path)
        with path.open("r") as f:
            data = json.load(f)
        return load_object(data)


# ---------------------------------------------------------------------------
# Object serialization API
# ---------------------------------------------------------------------------


def dump_object(obj: Any) -> dict:
    """Serialize an object to a type-tagged dict for round-trip persistence.

    The returned envelope has the form
    ``{"__llmeter_class__": "module.Class", "__llmeter_state__": {...}}``.

    Serialization strategy (checked in order):

    1. If the object has a custom ``__getstate__`` (not :func:`object.__getstate__`),
       calls it to obtain the state dict.
    2. If the object is a dataclass, uses :func:`dataclasses.asdict`.
    3. Otherwise, takes all public (non-underscore-prefixed) entries from ``__dict__``.

    Args:
        obj: The object to serialize.

    Returns:
        A JSON-serializable dict that :func:`load_object` can reconstruct.
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

    Imports the module identified by ``__llmeter_class__``, instantiates the class
    (bypassing ``__init__`` via ``__new__``), and calls ``__setstate__`` with the
    persisted state dict.

    Args:
        data: A dict with ``__llmeter_class__`` and ``__llmeter_state__`` keys, as
            produced by :func:`dump_object`.

    Returns:
        The reconstructed object instance.

    .. warning::
        Do not call on data from untrusted sources — it imports and instantiates
        arbitrary classes.
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

_SERIALIZERS: list[tuple[type | tuple[type, ...], Any]] = [
    (bytes, lambda v: {"__llmeter_bytes__": base64.b64encode(v).decode("utf-8")}),
    (datetime, datetime_to_str),
    (os.PathLike, lambda v: Path(v).as_posix()),
]

_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _serialize_value(val: Any) -> Any:
    """Recursively prepare a value for JSON persistence.

    Handles primitives, known types (bytes, datetime, PathLike), nested
    :class:`Serializable` objects (via :func:`dump_object`), dicts, and lists/tuples.
    Raises :exc:`TypeError` for objects it cannot serialize.
    """
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    for types, fn in _SERIALIZERS:
        if isinstance(val, types):
            return fn(val)
    if (
        hasattr(val, "__getstate__")
        and type(val).__getstate__ is not object.__getstate__
    ):
        return dump_object(val)
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(item) for item in val]
    raise TypeError(f"Cannot serialize {type(val).__name__!r} object: {val!r}")


def _deserialize_value(val: Any) -> Any:
    """Recursively restore a value from JSON persistence.

    Recognizes type-tagged dicts (``__llmeter_class__``), bytes markers
    (``__llmeter_bytes__``), ISO-8601 datetime strings, and recursively processes
    nested dicts and lists.
    """
    match val:
        case None | bool() | int() | float():
            return val
        case str() if _DATETIME_RE.fullmatch(val):
            return str_to_datetime(val)
        case str():
            return val
        case {"__llmeter_class__": _, "__llmeter_state__": _}:
            return load_object(val)
        case {"__llmeter_bytes__": b64} if len(val) == 1:
            return base64.b64decode(b64)
        case dict():
            return {k: _deserialize_value(v) for k, v in val.items()}
        case list() | tuple():
            return [_deserialize_value(item) for item in val]
        case _:
            return val
