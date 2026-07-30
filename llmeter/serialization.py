# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified JSON-based serialization for LLMeter objects.

Most of the objects we'd like to save to and load from file in LLMeter are configuration-like and
*almost* JSON-compatible, but with some extensions (like datetimes, binary image/etc payloads,
and callback objects). Rather than falling back to Pickle (which is Python-specific,
non-human-readable, and Python version fragile) - we implement in this module a JSON-based scheme
for saving and loading compatible LLMeter objects.

!!! warning "A warning on security"

    [`load_object`][llmeter.serialization.load_object] imports and instantiates whatever class path
    is in the `__llmeter_class__` field. Like unpickle, this has the potential to run arbitrary
    code. Do not load configs from untrusted sources!

### Key components

- **[`Serializable`][llmeter.serialization.Serializable]**: Mixin to give any class an automatic
    *state* protocol (`_get_llmeter_state`/`_set_llmeter_state`) by introspecting `__init__`.
    Deliberately distinct from the pickle protocol so `pickle` / `copy` / `deepcopy` keep their
    native behavior.
- **[`dump_object`][llmeter.serialization.dump_object]** and
    **[`load_object`][llmeter.serialization.load_object]**: Full round-trip persistence for
    `Serializable`-compatible objects, using a
    `{"__llmeter_class__": ..., "__llmeter_state__": ...}` envelope.


### Implementation details to be aware of

**State vs. identity:** We split an object's serialized form into two layers. The *state*
(`_get_llmeter_state`) contains the object's own field values and does *not* describe which class
those values belong to. The **envelope** built by
[`dump_object`][llmeter.serialization.dump_object] adds the identity (`__llmeter_class__`) around
that state. The "state" dictionaries handled by `_get_llmeter_state` / `_set_llmeter_state` are
not a fully self-describing representation of the object - only of its state. Keeping identity out
of the state dict avoids polluting the user's field namespace and lets nested polymorphic values
each carry their own envelope.
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

    Timezone-aware datetimes are converted to UTC first. Naive datetimes are serialized as-is
    (assumed UTC).
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
    """Serialize a single non-natively-JSON-serializable object.

    Intended for use as the `default` argument to `json.dump` or `json.dumps`. This **does not**
    handle full recursive LLMeter serialization - see
    [`dump_object`][llmeter.serialization.dump_object] instead.

    Type handling (checked in order):

    * `bytes` — wrapped in a `{"__llmeter_bytes__": "<base64>"}` marker so that
        [`bytes_decoder`][llmeter.serialization.bytes_decoder] can restore them on the way back.
    * `datetime` — converted to a UTC ISO-8601 string with a `Z` suffix.
    * `date` / `time` — converted via `.isoformat()`.
    * `os.PathLike` — converted to a POSIX path string.
    * Anything else — `str()` fallback (returns `None` if that also fails).

    Args:
        obj: The object that the default JSON encoder could not handle.

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
    """Decode `__llmeter_bytes__` marker objects back to Python `bytes`.

    Intended for use as the `object_hook` argument to `json.load` or `json.loads`. Marker objects
    produced by [`json_default`][llmeter.serialization.json_default] are detected and converted
    back to `bytes`; all other dicts pass through unchanged.

    Args:
        dct: A dictionary produced by the JSON parser.

    Returns:
        The original `bytes` if *dct* is an LLMeter bytes marker object, otherwise *dct* unchanged.
    """
    if "__llmeter_bytes__" in dct and len(dct) == 1:
        return base64.b64decode(dct["__llmeter_bytes__"])
    return dct


def _get_type_args(tp) -> tuple:
    """Return the members of a union type (e.g. `datetime | None` -> (datetime, NoneType))."""
    if isinstance(tp, _types.UnionType):
        return tp.__args__
    origin = getattr(tp, "__origin__", None)
    if origin is _types.UnionType:
        return tp.__args__
    return (tp,) if isinstance(tp, type) else ()


def restore_dataclass_types(cls: type, data: dict) -> None:
    """Restore typed fields in a dict destined for a dataclass constructor.

    Introspects `cls` (a dataclass) and converts JSON-native values back to their annotated Python
    types. Currently handles:

    * `datetime` fields — parses ISO-8601 strings via `str_to_datetime`.
    * `bytes` fields — decodes `__llmeter_bytes__` markers via base64.

    Only fields declared on `cls` are touched — nested user payloads (e.g. `input_payload`) are
    left unchanged. Mutates *data* in place.

    Args:
        cls: A dataclass type to introspect for field type annotations.
        data: A dictionary of field values (e.g. from `json.load`) to coerce.
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
    """Mixin providing a state extraction protocol compatible with LLMeter serialization.

    Serialization in LLMeter uses a state extraction protocol somewhat similar to, but deliberately
    separate from, the (`__getstate__` / `__setstate__`) interface used by `pickle`, `copy.copy`,
    and `copy.deepcopy`. Subclasses remain natively picklable/copyable, but can also
    be saved to and loaded from LLMeter's JSON-based format.

    This default implementation works with plain classes, `@dataclass`, and any class whose
    `__init__` parameters correspond to instance attributes (`self.x` or `self._x`). Nested
    `Serializable`-like objects are recursively persisted and loaded via
    [`dump_object`][llmeter.serialization.dump_object] and
    [`load_object`][llmeter.serialization.load_object].

    If your class needs more custom logic to represent and restore its state, customize the
    provided methods by overriding or implementing your own from scratch.
    """

    def _get_llmeter_state(self) -> dict:
        """Extract a JSON-serializable state dict by introspecting `__init__` parameters.

        Returns the object's constructor arguments (looked up as `self.<name>` or `self._<name>`),
        recursively serialized. Note that:

        1. This is state only — it carries no class identity; `dump_object` wraps it with
            `__llmeter_class__`.
        2. Any properties not exposed as `__init__` arguments will not be persisted. Override your
            class' state get and set methods if you need different behaviour.
        """
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

    def _set_llmeter_state(self, state: dict) -> None:
        """Restore this instance from a state dict produced by `_get_llmeter_state`.

        Note this rebuilds the object by *calling the constructor* with the state as keyword
        arguments. [`load_object`][llmeter.serialization.load_object] first creates a bare instance
        with `__new__`, then calls this method to populate it.
        """
        deserialized = {k: _deserialize_value(v) for k, v in state.items()}
        self.__init__(**deserialized)

    def save_to_file(self, path: WritablePathLike) -> Path:
        """Save this object to a JSON file.

        Uses the `_get_llmeter_state` protocol. Override `_get_llmeter_state` (not this method) if
        custom JSON-based serialization is needed.

        Args:
            path: (Local or Cloud) path where the object will be saved.

        Returns:
            The (validated/normalized) path the object was written to.
        """
        path = ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dump_object(self)
        with path.open("w") as f:
            json.dump(data, f, indent=4, default=json_default)
        return path

    @classmethod
    def load_from_file(cls, path: ReadablePathLike) -> "Serializable":
        """Load an object from a JSON file.

        Detects the type from the `__llmeter_class__` field and reconstructs it.

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
    `{"__llmeter_class__": "module.Class", "__llmeter_state__": {...}}`.

    Serialization strategy (checked in order):

    1. If the object implements the LLMeter state protocol (as in
        [`Serializable._get_llmeter_state`][llmeter.serialization.Serializable._get_llmeter_state])
        this will call it to obtain the state dict.
    2. If the object is a dataclass, uses `dataclasses.asdict`.
    3. Otherwise, takes all public (non-underscore-prefixed) entries from `__dict__`.

    Args:
        obj: The object to serialize.

    Returns:
        dct: A JSON-serializable dict that [`load_object`][llmeter.serialization.load_object] can
            reconstruct.
    """
    class_path = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
    if hasattr(obj, "_get_llmeter_state"):
        state = obj._get_llmeter_state()
    elif is_dataclass(obj) and not isinstance(obj, type):
        state = asdict(obj)
    elif hasattr(obj, "__dict__"):
        state = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    else:
        state = {}
    return {"__llmeter_class__": class_path, "__llmeter_state__": state}


def load_object(data: dict) -> Any:
    """Restore an object from a type-tagged dict produced by `dump_object`.

    !!! warning
        This method imports and instantiates class paths specified by the input data, which (like
        pickle) can enable arbitrary running arbitrary code. Do not run it on data from unstrusted
        sources!

    Imports the module identified by `__llmeter_class__`, instantiates the class (bypassing
    `__init__` via `__new__`), and restores its state via
    [`Serializable._set_llmeter_state`][llmeter.serialization.Serializable._set_llmeter_state].
    Objects that do not implement the protocol are restored by assigning the (deserialized) state
    onto their `__dict__`.

    Args:
        data: A dict with `__llmeter_class__` and `__llmeter_state__` keys, as
            produced by [`dump_object`][llmeter.serialization.dump_object].

    Returns:
        The reconstructed object instance.
    """
    class_path = data["__llmeter_class__"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    state = data["__llmeter_state__"]
    obj = cls.__new__(cls)
    if hasattr(obj, "_set_llmeter_state"):
        obj._set_llmeter_state(state)
    else:
        # Fallback for non-Serializable classes (state came from dump_object's dataclass /
        # __dict__ branches). We assign attributes directly rather than calling __setstate__:
        # `object` has no __setstate__ (so it would raise for a plain class), and if the class
        # *does* define one it's the pickle hook, which may expect a different state shape than
        # our JSON dict. Direct assignment (with per-value deserialization) is the safe generic
        # restore, and mirrors pickle's own "restore __dict__ without calling __init__" default.
        for key, value in state.items():
            setattr(obj, key, _deserialize_value(value))
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
    [`Serializable`][llmeter.serialization.Serializable] objects (via
    [`dump_object`][llmeter.serialization.dump_object]), dicts, and lists/tuples.

    Raises:
        TypeError: for objects it cannot serialize.
    """
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    for types, fn in _SERIALIZERS:
        if isinstance(val, types):
            return fn(val)
    if hasattr(val, "_get_llmeter_state"):
        return dump_object(val)
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(item) for item in val]
    raise TypeError(
        f"Cannot serialize {type(val).__name__!r} object: {val!r}. "
        "Only JSON primitives, bytes, datetime, PathLike, dicts, lists/tuples, and "
        "LLMeter-serializable objects are supported. To make a custom object "
        "serializable (e.g. a custom cost dimension or callback), have its class "
        "inherit from llmeter.serialization.Serializable."
    )


def _deserialize_value(val: Any) -> Any:
    """Recursively restore a value from JSON persistence.

    Recognizes type-tagged dicts (`__llmeter_class__`), bytes markers (`__llmeter_bytes__`),
    ISO-8601 datetime strings, and recursively processes nested dicts and lists.
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
