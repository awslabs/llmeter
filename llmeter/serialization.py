# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified serialization for all LLMeter objects.

Two operations, two purposes:

- ``to_dict(obj)`` - Python dict with native types preserved (datetime stays datetime).
  Use for in-memory access, stats computation, jmespath queries.
- ``serialize(obj)`` - JSON-safe dict (datetimes become strings, etc).
  Use for persistence to disk, JSON output, network transfer.

For objects with runtime state (endpoints, tokenizers, callbacks), ``dump_object``
and ``load_object`` provide full round-trip persistence using the
``__getstate__``/``__setstate__`` protocol::

    data = dump_object(endpoint)   # -> {"_class": "...", "_state": {...}}
    restored = load_object(data)   # -> new Endpoint instance

.. warning:: Security

    ``load_object`` will import and instantiate any class path found in the
    ``_class`` field. Do not load configs from untrusted sources.
"""

import importlib
import inspect
import json as _json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_dict(obj: Any) -> dict:
    """Convert any LLMeter object to a dict with native Python types preserved.

    Dispatch order:
    1. Objects with custom ``__getstate__`` -> ``__getstate__()``
    2. Dataclasses -> ``dataclasses.asdict()``
    3. Plain objects -> filtered ``__dict__`` (public attrs only)
    """
    has_custom_getstate = (
        hasattr(obj, "__getstate__")
        and type(obj).__getstate__ is not object.__getstate__
    )
    if has_custom_getstate:
        return obj.__getstate__()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return dict(obj)


def serialize(obj: Any) -> dict:
    """Convert any LLMeter object to a JSON-safe dict for persistence.

    Same dispatch as ``to_dict`` but intended for JSON output.
    Objects should ensure their ``__getstate__`` returns JSON-serializable data.
    """
    has_custom_getstate = (
        hasattr(obj, "__getstate__")
        and type(obj).__getstate__ is not object.__getstate__
    )
    if has_custom_getstate:
        return obj.__getstate__()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return dict(obj)


def dump_object(obj: Any) -> dict:
    """Serialize an object to a type-tagged dict for full round-trip persistence.

    Returns ``{"_class": "module.ClassName", "_state": {...}}``.
    """
    class_path = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
    try:
        state = serialize(obj)
        _json.dumps(state)  # verify JSON-serializable
    except Exception:
        logger.debug(
            "dump_object: serialize(%s) returned non-serializable data", class_path
        )
        state = {}
    return {"_class": class_path, "_state": state}


def load_object(data: dict) -> Any:
    """Restore an object from a type-tagged state dict.

    WARNING: Do not call on data from untrusted sources.
    """
    class_path = data["_class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    obj = cls.__new__(cls)
    obj.__setstate__(data["_state"])
    return obj


# ---------------------------------------------------------------------------
# Default __getstate__ / __setstate__ implementations
# ---------------------------------------------------------------------------


def _serialize_value(val: Any) -> Any:
    """Recursively serialize a value for JSON persistence.

    - Objects with custom ``__getstate__`` → ``dump_object(val)``
    - Dicts → recurse into values
    - Lists/tuples → recurse into items
    - Scalars (str, int, float, bool, None) → pass through
    """
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if hasattr(val, "__getstate__") and type(val).__getstate__ is not object.__getstate__:
        return dump_object(val)
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(item) for item in val]
    # Fallback: try str
    return str(val)


def _deserialize_value(val: Any) -> Any:
    """Recursively deserialize a value from JSON persistence.

    - Dicts with ``_class``/``_state`` → ``load_object(val)``
    - Other dicts → recurse into values
    - Lists → recurse into items
    - Scalars → pass through
    """
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, dict):
        if "_class" in val and "_state" in val:
            return load_object(val)
        return {k: _deserialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_deserialize_value(item) for item in val]
    return val


def default_getstate(obj: Any) -> dict:
    """Default __getstate__: infers state from __init__ signature.

    Matches __init__ parameter names to instance attributes (self.name or
    self._name). Nested objects with ``__getstate__`` are recursively
    serialized via ``dump_object``. Parameters without a match are omitted.
    """
    sig = inspect.signature(obj.__init__)
    state = {}
    for name, param in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if hasattr(obj, name):
            state[name] = _serialize_value(getattr(obj, name))
        elif hasattr(obj, f"_{name}"):
            state[name] = _serialize_value(getattr(obj, f"_{name}"))
    return state


def default_setstate(obj: Any, state: dict) -> None:
    """Default __setstate__: deserializes nested objects then calls __init__.

    Any dict with ``_class``/``_state`` keys is restored via ``load_object``.
    """
    deserialized = {k: _deserialize_value(v) for k, v in state.items()}
    obj.__init__(**deserialized)
