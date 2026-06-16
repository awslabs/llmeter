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


def default_getstate(obj: Any) -> dict:
    """Default __getstate__: infers state from __init__ signature.

    Matches __init__ parameter names to instance attributes (self.name or
    self._name). Parameters without a match are omitted -- __init__ will use
    defaults on reconstruction.
    """
    sig = inspect.signature(obj.__init__)
    state = {}
    for name, param in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if hasattr(obj, name):
            state[name] = getattr(obj, name)
        elif hasattr(obj, f"_{name}"):
            state[name] = getattr(obj, f"_{name}")
    return state


def default_setstate(obj: Any, state: dict) -> None:
    """Default __setstate__: calls __init__(**state) to reconstruct."""
    obj.__init__(**state)
