# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON encoding and decoding helpers used across LLMeter.

Provides a ``default``-compatible serializer function and a matching decoder hook
for round-tripping binary content (``bytes``) and Python class instances through
JSON via special marker objects, while also handling ``datetime``,
``os.PathLike``, and objects that implement ``to_dict()``.

Marker conventions
------------------
* ``__llmeter_bytes__`` – base-64 encoded ``bytes`` payload.
* ``__llmeter_class__`` – fully-qualified ``module:ClassName`` string that
  enables dynamic re-instantiation of the correct Python class on
  deserialization.

Example::

    import json
    from llmeter.json_utils import llmeter_default_serializer, llmeter_default_deserializer

    payload = {"image": {"bytes": b"\\xff\\xd8\\xff\\xe0"}}

    # Serialize
    encoded = json.dumps(payload, default=llmeter_default_serializer)

    # Deserialize (bytes are restored automatically)
    decoded = json.loads(encoded, object_hook=llmeter_default_deserializer)
    assert decoded == payload
"""

import base64
import importlib
import logging
import os
from datetime import date, datetime, time, timezone
from typing import Any

from upath import UPath as Path

logger = logging.getLogger(__name__)


def llmeter_default_serializer(obj: Any) -> Any:
    """Serialize a single non-JSON-serializable object.

    Intended for use as the ``default`` argument to :func:`json.dumps` or
    :func:`json.dump`.

    Type handling (checked in order):

    * Objects with a ``to_dict()`` method — delegates to that method.
    * ``bytes`` — wrapped in a ``{"__llmeter_bytes__": "<base64>"}`` marker so
      that :func:`llmeter_bytes_decoder` can restore them on the way back.
    * ``datetime`` — converted to a UTC ISO-8601 string with a ``Z`` suffix.
    * ``date`` / ``time`` — converted via ``.isoformat()``.
    * ``os.PathLike`` — converted to a POSIX path string.
    * Anything else — ``str()`` fallback (returns ``None`` if that also fails).

    Args:
        obj: The object that the default encoder could not handle.

    Returns:
        A JSON-serializable representation of *obj*.

    Example::

        >>> import json
        >>> from llmeter.json_utils import llmeter_default_serializer
        >>> json.dumps({"ts": datetime(2024, 1, 1)}, default=llmeter_default_serializer)
        '{"ts": "2024-01-01T00:00:00"}'
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        result = obj.to_dict()
        if not isinstance(result, dict):
            # This check guards against infinite recursion in case something tries to serialize a
            # MagicMock object with this function (in which to_dict returns another mock)
            raise TypeError(
                f"{type(obj).__name__}.to_dict() returned {type(result).__name__}, expected dict"
            )
        return result
    if isinstance(obj, bytes):
        return {"__llmeter_bytes__": base64.b64encode(obj).decode("utf-8")}
    if isinstance(obj, datetime):
        if obj.tzinfo is not None:
            obj = obj.astimezone(timezone.utc)
        return obj.isoformat(timespec="seconds").replace("+00:00", "Z")
    if isinstance(obj, (date, time)):
        return obj.isoformat()
    if isinstance(obj, (os.PathLike, Path)):
        return Path(obj).as_posix()
    try:
        return str(obj)
    except Exception:
        return None


def _resolve_class(qualified_name: str) -> type:
    """Import and return a class from a ``module:ClassName`` string.

    Args:
        qualified_name: A string in ``module:ClassName`` format, e.g.
            ``"llmeter.callbacks.cost.model:CostModel"``.

    Returns:
        The resolved Python class.

    Raises:
        ValueError: If *qualified_name* is not in ``module:ClassName`` format.
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    if ":" not in qualified_name:
        raise ValueError(
            f"__llmeter_class__ must be in 'module:ClassName' format, got: {qualified_name!r}"
        )
    module_path, class_name = qualified_name.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def llmeter_bytes_decoder(dct: dict) -> dict | bytes:
    """Decode ``__llmeter_bytes__`` marker objects back to ``bytes``.

    Intended for use as the ``object_hook`` argument to :func:`json.loads` or
    :func:`json.load`.  Marker objects produced by :func:`llmeter_default_serializer`
    are detected and converted back to ``bytes``; all other dicts pass through
    unchanged.

    Args:
        dct: A dictionary produced by the JSON parser.

    Returns:
        The original ``bytes`` if *dct* is a marker object, otherwise *dct* unchanged.

    Example::

        >>> import json
        >>> from llmeter.json_utils import llmeter_bytes_decoder
        >>> json.loads('{"__llmeter_bytes__": "/9j/4A=="}', object_hook=llmeter_bytes_decoder)
        b'\\xff\\xd8\\xff\\xe0'
    """
    if "__llmeter_bytes__" in dct and len(dct) == 1:
        return base64.b64decode(dct["__llmeter_bytes__"])
    return dct


def llmeter_default_deserializer(dct: dict) -> Any:
    """Decode LLMeter marker objects back to their Python representations.

    Handles both ``__llmeter_bytes__`` (base-64 → ``bytes``) and
    ``__llmeter_class__`` (dynamic class instantiation via ``from_dict``).

    Intended for use as the ``object_hook`` argument to :func:`json.loads` or
    :func:`json.load`.

    When a dict contains an ``__llmeter_class__`` key the referenced class is
    imported and its ``from_dict`` classmethod is called with the remaining
    dictionary contents.  If the class does not define ``from_dict``, the
    remaining contents are passed as keyword arguments to the constructor.

    Args:
        dct: A dictionary produced by the JSON parser.

    Returns:
        The decoded Python object, or *dct* unchanged if no marker is present.

    Example::

        >>> import json
        >>> from llmeter.json_utils import llmeter_default_deserializer
        >>> json.loads('{"__llmeter_bytes__": "/9j/4A=="}',
        ...            object_hook=llmeter_default_deserializer)
        b'\\xff\\xd8\\xff\\xe0'
    """
    # bytes marker (single-key dict)
    if "__llmeter_bytes__" in dct and len(dct) == 1:
        return base64.b64decode(dct["__llmeter_bytes__"])

    # class marker — dynamically instantiate the right Python class
    if "__llmeter_class__" in dct:
        qualified_name = dct.pop("__llmeter_class__")
        try:
            cls = _resolve_class(qualified_name)
        except (ValueError, ImportError, AttributeError) as exc:
            logger.warning(
                "Could not resolve __llmeter_class__ %r: %s. "
                "Returning raw dict instead.",
                qualified_name,
                exc,
            )
            dct["__llmeter_class__"] = qualified_name  # restore for transparency
            return dct

        if hasattr(cls, "from_dict") and callable(cls.from_dict):
            return cls.from_dict(dct)
        return cls(**dct)

    return dct
