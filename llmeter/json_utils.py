# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON encoding and decoding helpers used across LLMeter

Provides a unified :class:`json.JSONEncoder` subclass and a matching decoder hook for
round-tripping binary content (``bytes``) through JSON via base64 marker objects, while also
handling ``datetime``, ``os.PathLike``, and objects that implement ``to_dict()``.

Example::

    import json
    from llmeter.json_utils import LLMeterEncoder, llmeter_bytes_decoder

    payload = {"image": {"bytes": b"\\xff\\xd8\\xff\\xe0"}}

    # Serialize
    encoded = json.dumps(payload, cls=LLMeterEncoder)

    # Deserialize (bytes are restored automatically)
    decoded = json.loads(encoded, object_hook=llmeter_bytes_decoder)
    assert decoded == payload
"""

import base64
import json
import os
from datetime import date, datetime, time, timezone
from typing import Any

from upath import UPath as Path


class LLMeterEncoder(json.JSONEncoder):
    """JSON encoder that handles common non-serializable types found in LLMeter.

    Type handling (checked in order):

    * Objects with a ``to_dict()`` method — delegates to that method.
    * ``bytes`` — wrapped in a ``{"__llmeter_bytes__": "<base64>"}`` marker object
      so that :func:`llmeter_bytes_decoder` can restore them on the way back.
    * ``datetime`` — converted to a UTC ISO-8601 string with a ``Z`` suffix.
    * ``date`` / ``time`` — converted via ``.isoformat()``.
    * ``os.PathLike`` — converted to a POSIX path string.
    * Anything else — ``str()`` fallback (returns ``None`` if that also fails).

    Customization:
        To handle additional types, subclass ``LLMeterEncoder`` and override
        :meth:`default`.  Call ``super().default(obj)`` as a fallback so that the
        built-in type handling is preserved.  Because the encoder is used
        consistently across LLMeter (payloads, results, run configs), any type
        that implements a ``to_dict()`` method will be serialized automatically
        without needing a custom encoder.

    Example::

        Subclassing to handle a custom type:

        >>> import json
        >>> import numpy as np
        >>> from llmeter.json_utils import LLMeterEncoder
        >>>
        >>> class MyEncoder(LLMeterEncoder):
        ...     def default(self, obj):
        ...         if isinstance(obj, np.ndarray):
        ...             return obj.tolist()
        ...         return super().default(obj)
        ...
        >>> json.dumps({"data": np.array([1, 2, 3])}, cls=MyEncoder)
        '{"data": [1, 2, 3]}'

        Using ``to_dict()`` (no subclassing needed):

        >>> class MyPayload:
        ...     def __init__(self, model_id, temperature):
        ...         self.model_id = model_id
        ...         self.temperature = temperature
        ...     def to_dict(self):
        ...         return {"model_id": self.model_id, "temperature": self.temperature}
        ...
        >>> json.dumps({"payload": MyPayload("gpt-4", 0.7)}, cls=LLMeterEncoder)
        '{"payload": {"model_id": "gpt-4", "temperature": 0.7}}'
    """

    def default(self, obj: Any) -> Any:
        """Encode a single non-serializable object.

        Args:
            obj: The object that the default encoder could not handle.

        Returns:
            A JSON-serializable representation of *obj*.
        """
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
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


def llmeter_bytes_decoder(dct: dict) -> dict | bytes:
    """Decode ``__llmeter_bytes__`` marker objects back to ``bytes``.

    Intended for use as the ``object_hook`` argument to :func:`json.loads` or
    :func:`json.load`.  Marker objects produced by :class:`LLMeterEncoder` are
    detected and converted back to ``bytes``; all other dicts pass through unchanged.

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
