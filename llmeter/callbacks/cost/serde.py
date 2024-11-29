# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""(De/re)serialization interfaces for saving Cost Model objects to file and loading them back"""

# Python Built-Ins:
from dataclasses import is_dataclass
from datetime import date, datetime, time
import json
import logging
import os
from typing import Any, Callable, Dict, Protocol, Type, TypeVar

# External Dependencies:
from upath import UPath as Path

logger = logging.getLogger(__name__)

TSerializable = TypeVar("TSerializable", bound="ISerializable")


class ISerializable(Protocol):
    def to_dict(self) -> dict:
        """Represent this object with a fully JSON-serializable dictionary"""
        ...

    @classmethod
    def from_dict(cls, **kwargs) -> TSerializable:
        """Load an instance of this class from a JSON-able dictionary representation"""
        ...


def is_dataclass_instance(obj):
    """Check whether `obj` is an instance of any dataclass
    See: https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
    """
    return is_dataclass(obj) and not isinstance(obj, type)


def to_dict_recursive_generic(obj: object, **kwargs) -> dict:
    """Convert a vaguely dataclass-like object (with maybe IJSONable fields) to a JSON-ready dict
    The output dict is augmented with `_type` storing the `__class__.__name__` of the provided
    `obj`.
    Args:
        obj: The object to convert
        **kwargs: Optional extra parameters to insert in the output dictionary
    """
    obj_classname = obj.__class__.__name__
    result = {} if obj_classname == "dict" else {"_type": obj.__class__.__name__}
    if hasattr(obj, "__dict__"):
        # We *don't* use dataclass asdict() here because we want our custom behaviour instead of its
        # recursion:
        result.update(obj.__dict__)
    elif (
        isinstance(obj, dict)
        or hasattr(obj, "__keys__")
        and hasattr(obj, "__getitem__")
    ):
        result.update(obj)
    else:
        result.update({k: getattr(obj, k) for k in dir(obj)})
    result.update(kwargs)
    for k, v in result.items():
        if hasattr(v, "to_dict"):
            result[k] = v.to_dict()
        elif isinstance(v, dict):
            result[k] = to_dict_recursive_generic(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [to_dict_recursive_generic(item) for item in v]
        elif isinstance(v, (date, datetime, time)):
            result[k] = v.isoformat()
    return result


TFromDict = TypeVar("TFromDict")


def from_dict_with_class(raw: dict, cls: Type[TFromDict], **kwargs) -> TFromDict:
    """Initialize an instance of a class from a plain dict (with optional extra kwargs)
    If the input dictionary contains a `_type` key, and this doesn't match the provided
    `cls.__name__`, a warning will be logged.
    Args:
        raw: A plain Python dict, for example loaded from a JSON file
        cls: The class to create an instance of
        **kwargs: Optional extra keyword arguments to pass to the constructor
    """
    raw_args = {k: v for k, v in raw.items()}
    raw_type = raw_args.pop("_type", None)
    if raw_type is not None and raw_type != cls.__name__:
        logger.warning(
            "from_dict: _type '%s' doesn't match class '%s' being loaded. %s"
            % (raw_type, cls.__name__, raw)
        )
    return cls(**raw_args, **kwargs)


def from_dict_with_class_map(
    raw: dict, class_map: Dict[str, Type[TFromDict]], **kwargs
) -> TFromDict:
    """Initialize an instance of a class from a plain dict (with optional extra kwargs)
    Args:
        raw: A plain Python dict which must contain a `_type` key
        classes: A mapping from `_type` string to class to create an instance of
        **kwargs: Optional extra keyword arguments to pass to the constructor
    """
    if "_type" not in raw:
        raise ValueError("from_dict_with_class_map: No _type in raw dict: %s" % raw)
    if raw["_type"] not in class_map:
        raise ValueError(
            "Object _type '%s' not found in provided class_map %s"
            % (raw["_type"], class_map)
        )
    return from_dict_with_class(raw, class_map[raw["_type"]], **kwargs)


TJSONable = TypeVar("TJSONable", bound="JSONableBase")


class JSONableBase:
    """A base class for speeding up implementation of JSON-serializable objects"""

    @classmethod
    def from_dict(
        cls: Type[TJSONable],
        raw: dict,
        alt_classes: Dict[str, TJSONable] = {},
        **kwargs,
    ) -> TJSONable:
        """Initialize an instance of this class from a plain dict (with optional extra kwargs)

        Args:
            raw: A plain Python dict, for example loaded from a JSON file
            alt_classes: By default, this method will only use the class of the current object
                (i.e. `cls`). If you want to support loading of subclasses, provide a mapping
                from your raw dict's `_type` field to class, for example `{cls.__name__: cls}`.
            **kwargs: Optional extra keyword arguments to pass to the constructor
        """
        if alt_classes:
            return from_dict_with_class_map(
                raw=raw,
                class_map={cls.__name__: cls, **alt_classes},
                **kwargs,
            )
        else:
            return from_dict_with_class(raw=raw, cls=cls, **kwargs)

    @classmethod
    def from_file(cls: Type[TJSONable], input_path: os.PathLike, **kwargs) -> TJSONable:
        """Initialize an instance of this class from a (local or Cloud) JSON file
        Args:
            input_path: The path to the JSON data file.
            **kwargs: Optional extra keyword arguments to pass to `from_dict()`
        """
        input_path = Path(input_path)
        with input_path.open("r") as f:
            return cls.from_json(f.read(), **kwargs)

    @classmethod
    def from_json(cls: Type[TJSONable], json_string: str, **kwargs) -> TJSONable:
        """Initialize an instance of this class from a JSON string (with optional extra kwargs)

        Args:
            json_string: A string containing valid JSON data
            **kwargs: Optional extra keyword arguments to pass to `from_dict()``
        """
        return cls.from_dict(json.loads(json_string), **kwargs)

    def to_dict(self, **kwargs) -> dict:
        """Save the state of the object to a JSON-dumpable dictionary (with optional extra kwargs)

        Implementers of this method should ensure that the returned dict is fully JSON-compatible:
        Mapping any child fields from Python classes to dicts if necessary, avoiding any circular
        references, etc.
        """
        return to_dict_recursive_generic(self, **kwargs)

    def to_file(
        self,
        output_path: os.PathLike,
        indent: int | str | None = 4,
        default: Callable[[Any], Any] | None = str,
        **kwargs,
    ) -> Path:
        """Save the state of the object to a (local or Cloud) JSON file

        Args:
            output_path: The path where the configuration file will be saved.
            indent: Optional indentation passed through to `to_json()` and therefore `json.dumps()`
            default: Optional function to convert non-JSON-serializable objects to strings, passed
                through to `to_json()` and therefore to `json.dumps()`
            **kwargs: Optional extra keyword arguments to pass to `to_json()`

        Returns:
            output_path: Universal Path representation of the target file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(self.to_json(indent=indent, default=default, **kwargs))
        return output_path

    def to_json(self, **kwargs) -> str:
        """Serialize this object to JSON, with optional kwargs passed through to `json.dumps()`"""
        return json.dumps(self.to_dict(), **kwargs)
