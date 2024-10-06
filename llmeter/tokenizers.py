# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

from upath import UPath
import json


class Tokenizer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode(self, text: str):
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: list[str]):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Tokenizer:
            if any("encode" in B.__dict__ for B in C.__mro__) and any(
                "decode" in B.__dict__ for B in C.__mro__
            ):
                return True
        return NotImplemented

    @classmethod
    def __subclasscheck__(cls, subclass):
        required_attrs = getattr(cls, "_required_attrs", [])
        for attr in required_attrs:
            if any("encode" in B.__dict__ for B in subclass.__mro__) and any(
                "decode" in B.__dict__ for B in subclass.__mro__
            ):
                continue
            return False
        return True

    @staticmethod
    def load_from_file(tokenizer_path: UPath | None):
        """
        Loads a tokenizer from a file.

        Args:
            tokenizer_path (UPath): The path to the serialized tokenizer file.

        Returns:
            Tokenizer: The loaded tokenizer.
        """
        if tokenizer_path is None:
            return DummyTokenizer()
        with open(tokenizer_path, "r") as f:
            tokenizer_info = json.load(f)

        return _load_tokenizer_from_info(tokenizer_info)

    @staticmethod
    def load(tokenizer_info: dict):
        """
        Loads a tokenizer from a dictionary.

        Args:
            tokenizer_info (Dict): The tokenizer information to load.

        Returns:
            Tokenizer: The loaded tokenizer.
        """
        return _load_tokenizer_from_info(tokenizer_info)

    @staticmethod
    def to_dict(tokenizer: Any) -> dict:
        """
        Serializes a tokenizer to a dictionary.

        Args:
            tokenizer (Tokenizer): The tokenizer to serialize.

        Returns:
            Dict: The serialized tokenizer.
        """
        return _to_dict(tokenizer)


def _to_dict(tokenizer: Any) -> dict:
    """
    Serializes a tokenizer to a dictionary.

    Args:
        tokenizer (Tokenizer): The tokenizer to serialize.

    Returns:
        Dict: The serialized tokenizer.
    """
    if tokenizer.__module__.split(".")[0] == "transformers":
        return {"tokenizer_module": "transformers", "name": tokenizer.name_or_path}

    if tokenizer.__module__.split(".")[0] == "tiktoken":
        return {"tokenizer_module": "tiktoken", "name": tokenizer.name}

    if tokenizer.__module__.split(".")[0] == "llmeter":
        return {"tokenizer_module": "llmeter"}

    raise ValueError(f"Unknown tokenizer module: {tokenizer.__module__}")


def save_tokenizer(tokenizer: Any, output_path: UPath | str):
    """
    Save a tokenizer information to a file.

    Args:
        tokenizer (Tokenizer): The tokenizer to serialize.
        output_path (UPath): The path to save the serialized tokenizer to.

    Returns:
        UPath: The path to the serialized tokenizer file.
    """
    tokenizer_info = _to_dict(tokenizer)

    output_path = UPath(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tokenizer_info, f)

    return output_path


def _load_tokenizer_from_info(tokenizer_info: dict) -> Tokenizer:
    """
    Loads a tokenizer from a file.

    Args:
        tokenizer_info (Dict): The tokenizer information to load.

    Returns:
        Tokenizer: The loaded tokenizer.
    """
    if tokenizer_info["tokenizer_module"] == "transformers":
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_info["name"])  # type: ignore

    if tokenizer_info["tokenizer_module"] == "tiktoken":
        from tiktoken import get_encoding

        return get_encoding(tokenizer_info["name"])  # type: ignore

    if tokenizer_info["tokenizer_module"] == "llmeter":
        return DummyTokenizer()

    raise ValueError(f"Unknown tokenizer module: {tokenizer_info['tokenizer_module']}")


class DummyTokenizer(Tokenizer):
    """
    A dummy tokenizer that splits the input text on whitespace and returns the tokens as is.

    This tokenizer will generally under-estimate token counts in English and latin languages (where
    words comprise more than one token on average), and will give very poor results for languages
    where the whitespace/"word" heuristic doesn't work well (e.g. Chinese, Japanese, Korean, Thai).

    However, it requires no dependencies beyond the Python standard library, using `str.split()`
    """

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text: str):
        return [k for k in text.split()]

    def decode(self, tokens: list[str]):
        return " ".join(k for k in tokens)
