# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Dict, TypeVar

from .serde import JSONableBase

TTokenizer = TypeVar("TTokenizer", bound="Tokenizer")


class Tokenizer(JSONableBase, ABC):
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

    # Load any built-in Endpoint type (or custom ones) from a plain JSON dictionary

    @classmethod
    def from_dict(
        cls, raw: dict, alt_classes: Dict[str, TTokenizer] = {}, **kwargs
    ) -> TTokenizer:
        """Load any built-in Tokenizer type (or custom ones) from a plain JSON dictionary

        Args:
            raw: A plain Tokenizer config dictionary, as created with `to_dict()`, `to_json`, etc.
            alt_classes (Dict[str, type[Endpoint]]): A dictionary mapping additional custom type
                names (beyond those in `llmeter.tokenizers`, which are included automatically), to
                corresponding classes for loading custom endpoint types.
            **kwargs: Optional extra keyword arguments to pass to the constructor

        Returns:
            endpoint: An instance of the appropriate endpoint class, initialized with the
                configuration from the file.
        """
        builtin_endpoint_types = import_module("llmeter.tokenizers")
        class_map = {
            **builtin_endpoint_types,
            **alt_classes,
        }
        return super().from_dict(raw, alt_classes=class_map, **kwargs)


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


class TikTokenTokenizer(Tokenizer):
    """A tokenizer based on TikToken get_encoding

    (Note: You must have the `tiktoken` library installed to use this in LLMeter)
    """

    name: str

    def __init__(self, name: str):
        from tiktoken import get_encoding

        self._tokenizer = get_encoding(name)

    def encode(self, text: str):
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[str]):
        return self._tokenizer.decode(tokens)


class TransformersAutoTokenizer(Tokenizer):
    """A tokenizer based on Hugging Face Transformers' AutoTokenizer

    (Note: You must have the `transformers` library installed to use this in LLMeter)
    """

    name_or_path: str

    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    def encode(self, text: str):
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[str]):
        return self._tokenizer.decode(tokens)
