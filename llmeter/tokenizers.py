# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

from .serialization import Serializable


class Tokenizer(Serializable, ABC):
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
    def load(tokenizer_info: dict) -> "Tokenizer":
        """Load a tokenizer from a dictionary.

        This supports configs saved before the unified ``dump_object``/``load_object``
        serialization was introduced. New code should use
        :func:`~llmeter.serialization.load_object` instead.

        Args:
            tokenizer_info (dict): The tokenizer information to load. Must include at minimum
                a ``tokenizer_module`` key.

        Returns:
            Tokenizer: The loaded tokenizer.
        """
        return _load_tokenizer_from_info(tokenizer_info)


def _load_tokenizer_from_info(tokenizer_info: dict) -> Tokenizer:
    """Load a tokenizer from a legacy info dictionary.

    Args:
        tokenizer_info (dict): The tokenizer information to load.

    Returns:
        Tokenizer: The loaded tokenizer.
    """
    module = tokenizer_info["tokenizer_module"]

    if module == "transformers":
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_info["name"])  # type: ignore

    if module == "tiktoken":
        from tiktoken import get_encoding

        return get_encoding(tokenizer_info["name"])  # type: ignore

    if module == "llmeter":
        return DummyTokenizer()

    raise ValueError(f"Unknown tokenizer module: {module}")


class DummyTokenizer(Tokenizer):
    """A dummy tokenizer that splits the input text on whitespace and returns the tokens as is.

    This tokenizer will generally under-estimate token counts in English and latin languages (where
    words comprise more than one token on average), and will give very poor results for languages
    where the whitespace/"word" heuristic doesn't work well (e.g. Chinese, Japanese, Korean, Thai).

    However, it requires no dependencies beyond the Python standard library, using `str.split()`
    """

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text: str) -> list[str]:
        return [k for k in text.split()]

    def decode(self, tokens: list[str]) -> str:
        return " ".join(k for k in tokens)
