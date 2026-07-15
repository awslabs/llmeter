# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from llmeter.tokenizers import (
    DummyTokenizer,
    Tokenizer,
    _load_tokenizer_from_info,
)

# Check for optional dependencies
try:
    import transformers  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tiktoken  # noqa: F401

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# Mock classes for testing
class MockTransformersTokenizer:
    name_or_path = "mock-transformer"

    def __init__(self, *args, **kwargs):
        pass


MockTransformersTokenizer.__module__ = "transformers.dummy"


class MockTiktokenTokenizer:
    name = "mock-tiktoken"

    def __init__(self, *args, **kwargs):
        pass


MockTiktokenTokenizer.__module__ = "tiktoken.dummy"


# Test Tokenizer abstract base class
def test_tokenizer_abstract_methods():
    with pytest.raises(TypeError):
        Tokenizer()  # type: ignore


# Test DummyTokenizer
def test_dummy_tokenizer():
    tokenizer = DummyTokenizer()
    text = "This is a test sentence."
    tokens = tokenizer.encode(text)
    assert tokens == ["This", "is", "a", "test", "sentence."]
    assert tokenizer.decode(tokens) == text


# Test Tokenizer.load (legacy format)
def test_load():
    tokenizer_info = {"tokenizer_module": "llmeter"}
    tokenizer = Tokenizer.load(tokenizer_info)
    assert isinstance(tokenizer, DummyTokenizer)


# Test _load_tokenizer_from_info function
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers is not installed")
def test_load_tokenizer_from_info_transformers(monkeypatch):
    def mock_from_pretrained(name):
        return MockTransformersTokenizer()

    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", mock_from_pretrained)

    tokenizer_info = {"tokenizer_module": "transformers", "name": "mock-transformer"}
    tokenizer = _load_tokenizer_from_info(tokenizer_info)
    assert isinstance(tokenizer, MockTransformersTokenizer)


@pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken is not installed")
def test_load_tokenizer_from_info_tiktoken(monkeypatch):
    def mock_get_encoding(name):
        return MockTiktokenTokenizer()

    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", mock_get_encoding)

    tokenizer_info = {"tokenizer_module": "tiktoken", "name": "mock-tiktoken"}
    tokenizer = _load_tokenizer_from_info(tokenizer_info)
    assert isinstance(tokenizer, MockTiktokenTokenizer)


def test_load_tokenizer_from_info_llmeter():
    tokenizer_info = {"tokenizer_module": "llmeter"}
    tokenizer = _load_tokenizer_from_info(tokenizer_info)
    assert isinstance(tokenizer, DummyTokenizer)


def test_load_tokenizer_from_info_unknown():
    tokenizer_info = {"tokenizer_module": "unknown"}
    with pytest.raises(ValueError, match="Unknown tokenizer module"):
        _load_tokenizer_from_info(tokenizer_info)


# Test dump_object/load_object round-trip (new serialization)
def test_serialization_roundtrip():
    from llmeter.serialization import dump_object, load_object

    original = DummyTokenizer()
    data = dump_object(original)

    assert "__llmeter_class__" in data
    assert "llmeter.tokenizers.DummyTokenizer" in data["__llmeter_class__"]

    restored = load_object(data)
    assert isinstance(restored, DummyTokenizer)

    # Behavior preserved
    text = "hello world test"
    assert original.encode(text) == restored.encode(text)
