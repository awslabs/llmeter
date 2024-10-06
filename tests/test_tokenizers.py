import pytest
import json
from llmeter.tokenizers import (
    Tokenizer,
    DummyTokenizer,
    save_tokenizer,
    _to_dict,
    _load_tokenizer_from_info,
)


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


# Test Tokenizer.load_from_file
def test_load_from_file(tmp_path):
    file_path = tmp_path / "tokenizer.json"
    tokenizer_info = {"tokenizer_module": "llmeter"}
    with open(file_path, "w") as f:
        json.dump(tokenizer_info, f)

    tokenizer = Tokenizer.load_from_file(file_path)
    assert isinstance(tokenizer, DummyTokenizer)


def test_load_from_file_none():
    tokenizer = Tokenizer.load_from_file(None)
    assert isinstance(tokenizer, DummyTokenizer)


# Test Tokenizer.load
def test_load():
    tokenizer_info = {"tokenizer_module": "llmeter"}
    tokenizer = Tokenizer.load(tokenizer_info)
    assert isinstance(tokenizer, DummyTokenizer)


# Test Tokenizer.to_dict
def test_to_dict():
    dummy_tokenizer = DummyTokenizer()
    tokenizer_dict = Tokenizer.to_dict(dummy_tokenizer)
    assert tokenizer_dict == {"tokenizer_module": "llmeter"}


# Test _to_dict function
def test_to_dict_transformers(monkeypatch):
    # monkeypatch.setattr("llmeter.tokenizers.AutoTokenizer", MockTransformersTokenizer)
    tokenizer = MockTransformersTokenizer()
    tokenizer_dict = _to_dict(tokenizer)
    assert tokenizer_dict == {
        "tokenizer_module": "transformers",
        "name": "mock-transformer",
    }


def test_to_dict_tiktoken(monkeypatch):
    tokenizer = MockTiktokenTokenizer()
    tokenizer_dict = _to_dict(tokenizer)
    assert tokenizer_dict == {"tokenizer_module": "tiktoken", "name": "mock-tiktoken"}


def test_to_dict_unknown():
    class UnknownTokenizer:
        pass

    with pytest.raises(ValueError, match="Unknown tokenizer module"):
        _to_dict(UnknownTokenizer())


# Test save_tokenizer function
def test_save_tokenizer(tmp_path):
    dummy_tokenizer = DummyTokenizer()
    output_path = tmp_path / "tokenizer.json"
    saved_path = save_tokenizer(dummy_tokenizer, output_path)
    assert saved_path == output_path
    assert output_path.exists()
    with open(output_path, "r") as f:
        saved_info = json.load(f)
    assert saved_info == {"tokenizer_module": "llmeter"}


# Test _load_tokenizer_from_info function
@pytest.mark.skip(reason="transformers is not installed")
def test_load_tokenizer_from_info_transformers(monkeypatch):
    tokenizer_info = {"tokenizer_module": "transformers", "name": "mock-transformer"}
    tokenizer = _load_tokenizer_from_info(tokenizer_info)
    assert isinstance(tokenizer, MockTransformersTokenizer)


@pytest.mark.skip(reason="tiktoken is not installed")
def test_load_tokenizer_from_info_tiktoken(monkeypatch):
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
