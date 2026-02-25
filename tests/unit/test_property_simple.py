# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simplified property-based tests for llmeter components."""

import json
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from llmeter.endpoints.base import InvocationResponse
from llmeter.endpoints.openai import OpenAICompletionEndpoint
from llmeter.prompt_utils import load_payloads, save_payloads
from llmeter.tokenizers import DummyTokenizer, _to_dict


# Custom strategies
@composite
def valid_openai_messages(draw):
    """Generate valid OpenAI message structures."""
    num_messages = draw(st.integers(min_value=1, max_value=10))
    messages = []
    for _ in range(num_messages):
        content = draw(st.text(min_size=1, max_size=500))
        role = draw(st.sampled_from(["user", "assistant", "system"]))
        messages.append({"role": role, "content": content})
    return messages


@composite
def openai_payload_strategy(draw):
    """Generate valid OpenAI API payloads."""
    messages = draw(valid_openai_messages())
    max_tokens = draw(st.integers(min_value=1, max_value=4096))
    return {"messages": messages, "max_tokens": max_tokens}


# Tokenizer property tests
class TestTokenizerProperties:
    """Property-based tests for tokenizer functionality."""

    @given(st.text(min_size=0, max_size=10000))
    def test_dummy_tokenizer_encode_decode_preserves_words(self, text):
        """Encoding then decoding should preserve whitespace-separated words."""
        tokenizer = DummyTokenizer()
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # The decoded text should have the same words
        assert decoded.split() == text.split()

    @given(st.text(min_size=0, max_size=1000))
    def test_dummy_tokenizer_encode_returns_list(self, text):
        """Encode should always return a list of strings."""
        tokenizer = DummyTokenizer()
        result = tokenizer.encode(text)
        assert isinstance(result, list)
        assert all(isinstance(token, str) for token in result)

    @given(st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=100))
    def test_dummy_tokenizer_decode_returns_string(self, tokens):
        """Decode should always return a string."""
        tokenizer = DummyTokenizer()
        result = tokenizer.decode(tokens)
        assert isinstance(result, str)

    @given(st.text(min_size=0, max_size=1000))
    def test_tokenizer_serialization(self, text):
        """Tokenizer should serialize correctly."""
        tokenizer = DummyTokenizer()
        tokenizer_dict = _to_dict(tokenizer)

        assert isinstance(tokenizer_dict, dict)
        assert "tokenizer_module" in tokenizer_dict
        assert tokenizer_dict["tokenizer_module"] == "llmeter"


# OpenAI endpoint property tests
class TestOpenAIEndpointProperties:
    """Property-based tests for OpenAI endpoint functionality."""

    @given(openai_payload_strategy())
    def test_parse_payload_always_returns_string(self, payload):
        """_parse_payload should always return a string."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )
        result = endpoint._parse_payload(payload)
        assert isinstance(result, str)

    @given(st.text(min_size=1, max_size=1000))
    def test_create_payload_single_message_structure(self, message):
        """create_payload with single message should produce valid structure."""
        payload = OpenAICompletionEndpoint.create_payload(message)

        assert "messages" in payload
        assert "max_tokens" in payload
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == message

    @given(st.lists(st.text(min_size=1, max_size=500), min_size=1, max_size=10))
    def test_create_payload_preserves_all_messages(self, messages):
        """create_payload should preserve all input messages."""
        payload = OpenAICompletionEndpoint.create_payload(messages)

        assert "messages" in payload
        assert len(payload["messages"]) == len(messages)
        for i, msg in enumerate(messages):
            assert payload["messages"][i]["content"] == msg

    @given(st.text(min_size=1, max_size=500), st.integers(min_value=1, max_value=4096))
    def test_create_payload_respects_max_tokens(self, message, max_tokens):
        """create_payload should respect max_tokens parameter."""
        payload = OpenAICompletionEndpoint.create_payload(
            message, max_tokens=max_tokens
        )
        assert payload["max_tokens"] == max_tokens

    @given(st.dictionaries(st.text(min_size=1, max_size=20), st.integers()))
    def test_parse_payload_handles_missing_messages_gracefully(self, payload):
        """_parse_payload should handle payloads without messages."""
        endpoint = OpenAICompletionEndpoint(
            model_id="gpt-3.5-turbo", api_key="test_key"
        )
        # Should not raise, should return empty string
        result = endpoint._parse_payload(payload)
        assert isinstance(result, str)


# InvocationResponse property tests
class TestInvocationResponseProperties:
    """Property-based tests for InvocationResponse."""

    @given(
        st.text(min_size=1, max_size=50),
        st.text(max_size=1000),
        st.floats(min_value=0, max_value=120, allow_nan=False),
    )
    def test_invocation_response_minimal_construction(self, id_val, text, ttlt):
        """InvocationResponse should work with minimal required fields."""
        response = InvocationResponse(
            id=id_val, response_text=text, time_to_last_token=ttlt
        )

        assert response.id == id_val
        assert response.response_text == text
        assert response.time_to_last_token == ttlt

    @given(
        st.text(min_size=1, max_size=50),
        st.text(max_size=500),
        st.floats(min_value=0, max_value=120, allow_nan=False),
    )
    def test_invocation_response_to_dict_always_succeeds(self, id_val, text, ttlt):
        """to_dict should always succeed for any valid response."""
        response = InvocationResponse(
            id=id_val, response_text=text, time_to_last_token=ttlt
        )
        result = response.to_dict()
        assert isinstance(result, dict)
        assert "id" in result

    @given(
        st.text(min_size=1, max_size=50),
        st.text(max_size=500),
        st.floats(min_value=0, max_value=120, allow_nan=False),
    )
    def test_invocation_response_to_json_is_valid(self, id_val, text, ttlt):
        """to_json should produce valid JSON."""
        response = InvocationResponse(
            id=id_val, response_text=text, time_to_last_token=ttlt
        )
        json_str = response.to_json()
        # Should be parseable
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    @given(st.text(min_size=1, max_size=50), st.text(max_size=100))
    def test_invocation_response_error_output_factory(self, id_val, error_msg):
        """error_output factory should create valid error responses."""
        error_response = InvocationResponse.error_output(
            input_payload={"test": "payload"}, id=id_val, error=error_msg
        )

        assert error_response.error == error_msg
        assert error_response.id == id_val
        assert error_response.input_payload == {"test": "payload"}


# Payload file I/O property tests
class TestPayloadIOProperties:
    """Property-based tests for payload loading and saving."""

    @given(
        st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=50),
                st.one_of(
                    st.text(max_size=200), st.integers(), st.floats(allow_nan=False)
                ),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(deadline=None)
    def test_save_load_payloads_roundtrip(self, payloads):
        """Saving and loading payloads should preserve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Save payloads
            saved_path = save_payloads(payloads, output_path)
            assert saved_path.exists()

            # Load payloads back
            loaded_payloads = list(load_payloads(saved_path))

            # Should have same number of payloads
            assert len(loaded_payloads) == len(payloads)

            # Each payload should match
            for original, loaded in zip(payloads, loaded_payloads):
                assert loaded == original

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(st.text(max_size=200), st.integers()),
            min_size=1,
            max_size=10,
        )
    )
    @settings(deadline=None)
    def test_save_single_payload_as_dict(self, payload):
        """save_payloads should handle single dict input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Save single payload as dict
            saved_path = save_payloads(payload, output_path)
            assert saved_path.exists()

            # Load it back
            loaded_payloads = list(load_payloads(saved_path))
            assert len(loaded_payloads) == 1
            assert loaded_payloads[0] == payload

    @given(
        st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.text(max_size=100),
                min_size=1,
                max_size=5,
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(deadline=None)
    def test_jsonl_format_one_object_per_line(self, payloads):
        """JSONL files should have one JSON object per line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            saved_path = save_payloads(payloads, output_path)

            # Read raw file
            with open(saved_path, "r") as f:
                lines = f.readlines()

            # Should have one line per payload
            assert len(lines) == len(payloads)

            # Each line should be valid JSON
            for line in lines:
                parsed = json.loads(line.strip())
                assert isinstance(parsed, dict)


# String handling property tests
class TestStringHandlingProperties:
    """Property-based tests for string handling edge cases."""

    @given(st.text(max_size=1000))
    def test_tokenizer_handles_any_unicode(self, text):
        """Tokenizer should handle any Unicode text."""
        tokenizer = DummyTokenizer()
        # Should not raise
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)

    @given(
        st.text(
            min_size=0,
            max_size=500,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        )
    )
    def test_openai_payload_with_alphanumeric(self, text):
        """OpenAI payload creation should handle alphanumeric text."""
        if text:  # Only test non-empty strings
            payload = OpenAICompletionEndpoint.create_payload(text)
            assert payload["messages"][0]["content"] == text
