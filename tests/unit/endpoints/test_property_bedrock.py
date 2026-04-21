# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for Bedrock endpoint process_raw_response logic.

Covers changes from 09f080aa ("refactor(endpoint): process_raw_response")
and the zero-token-count fix in 6dc11572 ("fix(bedrock): use `is not None`
for streaming token count checks"):

- Streaming metadata parsing: 0-valued token counts must not be dropped
- Non-streaming response field extraction
- Text concatenation from content block deltas
- TTFT / TTLT ordering invariants
"""

import time

from hypothesis import given, settings
from hypothesis import strategies as st

from llmeter.endpoints.base import InvocationResponse

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

non_negative_ints = st.integers(min_value=0, max_value=100_000)
optional_non_negative_ints = st.one_of(st.none(), non_negative_ints)

_safe_text = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
)


@st.composite
def bedrock_stream_usage(draw):
    """Generate a Bedrock-style streaming usage dict.

    Values can be 0 (e.g. cacheReadInputTokens on a cache write) — the code
    must not confuse 0 with None.
    """
    return {
        "inputTokens": draw(non_negative_ints),
        "outputTokens": draw(non_negative_ints),
        "cacheReadInputTokens": draw(non_negative_ints),
    }


@st.composite
def bedrock_stream_chunks(draw):
    """Generate a list of Bedrock ConverseStream-style chunks.

    Always produces at least one contentBlockDelta so timing assertions
    are meaningful, followed by a contentBlockStop and a metadata chunk.
    """
    n_deltas = draw(st.integers(min_value=1, max_value=10))
    texts = draw(st.lists(_safe_text, min_size=n_deltas, max_size=n_deltas))
    chunks = [{"contentBlockDelta": {"delta": {"text": t}}} for t in texts]
    chunks.append({"contentBlockStop": {}})
    usage = draw(bedrock_stream_usage())
    chunks.append({"metadata": {"usage": usage}})
    return chunks, texts, usage


# ===================================================================
# BedrockConverseStream: streaming metadata parsing
# ===================================================================


class TestBedrockConverseStreamProperties:
    """The bug fix: `if value:` drops 0, but `if value is not None:` does not.

    These tests exercise BedrockConverseStream.process_raw_response
    metadata parsing logic with generated usage dicts.
    """

    @given(data=bedrock_stream_chunks())
    @settings(deadline=None)
    def test_all_token_counts_preserved_including_zero(self, data):
        """Token counts of 0 must be stored as 0, not dropped to None."""
        from llmeter.endpoints.bedrock import BedrockConverseStream

        chunks, _texts, usage = data
        raw_response = {
            "stream": chunks,
            "ResponseMetadata": {"RequestId": "test-id", "RetryAttempts": 0},
        }

        response = InvocationResponse(id=None, response_text=None)
        endpoint = BedrockConverseStream(model_id="test-model")
        endpoint.process_raw_response(raw_response, time.perf_counter(), response)

        assert response.num_tokens_input == usage["inputTokens"]
        assert response.num_tokens_output == usage["outputTokens"]
        assert response.num_tokens_input_cached == usage["cacheReadInputTokens"]

    @given(data=bedrock_stream_chunks())
    @settings(deadline=None)
    def test_response_text_is_concatenation_of_deltas(self, data):
        """response_text must be the exact concatenation of all delta texts."""
        from llmeter.endpoints.bedrock import BedrockConverseStream

        chunks, texts, _usage = data
        raw_response = {
            "stream": chunks,
            "ResponseMetadata": {"RequestId": "test-id", "RetryAttempts": 0},
        }

        response = InvocationResponse(id=None, response_text=None)
        endpoint = BedrockConverseStream(model_id="test-model")
        endpoint.process_raw_response(raw_response, time.perf_counter(), response)

        assert response.response_text == "".join(texts)

    @given(data=bedrock_stream_chunks())
    @settings(deadline=None)
    def test_timing_set_for_non_empty_streams(self, data):
        """Streams with content must have TTFT and TTLT set, with TTLT >= TTFT."""
        from llmeter.endpoints.bedrock import BedrockConverseStream

        chunks, _texts, _usage = data
        raw_response = {
            "stream": chunks,
            "ResponseMetadata": {"RequestId": "test-id", "RetryAttempts": 0},
        }

        response = InvocationResponse(id=None, response_text=None)
        endpoint = BedrockConverseStream(model_id="test-model")
        start_t = time.perf_counter()
        endpoint.process_raw_response(raw_response, start_t, response)

        assert response.time_to_first_token is not None
        assert response.time_to_first_token >= 0
        assert response.time_to_last_token is not None
        assert response.time_to_last_token >= response.time_to_first_token

    @given(
        input_tokens=non_negative_ints,
        output_tokens=non_negative_ints,
        cached_tokens=non_negative_ints,
    )
    @settings(deadline=None)
    def test_zero_token_counts_not_confused_with_none(
        self, input_tokens, output_tokens, cached_tokens
    ):
        """Directly test that 0-valued token counts are stored as 0, not None.

        This is the specific regression the is-not-None fix addresses.
        """
        from llmeter.endpoints.bedrock import BedrockConverseStream

        raw_response = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "hi"}}},
                {"contentBlockStop": {}},
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": input_tokens,
                            "outputTokens": output_tokens,
                            "cacheReadInputTokens": cached_tokens,
                        }
                    }
                },
            ],
            "ResponseMetadata": {"RequestId": "r", "RetryAttempts": 0},
        }

        response = InvocationResponse(id=None, response_text=None)
        endpoint = BedrockConverseStream(model_id="test-model")
        endpoint.process_raw_response(raw_response, time.perf_counter(), response)

        # The critical assertion: 0 is not None
        if input_tokens == 0:
            assert response.num_tokens_input == 0
        if output_tokens == 0:
            assert response.num_tokens_output == 0
        if cached_tokens == 0:
            assert response.num_tokens_input_cached == 0


# ===================================================================
# BedrockConverse (non-streaming) process_raw_response
# ===================================================================


class TestBedrockConverseProperties:
    """Property tests for the non-streaming BedrockConverse endpoint."""

    @given(
        text_parts=st.lists(_safe_text, min_size=1, max_size=5),
        input_tokens=non_negative_ints,
        output_tokens=non_negative_ints,
        cached_tokens=optional_non_negative_ints,
        request_id=st.text(min_size=1, max_size=40),
        retries=st.integers(min_value=0, max_value=5),
    )
    @settings(deadline=None)
    def test_non_streaming_parses_all_fields(
        self,
        text_parts,
        input_tokens,
        output_tokens,
        cached_tokens,
        request_id,
        retries,
    ):
        """BedrockConverse.process_raw_response should correctly extract all
        fields from a well-formed Converse API response."""
        from llmeter.endpoints.bedrock import BedrockConverse

        content = [{"text": t} for t in text_parts]
        usage = {"inputTokens": input_tokens, "outputTokens": output_tokens}
        if cached_tokens is not None:
            usage["cacheReadInputTokens"] = cached_tokens

        raw_response = {
            "output": {"message": {"content": content}},
            "usage": usage,
            "ResponseMetadata": {"RequestId": request_id, "RetryAttempts": retries},
        }

        response = InvocationResponse(id=None, response_text=None)
        endpoint = BedrockConverse(model_id="test-model")
        endpoint.process_raw_response(raw_response, time.perf_counter(), response)

        assert response.response_text == "".join(text_parts)
        assert response.id == request_id
        assert response.retries == retries
        assert response.num_tokens_input == input_tokens
        assert response.num_tokens_output == output_tokens
        if cached_tokens is not None:
            assert response.num_tokens_input_cached == cached_tokens
