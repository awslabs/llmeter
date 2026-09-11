# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `delta_has_reasoning_content`.

The helper is public API (custom endpoints can use it), so it must cope with both shapes a
streaming delta arrives in: a mapping, when the caller parsed raw SSE JSON itself, and an object,
when it came from a provider SDK.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llmeter.endpoints.base import delta_has_reasoning_content


# ---------------------------------------------------------------------------
# Mapping deltas (custom endpoints parsing raw SSE JSON)
# ---------------------------------------------------------------------------


class TestMappingDeltas:
    @pytest.mark.parametrize(
        "delta",
        [
            {"reasoning_content": "thinking..."},
            {"reasoning": "thinking..."},
            {"thinking_blocks": [{"type": "thinking", "thinking": "hmm"}]},
            {"content": None, "reasoning_content": "thinking..."},
        ],
    )
    def test_reasoning_detected(self, delta):
        assert delta_has_reasoning_content(delta) is True

    @pytest.mark.parametrize(
        "delta",
        [
            {},
            {"content": "Hello"},
            {"content": None},
            {"reasoning_content": ""},  # empty is not reasoning output
            {"reasoning_content": None},
            {"thinking_blocks": []},
            {"role": "assistant"},
        ],
    )
    def test_no_reasoning(self, delta):
        assert delta_has_reasoning_content(delta) is False

    def test_mapping_subclass_supported(self):
        """Any Mapping, not just dict."""
        from collections import OrderedDict

        assert delta_has_reasoning_content(OrderedDict(reasoning_content="x")) is True


# ---------------------------------------------------------------------------
# Object deltas
# ---------------------------------------------------------------------------


class TestObjectDeltas:
    def test_plain_object(self):
        assert (
            delta_has_reasoning_content(SimpleNamespace(reasoning_content="x")) is True
        )

    def test_plain_object_without_reasoning(self):
        assert delta_has_reasoning_content(SimpleNamespace(content="Hello")) is False

    def test_missing_attributes_do_not_raise(self):
        assert delta_has_reasoning_content(SimpleNamespace()) is False

    def test_magicmock_not_mistaken_for_reasoning(self):
        """MagicMock auto-creates attributes, so truthiness alone would give a false positive.

        This matters because much of the existing test suite builds deltas from MagicMock.
        """
        assert delta_has_reasoning_content(MagicMock()) is False

    def test_non_string_value_rejected(self):
        assert (
            delta_has_reasoning_content(SimpleNamespace(reasoning_content=1)) is False
        )

    def test_thinking_blocks_must_be_a_sequence(self):
        """A stray string in `thinking_blocks` is not a block list."""
        assert (
            delta_has_reasoning_content(SimpleNamespace(thinking_blocks="x")) is False
        )


# ---------------------------------------------------------------------------
# Real SDK delta types
# ---------------------------------------------------------------------------


class TestOpenAISdkDeltas:
    """Guards against SDK drift in how unmodelled provider fields are surfaced."""

    def test_reasoning_content_as_unmodelled_extra(self):
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        delta = ChoiceDelta.model_validate(
            {"content": None, "reasoning_content": "thinking..."}
        )
        assert delta_has_reasoning_content(delta) is True

    def test_reasoning_alias_as_unmodelled_extra(self):
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        delta = ChoiceDelta.model_validate(
            {"content": None, "reasoning": "thinking..."}
        )
        assert delta_has_reasoning_content(delta) is True

    def test_ordinary_content_delta(self):
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        delta = ChoiceDelta.model_validate({"content": "Hello"})
        assert delta_has_reasoning_content(delta) is False


class TestLiteLLMSdkDeltas:
    def test_declared_reasoning_content_field(self):
        from litellm.types.utils import Delta

        assert delta_has_reasoning_content(Delta(reasoning_content="thinking")) is True

    def test_attribute_absent_when_unset(self):
        """LiteLLM deletes `reasoning_content` when it is None, rather than leaving it set."""
        from litellm.types.utils import Delta

        assert delta_has_reasoning_content(Delta(content="Hello")) is False

    def test_thinking_blocks(self):
        from litellm.types.utils import Delta

        delta = Delta(
            content=None, thinking_blocks=[{"type": "thinking", "thinking": "hmm"}]
        )
        assert delta_has_reasoning_content(delta) is True
