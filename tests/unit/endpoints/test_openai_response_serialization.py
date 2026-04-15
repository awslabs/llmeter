# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for OpenAIResponseEndpoint and OpenAIResponseStreamEndpoint serialization.

This module tests endpoint serialization/deserialization including to_dict,
save, load_from_file, and round-trip operations.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from llmeter.endpoints.base import Endpoint
from llmeter.endpoints.openai_response import OpenAIResponseEndpoint, OpenAIResponseStreamEndpoint


class TestEndpointSerialization:
    """Test endpoint serialization methods.

    **Validates: Requirements 10.4, 10.5**
    """

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_to_dict_produces_valid_dictionary(self, mock_openai_class):
        """Test to_dict produces valid dictionary."""
        endpoint = OpenAIResponseEndpoint(
            model_id="gpt-4",
            endpoint_name="test-response",
            provider="openai",
        )

        result = endpoint.to_dict()

        # Verify result is a dictionary
        assert isinstance(result, dict)

        # Verify required fields are present
        assert "model_id" in result
        assert "endpoint_name" in result
        assert "provider" in result
        assert "endpoint_type" in result

        # Verify values are correct
        assert result["model_id"] == "gpt-4"
        assert result["endpoint_name"] == "test-response"
        assert result["provider"] == "openai"
        assert result["endpoint_type"] == "OpenAIResponseEndpoint"

        # Verify private attributes are excluded
        assert not any(key.startswith("_") for key in result.keys())

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_save_creates_file_with_correct_content(self, mock_openai_class):
        """Test save creates file with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            endpoint = OpenAIResponseEndpoint(
                model_id="gpt-4-turbo",
                endpoint_name="saved-endpoint",
                provider="openai",
            )

            output_path = Path(tmpdir) / "endpoint.json"
            saved_path = endpoint.save(output_path)

            # Verify file was created
            assert saved_path.exists()
            assert saved_path == output_path

            # Verify file contains valid JSON
            with open(saved_path, "r") as f:
                data = json.load(f)

            # Verify content matches endpoint configuration
            assert data["model_id"] == "gpt-4-turbo"
            assert data["endpoint_name"] == "saved-endpoint"
            assert data["provider"] == "openai"
            assert data["endpoint_type"] == "OpenAIResponseEndpoint"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_load_from_file_reconstructs_endpoint(self, mock_openai_class):
        """Test load_from_file reconstructs endpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save endpoint
            original = OpenAIResponseEndpoint(
                model_id="gpt-4",
                endpoint_name="original-endpoint",
                provider="openai",
            )

            output_path = Path(tmpdir) / "endpoint.json"
            original.save(output_path)

            # Load endpoint
            loaded = Endpoint.load_from_file(output_path)

            # Verify loaded endpoint has correct type
            assert isinstance(loaded, OpenAIResponseEndpoint)

            # Verify loaded endpoint has correct attributes
            assert loaded.model_id == "gpt-4"
            assert loaded.endpoint_name == "original-endpoint"
            assert loaded.provider == "openai"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_round_trip_save_then_load(self, mock_openai_class):
        """Test round trip (save then load)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original endpoint
            original = OpenAIResponseEndpoint(
                model_id="gpt-4-turbo",
                endpoint_name="roundtrip-test",
                provider="openai",
            )

            # Save
            output_path = Path(tmpdir) / "roundtrip.json"
            original.save(output_path)

            # Load
            loaded = Endpoint.load_from_file(output_path)

            # Verify loaded endpoint matches original
            assert isinstance(loaded, type(original))
            assert loaded.model_id == original.model_id
            assert loaded.endpoint_name == original.endpoint_name
            assert loaded.provider == original.provider

            # Verify to_dict produces same output (excluding private attributes)
            original_dict = original.to_dict()
            loaded_dict = loaded.to_dict()

            assert original_dict == loaded_dict


class TestStreamEndpointSerialization:
    """Test streaming endpoint serialization methods.

    **Validates: Requirements 10.4, 10.5**
    """

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_stream_endpoint_to_dict(self, mock_openai_class):
        """Test streaming endpoint to_dict produces valid dictionary."""
        endpoint = OpenAIResponseStreamEndpoint(
            model_id="gpt-4",
            endpoint_name="test-stream",
            provider="openai",
        )

        result = endpoint.to_dict()

        # Verify result is a dictionary
        assert isinstance(result, dict)

        # Verify required fields are present
        assert "model_id" in result
        assert "endpoint_name" in result
        assert "provider" in result
        assert "endpoint_type" in result

        # Verify values are correct
        assert result["model_id"] == "gpt-4"
        assert result["endpoint_name"] == "test-stream"
        assert result["provider"] == "openai"
        assert result["endpoint_type"] == "OpenAIResponseStreamEndpoint"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_stream_endpoint_save_and_load(self, mock_openai_class):
        """Test streaming endpoint save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save endpoint
            original = OpenAIResponseStreamEndpoint(
                model_id="gpt-4-turbo",
                endpoint_name="stream-endpoint",
                provider="openai",
            )

            output_path = Path(tmpdir) / "stream_endpoint.json"
            original.save(output_path)

            # Load endpoint
            loaded = Endpoint.load_from_file(output_path)

            # Verify loaded endpoint has correct type
            assert isinstance(loaded, OpenAIResponseStreamEndpoint)

            # Verify loaded endpoint has correct attributes
            assert loaded.model_id == "gpt-4-turbo"
            assert loaded.endpoint_name == "stream-endpoint"
            assert loaded.provider == "openai"

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_stream_endpoint_round_trip(self, mock_openai_class):
        """Test streaming endpoint round trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original endpoint
            original = OpenAIResponseStreamEndpoint(
                model_id="gpt-4",
                endpoint_name="stream-roundtrip",
                provider="openai",
            )

            # Save
            output_path = Path(tmpdir) / "stream_roundtrip.json"
            original.save(output_path)

            # Load
            loaded = Endpoint.load_from_file(output_path)

            # Verify loaded endpoint matches original
            assert isinstance(loaded, type(original))
            assert loaded.model_id == original.model_id
            assert loaded.endpoint_name == original.endpoint_name
            assert loaded.provider == original.provider


class TestSerializationPropertyTests:
    """Property-based tests for endpoint serialization.

    **Property 10: Endpoint Serialization Round Trip**
    **Validates: Requirements 10.4, 10.5**
    """

    @given(
        model_id=st.text(min_size=1, max_size=50),
        endpoint_name=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                min_codepoint=32,
                max_codepoint=126,
                blacklist_characters='\\/:*?"<>|',
            ),
        ),
        provider=st.text(min_size=1, max_size=30),
    )
    @settings(deadline=None, max_examples=100)
    @patch("llmeter.endpoints.openai.OpenAI")
    def test_property_response_endpoint_serialization_round_trip(
        self, mock_openai_class, model_id, endpoint_name, provider
    ):
        """
        Property 10: Endpoint Serialization Round Trip

        For any ResponseEndpoint instance, serializing via to_dict and then
        loading via Endpoint.load should produce an equivalent endpoint with
        the same configuration.

        **Validates: Requirements 10.4, 10.5**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original endpoint
            original = OpenAIResponseEndpoint(
                model_id=model_id,
                endpoint_name=endpoint_name,
                provider=provider,
            )

            # Save
            output_path = Path(tmpdir) / "endpoint.json"
            original.save(output_path)

            # Load
            loaded = Endpoint.load_from_file(output_path)

            # Verify loaded endpoint matches original
            assert isinstance(loaded, type(original))
            assert loaded.model_id == original.model_id
            assert loaded.endpoint_name == original.endpoint_name
            assert loaded.provider == original.provider

            # Verify to_dict produces same output
            original_dict = original.to_dict()
            loaded_dict = loaded.to_dict()
            assert original_dict == loaded_dict

    @given(
        model_id=st.text(min_size=1, max_size=50),
        endpoint_name=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                min_codepoint=32,
                max_codepoint=126,
                blacklist_characters='\\/:*?"<>|',
            ),
        ),
        provider=st.text(min_size=1, max_size=30),
    )
    @settings(deadline=None, max_examples=100)
    @patch("llmeter.endpoints.openai.OpenAI")
    def test_property_stream_endpoint_serialization_round_trip(
        self, mock_openai_class, model_id, endpoint_name, provider
    ):
        """
        Property 10: Endpoint Serialization Round Trip (Streaming)

        For any ResponseStreamEndpoint instance, serializing via to_dict and
        then loading via Endpoint.load should produce an equivalent endpoint
        with the same configuration.

        **Validates: Requirements 10.4, 10.5**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original endpoint
            original = OpenAIResponseStreamEndpoint(
                model_id=model_id,
                endpoint_name=endpoint_name,
                provider=provider,
            )

            # Save
            output_path = Path(tmpdir) / "stream_endpoint.json"
            original.save(output_path)

            # Load
            loaded = Endpoint.load_from_file(output_path)

            # Verify loaded endpoint matches original
            assert isinstance(loaded, type(original))
            assert loaded.model_id == original.model_id
            assert loaded.endpoint_name == original.endpoint_name
            assert loaded.provider == original.provider

            # Verify to_dict produces same output
            original_dict = original.to_dict()
            loaded_dict = loaded.to_dict()
            assert original_dict == loaded_dict


class TestInvocationResponseTypeConsistency:
    """Test InvocationResponse type consistency.

    **Property 11: InvocationResponse Type Consistency**
    **Validates: Requirements 10.1**
    """

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_response_endpoint_returns_invocation_response(self, mock_openai_class):
        """
        Verify invoke always returns InvocationResponse for ResponseEndpoint.

        **Validates: Requirements 10.1**
        """
        from llmeter.endpoints.base import InvocationResponse

        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_response.output_text = "Test response"
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_client.responses.create.return_value = mock_response

        # Create endpoint and invoke
        endpoint = OpenAIResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify return type
        assert isinstance(response, InvocationResponse)

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_stream_endpoint_returns_invocation_response(self, mock_openai_class):
        """
        Verify invoke always returns InvocationResponse for ResponseStreamEndpoint.

        **Validates: Requirements 10.1**
        """
        from llmeter.endpoints.base import InvocationResponse

        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response events
        event_created = Mock()
        event_created.type = "response.created"
        event_created.response = Mock()
        event_created.response.id = "resp_stream"

        event_delta1 = Mock()
        event_delta1.type = "response.output_text.delta"
        event_delta1.delta = "Test "

        event_delta2 = Mock()
        event_delta2.type = "response.output_text.delta"
        event_delta2.delta = "response"

        event_completed = Mock()
        event_completed.type = "response.completed"
        event_completed.response = Mock()
        event_completed.response.usage = Mock(spec=["input_tokens", "output_tokens"])
        event_completed.response.usage.input_tokens = 10
        event_completed.response.usage.output_tokens = 5

        mock_client.responses.create.return_value = iter(
            [event_created, event_delta1, event_delta2, event_completed]
        )

        # Create endpoint and invoke
        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify return type
        assert isinstance(response, InvocationResponse)

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_response_endpoint_error_returns_invocation_response(
        self, mock_openai_class
    ):
        """
        Verify invoke returns InvocationResponse even on error for ResponseEndpoint.

        **Validates: Requirements 10.1**
        """
        from openai import APIConnectionError
        from llmeter.endpoints.base import InvocationResponse

        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_request = Mock()
        mock_client.responses.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock_request
        )

        # Create endpoint and invoke
        endpoint = OpenAIResponseEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify return type even on error
        assert isinstance(response, InvocationResponse)
        assert response.error is not None

    @patch("llmeter.endpoints.openai.OpenAI")
    def test_stream_endpoint_error_returns_invocation_response(self, mock_openai_class):
        """
        Verify invoke returns InvocationResponse even on error for ResponseStreamEndpoint.

        **Validates: Requirements 10.1**
        """
        from openai import APIConnectionError
        from llmeter.endpoints.base import InvocationResponse

        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_request = Mock()
        mock_client.responses.create.side_effect = APIConnectionError(
            message="Connection failed", request=mock_request
        )

        # Create endpoint and invoke
        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4")
        payload = {"input": "Test", "max_tokens": 256}
        response = endpoint.invoke(payload)

        # Verify return type even on error
        assert isinstance(response, InvocationResponse)
        assert response.error is not None

    @given(
        payload=st.fixed_dictionaries(
            {
                "input": st.text(min_size=1, max_size=100),
                "max_tokens": st.integers(min_value=1, max_value=4096),
            }
        )
    )
    @settings(deadline=None, max_examples=100)
    @patch("llmeter.endpoints.openai.OpenAI")
    def test_property_response_endpoint_always_returns_invocation_response(
        self, mock_openai_class, payload
    ):
        """
        Property 11: InvocationResponse Type Consistency

        For any invocation of ResponseEndpoint, the return value should be
        an InvocationResponse object.

        **Validates: Requirements 10.1**
        """
        from llmeter.endpoints.base import InvocationResponse

        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.id = "resp_prop"
        mock_response.output_text = "Property test response"
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_client.responses.create.return_value = mock_response

        # Create endpoint and invoke
        endpoint = OpenAIResponseEndpoint(model_id="gpt-4")
        response = endpoint.invoke(payload)

        # Verify return type
        assert isinstance(response, InvocationResponse)

    @given(
        payload=st.fixed_dictionaries(
            {
                "input": st.text(min_size=1, max_size=100),
                "max_tokens": st.integers(min_value=1, max_value=4096),
            }
        )
    )
    @settings(deadline=None, max_examples=100)
    @patch("llmeter.endpoints.openai.OpenAI")
    def test_property_stream_endpoint_always_returns_invocation_response(
        self, mock_openai_class, payload
    ):
        """
        Property 11: InvocationResponse Type Consistency (Streaming)

        For any invocation of ResponseStreamEndpoint, the return value should
        be an InvocationResponse object.

        **Validates: Requirements 10.1**
        """
        from llmeter.endpoints.base import InvocationResponse

        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock streaming response events
        event_created = Mock()
        event_created.type = "response.created"
        event_created.response = Mock()
        event_created.response.id = "resp_prop_stream"

        event_delta1 = Mock()
        event_delta1.type = "response.output_text.delta"
        event_delta1.delta = "Property "

        event_delta2 = Mock()
        event_delta2.type = "response.output_text.delta"
        event_delta2.delta = "test"

        event_completed = Mock()
        event_completed.type = "response.completed"
        event_completed.response = Mock()
        event_completed.response.usage = Mock(spec=["input_tokens", "output_tokens"])
        event_completed.response.usage.input_tokens = 10
        event_completed.response.usage.output_tokens = 5

        mock_client.responses.create.return_value = iter(
            [event_created, event_delta1, event_delta2, event_completed]
        )

        # Create endpoint and invoke
        endpoint = OpenAIResponseStreamEndpoint(model_id="gpt-4")
        response = endpoint.invoke(payload)

        # Verify return type
        assert isinstance(response, InvocationResponse)
