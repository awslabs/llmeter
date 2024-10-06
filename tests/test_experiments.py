# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llmeter.experiments import LoadTest, LatencyHeatmap
from llmeter.runner import Runner


@pytest.fixture
def mock_endpoint():
    return MagicMock()


@pytest.fixture
def mock_runner():
    return AsyncMock(spec=Runner)


def test_load_test_post_init(mock_endpoint):
    with patch("llmeter.experiments.Runner") as mock_runner_class:
        load_test = LoadTest(
            endpoint=mock_endpoint,
            payload={"input": "test"},
            sequence_of_clients=[1, 2, 3],
            n_requests=10,
        )
        mock_runner_class.assert_called_once_with(
            endpoint=mock_endpoint, tokenizer=None
        )


# mock the run method of the Runner class
@patch("llmeter.experiments.Runner.run", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_load_test_run(mock_endpoint, mock_runner):
    with patch("llmeter.experiments.Runner", return_value=mock_runner):
        load_test = LoadTest(
            endpoint=mock_endpoint,
            payload={"input": "test"},
            sequence_of_clients=[1, 2, 3],
            n_requests=10,
        )

        results = await load_test.run()

        assert len(results) == 3
        mock_runner.run.assert_called()
        assert mock_runner.run.call_count == 3


def test_load_test_with_different_params(mock_endpoint):
    load_test = LoadTest(
        endpoint=mock_endpoint,
        payload=[{"input": "test1"}, {"input": "test2"}],
        sequence_of_clients=[1, 5, 10],
        n_requests=50,
        output_path="test_output.json",
        tokenizer=MagicMock(),
    )
    assert load_test.n_requests == 50
    assert load_test.output_path == "test_output.json"
    assert load_test.tokenizer is not None


def test_latency_heatmap_post_init(mock_endpoint):
    with patch("llmeter.experiments.CreatePromptCollection") as mock_create_prompt:
        with patch("llmeter.experiments.Runner"):
            mock_create_prompt.return_value.create_collection.return_value = [
                ("input1", 128),
                ("input2", 256),
            ]
            heatmap = LatencyHeatmap(
                endpoint=mock_endpoint,
                source_file="test_file.txt",
                clients=2,
                requests_per_combination=1,
            )
            assert len(heatmap.payload) == 2
            mock_create_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_latency_heatmap_run(mock_endpoint, mock_runner):
    with patch("llmeter.experiments.Runner", return_value=mock_runner):
        with patch("llmeter.experiments.CreatePromptCollection") as mock_create_prompt:
            mock_create_prompt.return_value.create_collection.return_value = [
                ("input1", 128),
                ("input2", 256),
            ]

            heatmap = LatencyHeatmap(
                endpoint=mock_endpoint,
                source_file="test_file.txt",
                clients=2,
                requests_per_combination=1,
                input_lengths=[10, 50],
                output_lengths=[128, 256],
            )

            results = await heatmap.run()

            mock_runner.run.assert_called_once()
            assert results == mock_runner.run.return_value


def test_latency_heatmap_with_different_params(mock_endpoint, tmp_path):
    (tmp_path / "test_file.txt").write_text("test")

    def create_payload_fn(input_text, max_tokens, **kwargs):
        return {"input": input_text, "max_tokens": max_tokens, **kwargs}

    heatmap = LatencyHeatmap(
        endpoint=mock_endpoint,
        source_file=tmp_path / "test_file.txt",
        clients=4,
        requests_per_combination=2,
        input_lengths=[100, 200, 300],
        output_lengths=[512, 1024],
        create_payload_fn=create_payload_fn,
        create_payload_kwargs={"temperature": 0.7},
        tokenizer=MagicMock(),
    )
    assert heatmap.clients == 4
    assert heatmap.requests_per_combination == 2
    assert heatmap.input_lengths == [100, 200, 300]
    assert heatmap.output_lengths == [512, 1024]
    assert heatmap.create_payload_kwargs == {"temperature": 0.7}
    assert heatmap.tokenizer is not None


def test_load_test_plot_sweep_results(mock_endpoint):
    load_test = LoadTest(
        endpoint=mock_endpoint,
        payload={"input": "test"},
        sequence_of_clients=[1, 2, 3],
        n_requests=10,
    )

    with pytest.raises(AttributeError):
        load_test.plot_sweep_results()


@pytest.mark.asyncio
async def test_latency_heatmap_plot_heatmap(mock_endpoint, tmp_path):
    (tmp_path / "test_file.txt").write_text("test")

    heatmap = LatencyHeatmap(
        endpoint=mock_endpoint,
        source_file=tmp_path / "test_file.txt",
        clients=2,
        requests_per_combination=1,
    )

    with pytest.raises(AttributeError):
        heatmap.plot_heatmap()
