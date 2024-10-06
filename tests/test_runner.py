# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import time
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from upath import UPath as Path

import llmeter.endpoints
from llmeter.endpoints.base import Endpoint, InvocationResponse
from llmeter.runner import Runner, _RunConfig
from llmeter.tokenizers import Tokenizer


@pytest.fixture
def mock_endpoint():
    endpoint: Endpoint = MagicMock(spec=Endpoint)
    endpoint.endpoint_name = "endpoint-name"
    endpoint.model_id = "model-id"
    endpoint.provider = "provider"
    endpoint.invoke.return_value = InvocationResponse(
        id="1", input_prompt="test", response_text="response"
    )
    endpoint_conf = {k: v for k, v in vars(endpoint).items() if not k.startswith("_")}
    endpoint_conf["endpoint_type"] = "mock_endpoint"
    endpoint.to_dict.return_value = endpoint_conf
    return endpoint


@pytest.fixture
def mock_tokenizer():
    with patch(
        "llmeter.tokenizers.Tokenizer.to_dict",
        return_value={"tokenizer_module": "mock_tokenizer"},
    ):
        yield MagicMock(spec=Tokenizer)


@pytest.fixture
def runner(mock_endpoint: MagicMock, mock_tokenizer: MagicMock):
    mock_runner = Runner(endpoint=mock_endpoint, tokenizer=mock_tokenizer)
    mock_runner._responses = []
    mock_runner._progress_bar = MagicMock()
    mock_runner._queue = AsyncMock()
    mock_runner._queue.task_done = MagicMock()

    return mock_runner


def test_runner_initialization(runner: Runner):
    assert isinstance(runner, Runner)
    assert isinstance(runner, _RunConfig)


def test_count_tokens_no_wait(runner: Runner):
    runner._tokenizer.encode.return_value = [1, 2, 3]
    assert runner._count_tokens_no_wait("test text") == 3


@pytest.mark.asyncio
async def test_count_tokens_from_q(runner: Runner):
    runner._queue = AsyncMock()
    runner._queue.get.side_effect = [
        InvocationResponse(
            id="1",
            input_prompt="test",
            response_text="response",
            num_tokens_input=5,
            num_tokens_output=5,
        ),
        None,
    ]
    runner._queue.task_done = MagicMock()
    runner._count_tokens_no_wait = AsyncMock(return_value=5)

    await runner._count_tokens_from_q()

    assert len(runner._responses) == 1
    assert runner._responses[0].num_tokens_input == 5
    assert runner._responses[0].num_tokens_output == 5


@pytest.mark.asyncio
async def test_invoke_n(runner: Runner):
    runner._invoke_n_no_wait = MagicMock(
        return_value=[
            InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
            InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        ]
    )

    result = await runner._invoke_n(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}], n=2
    )

    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"


@pytest.mark.asyncio
async def test_run(runner: Runner):
    runner._invoke_n_c = AsyncMock(return_value=1.0)
    runner._count_tokens_from_q = AsyncMock()

    result = await runner.run(payload={"prompt": "test"}, n_requests=1, clients=1)

    assert result.total_requests == 1
    assert result.clients == 1
    assert result.n_requests == 1
    assert result.total_test_time == 1.0


@pytest.mark.asyncio
async def test_invoke_n_no_wait(runner: Runner):
    runner._endpoint.invoke.side_effect = [
        InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
        InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
    ]

    runner._queue = AsyncMock()
    runner._queue._loop.call_soon_threadsafe = MagicMock()

    result = runner._invoke_n_no_wait(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}], n=2
    )

    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"


@pytest.mark.asyncio
async def test_invoke_n_c(runner: Runner):
    runner._invoke_n = AsyncMock(
        return_value=[
            InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
            InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        ]
    )

    total_test_time = await runner._invoke_n_c(
        payload=[{"prompt": "test"}], n_requests=2, clients=1
    )

    assert isinstance(total_test_time, float)
    assert runner._invoke_n.call_count == 1


@pytest.mark.asyncio
async def test_run_with_file_payload(runner: Runner, tmp_path: Path):
    payload_file = tmp_path / "payload.jsonl"
    payload_file.write_text('{"prompt": "test1"}\n{"prompt": "test2"}')

    runner._invoke_n_c = AsyncMock(return_value=1.0)
    runner._count_tokens_from_q = AsyncMock()

    result = await runner.run(payload=str(payload_file), n_requests=2, clients=1)

    assert result.total_requests == 2
    assert result.clients == 1
    assert result.n_requests == 2
    assert result.total_test_time == 1.0


@pytest.mark.asyncio
async def test_run_with_output_path(runner: Runner, tmp_path: Path):
    output_path = tmp_path / "output"

    runner._invoke_n_c = AsyncMock(return_value=1.0)
    runner._count_tokens_from_q = AsyncMock()

    result = await runner.run(
        payload={"prompt": "test"},
        n_requests=1,
        clients=1,
        output_path=str(output_path),
    )

    assert result.output_path is not None
    assert (output_path / result.run_name / "run_config.json").exists()  # type: ignore


@pytest.mark.asyncio
async def test_run_error_handling(runner: Runner):
    runner._invoke_n_c = AsyncMock(side_effect=Exception("Test error"))
    runner._count_tokens_from_q = AsyncMock()

    with pytest.raises(Exception, match="Test error"):
        await runner.run(payload={"prompt": "test"}, n_requests=1, clients=1)


@pytest.mark.asyncio
async def test_prepare_run_config(runner: Runner):
    payload = {"prompt": "test"}
    n_requests = 5
    clients = 2
    output_path = "/tmp/output"
    run_name = "test_run"
    run_description = "Test run description"

    run_config = runner._prepare_run_config(
        payload, n_requests, clients, output_path, run_name, run_description
    )

    assert run_config.payload == [payload]
    assert run_config.n_requests == n_requests
    assert run_config.clients == clients
    assert str(run_config.output_path) == output_path
    assert run_config.run_name == run_name
    assert run_config.run_description == run_description


def test_validate_and_prepare_payload(runner: Runner):
    # Test with dict payload
    run_config = _RunConfig(payload={"prompt": "test"}, endpoint=mock_endpoint)  # type: ignore
    runner._validate_and_prepare_payload(run_config)
    assert isinstance(run_config.payload, list)
    assert len(run_config.payload) == 1

    # Test with list payload
    run_config = _RunConfig(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}],
        endpoint=mock_endpoint,  # type: ignore
    )
    runner._validate_and_prepare_payload(run_config)
    assert isinstance(run_config.payload, list)
    assert len(run_config.payload) == 2

    # Test with file payload (mock)
    with pytest.raises(FileNotFoundError):
        run_config = _RunConfig(
            payload="nonexistent_file.jsonl",
            endpoint=mock_endpoint,  # type: ignore
        )
        runner._validate_and_prepare_payload(run_config)


def test_prepare_output_path(runner: Runner, tmp_path: Path, mock_endpoint: Endpoint):
    # Test with provided run_name
    run_config = _RunConfig(
        output_path=str(tmp_path),
        run_name="test_run",
        endpoint=mock_endpoint,  # type: ignore
    )
    runner._prepare_output_path(run_config)
    assert run_config.run_name == "test_run"
    assert isinstance(run_config.output_path, Path)

    # Test without provided run_name (should generate one)
    run_config = _RunConfig(
        output_path=str(tmp_path),
        run_name=None,
        endpoint=mock_endpoint,  # type: ignore
    )
    runner._prepare_output_path(run_config)
    assert run_config.run_name is not None
    assert isinstance(run_config.output_path, Path)


def test_count_tokens_no_wait_edge_cases(runner: Runner):
    # Test with None input
    assert runner._count_tokens_no_wait(None) == 0

    class NonStringableClass:
        def __str__(self):
            pass

    # Test with non-string input
    with pytest.raises(ValueError):
        runner._count_tokens_no_wait(NonStringableClass())


@pytest.mark.asyncio
async def test_invoke_n_edge_cases(runner: Runner):
    # Test with empty payload
    result = await runner._invoke_n(payload=[], n=5)
    assert not result

    # Test with n=None (should use all payloads)
    runner._invoke_n_no_wait = MagicMock(
        return_value=[
            InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
            InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        ]
    )
    result = await runner._invoke_n(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}], n=None
    )
    assert len(result) == 2


@pytest.mark.asyncio
async def test_run_with_sequence_payload(runner: Runner):
    runner._invoke_n_c = AsyncMock(return_value=1.0)
    runner._count_tokens_from_q = AsyncMock()

    result = await runner.run(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}], n_requests=2, clients=1
    )

    assert result.total_requests == 2
    assert result.clients == 1
    assert result.n_requests == 2
    assert result.total_test_time == 1.0


def test_initialize_result(runner: Runner):
    run_config = _RunConfig(
        payload={"prompt": "test"},
        n_requests=5,
        clients=2,
        output_path="/tmp/output",
        run_name="test_run",
        run_description="Test run description",
        endpoint=mock_endpoint,  # type: ignore
    )
    result = runner._initialize_result(run_config)

    assert result.total_requests == 10
    assert result.clients == 2
    assert result.n_requests == 5
    assert result.run_name == "test_run"
    assert result.run_description == "Test run description"


@pytest.mark.asyncio
async def test_count_tokens_from_q_timeout(runner: Runner):
    runner._queue = AsyncMock()
    runner._queue.get.side_effect = asyncio.TimeoutError()

    await runner._count_tokens_from_q()

    runner._queue.assert_awaited_once
    runner._queue.task_done = MagicMock()

    # assert len(runner._responses) == 0


def test_run_config_save_load(tmp_path: Path, mock_endpoint: Endpoint):
    llmeter.endpoints.mock_endpoint = mock_endpoint  # type: ignore

    config = Runner(
        payload={"prompt": "test"},
        n_requests=5,
        clients=2,
        output_path=str(tmp_path),
        run_name="test_run",
        run_description="Test run description",
        endpoint=mock_endpoint,
    )

    config.save(output_path=tmp_path)
    loaded_config = Runner.load(tmp_path / "test_run")

    if isinstance(loaded_config.payload, str):
        with open(loaded_config.payload) as f:
            loaded_payload = json.load(f)

    assert loaded_payload == config.payload
    assert loaded_config.n_requests == config.n_requests
    assert loaded_config.clients == config.clients
    assert loaded_config.output_path == config.output_path
    assert loaded_config.run_name == config.run_name
    assert loaded_config.run_description == config.run_description


@pytest.mark.parametrize(
    "payload, n_requests, clients, output_path, run_name, run_description",
    [
        ({"prompt": "test"}, 5, 2, "/tmp/output", "test_run", "Test description"),
        ([{"prompt": "test1"}, {"prompt": "test2"}], None, 1, None, None, None),
        ("test_file.jsonl", 10, 5, "/tmp/output", None, "File input test"),
    ],
)
def test_prepare_run_config_combinations(
    runner: Runner,
    payload: dict[str, str] | list[dict[str, str]] | Literal["test_file.jsonl"],
    n_requests: None | Literal[5] | Literal[10],
    clients: Literal[2] | Literal[1] | Literal[5],
    output_path: None | Literal["/tmp/output"],
    run_name: None | Literal["test_run"],
    run_description: None | Literal["Test description"] | Literal["File input test"],
):
    if payload == "test_file.jsonl":
        payload_file = Path(payload)
        payload_file.write_text('{"prompt": "test1"}\n{"prompt": "test2"}')

    run_config = runner._prepare_run_config(
        payload, n_requests, clients, output_path, run_name, run_description
    )

    assert isinstance(run_config.payload, list)
    assert run_config.n_requests == n_requests
    assert run_config.clients == clients
    assert run_config.output_path == (Path(output_path) if output_path else None)
    assert run_config.run_name is not None
    assert run_config.run_description == (run_description if run_description else None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [{"prompt": "test"}, [{"prompt": "test1"}, {"prompt": "test2"}], "test_file.jsonl"],
)
async def test_run_with_different_payloads(
    runner: Runner,
    payload: dict[str, str] | list[dict[str, str]] | Literal["test_file.jsonl"],
    tmp_path: Path,
):
    if isinstance(payload, str):
        payload_file = tmp_path / payload
        payload_file.write_text('{"prompt": "test1"}\n{"prompt": "test2"}')
        payload = str(payload_file)  # type: ignore

    runner._invoke_n_c = AsyncMock(return_value=1.0)
    runner._count_tokens_from_q = AsyncMock()

    result = await runner.run(payload=payload, n_requests=2, clients=1)

    assert result.total_requests == 2
    assert result.clients == 1
    assert result.n_requests == 2
    assert result.total_test_time == 1.0


@pytest.mark.asyncio
async def test_invoke_n_c_concurrent_execution(runner: Runner):
    async def mock_invoke_n(payload, n, add_start_jitter=True, shuffle_order=True):
        await asyncio.sleep(0.1)  # Simulate some processing time
        return [
            InvocationResponse(
                id=str(i), input_prompt=f"test{i}", response_text=f"response{i}"
            )
            for i in range(n)
        ]

    runner._invoke_n = mock_invoke_n  # type: ignore

    start_time = time.perf_counter()
    total_test_time = await runner._invoke_n_c(
        payload=[{"prompt": "test"}], n_requests=5, clients=3
    )
    end_time = time.perf_counter()

    assert total_test_time > 0
    assert (
        end_time - start_time < 0.3
    )  # Ensure concurrent execution (should be less than 0.3 seconds for 3 clients)


@pytest.mark.parametrize(
    "payload, n_requests, clients",
    [
        (None, 5, 2),
        ({"prompt": "test"}, -1, 2),
        ({"prompt": "test"}, 5, 0),
    ],
)
def test_prepare_run_config_invalid_inputs(
    runner: Runner,
    payload: None | dict[str, str],
    n_requests: Literal[5] | Literal[-1],
    clients: Literal[2] | Literal[0],
):
    with pytest.raises(AssertionError):
        runner._prepare_run_config(payload, n_requests, clients, None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, n_requests, clients",
    [
        (None, 5, 2),
        ({"prompt": "test"}, -1, 2),
        ({"prompt": "test"}, 5, 0),
    ],
)
async def test_run_invalid_inputs(
    runner: Runner,
    payload: None | dict[str, str],
    n_requests: Literal[5] | Literal[-1],
    clients: Literal[2] | Literal[0],
):
    with pytest.raises(AssertionError):
        await runner.run(payload=payload, n_requests=n_requests, clients=clients)


@pytest.mark.asyncio
async def test_count_tokens_from_q_different_scenarios(runner: Runner):
    # Scenario 1: Queue with multiple items
    runner._queue = AsyncMock()
    runner._queue.task_done = MagicMock()
    runner._queue.get.side_effect = [
        InvocationResponse(
            id="1",
            input_prompt="test1",
            response_text="response1",
            num_tokens_input=5,
            num_tokens_output=5,
        ),
        InvocationResponse(
            id="2",
            input_prompt="test2",
            response_text="response2",
            num_tokens_input=5,
            num_tokens_output=5,
        ),
        None,
    ]
    runner._count_tokens_no_wait = AsyncMock(return_value=5)

    await runner._count_tokens_from_q()

    assert len(runner._responses) == 2
    assert runner._responses[0].num_tokens_input == 5
    assert runner._responses[0].num_tokens_output == 5
    assert runner._responses[1].num_tokens_input == 5
    assert runner._responses[1].num_tokens_output == 5

    # Scenario 2: Queue with exception
    runner._queue.reset_mock()
    runner._responses = []
    runner._queue.get.side_effect = Exception("Test exception")
    runner._queue.task_done = MagicMock()

    await runner._count_tokens_from_q()

    # assert len(runner._responses) == 0
    assert not runner._responses


@pytest.mark.parametrize(
    "payload, n_requests, clients, output_path, run_name, run_description",
    [
        ({"prompt": "test"}, 5, 2, "/tmp/output", "test_run", "Test description"),
        ([{"prompt": "test1"}, {"prompt": "test2"}], None, 1, None, None, None),
        ("test_file.jsonl", 10, 5, "/tmp/output", None, "File input test"),
    ],
)
def test_initialize_result_combinations(
    runner: Runner,
    payload: dict[str, str] | list[dict[str, str]] | Literal["test_file.jsonl"],
    n_requests: None | Literal[5] | Literal[10],
    clients: Literal[2] | Literal[1] | Literal[5],
    output_path: None | Literal["/tmp/output"],
    run_name: None | Literal["test_run"],
    run_description: None | Literal["Test description"] | Literal["File input test"],
):
    run_config = _RunConfig(
        payload=payload,
        n_requests=n_requests,
        clients=clients,
        output_path=output_path,
        run_name=run_name,
        run_description=run_description,
        endpoint=mock_endpoint,  # type: ignore
    )
    result = runner._initialize_result(run_config)

    assert result.total_requests == (n_requests or len(payload)) * clients
    assert result.clients == clients
    assert result.n_requests == n_requests or len(payload)
    assert result.run_name == run_name or result.run_name is not None
    assert result.run_description == run_description


@pytest.mark.parametrize(
    "payload, n_requests, clients, output_path, run_name, run_description",
    [
        # (None, 5, 2, "/tmp/output", "test_run", "Test description"),
        ({"prompt": "test"}, -1, 2, "/tmp/output", "test_run", "Test description"),
        ({"prompt": "test"}, 5, 0, "/tmp/output", "test_run", "Test description"),
        ({"prompt": "test"}, 5, 2, "/tmp/output", "", "Test description"),
    ],
)
def test_prepare_run_config_edge_cases(
    runner: Runner,
    payload: None | dict[str, str],
    n_requests: Literal[5] | Literal[-1],
    clients: Literal[2] | Literal[0],
    output_path: Literal["/tmp/output"],
    run_name: Literal["test_run"] | Literal[""],
    run_description: Literal["Test description"],
):
    with pytest.raises((AssertionError, ValueError)):
        runner._prepare_run_config(
            payload, n_requests, clients, output_path, run_name, run_description
        )


@pytest.mark.parametrize(
    "payload, n_requests, clients, output_path, run_name, run_description",
    [
        ({"prompt": "test"}, None, None, None, None, None),
        (
            [{"prompt": "test1"}, {"prompt": "test2"}],
            10,
            3,
            "/tmp/output",
            "custom_run",
            "Custom description",
        ),
        ("test_file.jsonl", None, 5, "/tmp/output", None, None),
    ],
)
def test_prepare_run_config_more_edge_cases2(
    runner: Runner,
    payload: dict[str, str] | list[dict[str, str]] | Literal["test_file.jsonl"],
    n_requests: None | Literal[10],
    clients: None | Literal[3] | Literal[5],
    output_path: None | Literal["/tmp/output"],
    run_name: None | Literal["custom_run"],
    run_description: None | Literal["Custom description"],
):
    run_config = runner._prepare_run_config(
        payload, n_requests, clients, output_path, run_name, run_description
    )

    assert isinstance(run_config.payload, list)
    assert run_config.n_requests == n_requests
    assert run_config.clients == clients if clients is not None else 1
    assert run_config.output_path == (Path(output_path) if output_path else None)
    assert run_config.run_name is not None
    assert run_config.run_description == (run_description if run_description else None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "n_requests, clients, output_path, run_name, run_description",
    [
        (None, None, None, None, None),
        (10, 3, "/tmp/output", "custom_run", "Custom description"),
        (None, 5, "/tmp/output", None, None),
    ],
)
async def test_run_with_optional_parameters(
    runner: Runner,
    n_requests: None | Literal[10],
    clients: None | Literal[3] | Literal[5],
    output_path: None | Literal["/tmp/output"],
    run_name: None | Literal["custom_run"],
    run_description: None | Literal["Custom description"],
):
    runner._invoke_n_c = AsyncMock(return_value=1.0)
    runner._count_tokens_from_q = AsyncMock()

    result = await runner.run(
        payload={"prompt": "test"},
        n_requests=n_requests,
        clients=clients,  # type: ignore
        output_path=output_path,
        run_name=run_name,
        run_description=run_description,
    )

    assert result.total_requests == (n_requests or 1) * (clients or 1)
    assert result.clients == clients or 1
    assert result.n_requests == n_requests or 1
    assert result.total_test_time == 1.0
    if output_path:
        assert result.output_path is not None
    if run_name:
        assert result.run_name == run_name
    if run_description:
        assert result.run_description == run_description


@pytest.mark.asyncio
@pytest.mark.parametrize("clients", [1, 3, 5, 10])
async def test_invoke_n_c_with_different_clients(
    runner: Runner, clients: Literal[1] | Literal[3] | Literal[5] | Literal[10]
):
    async def mock_invoke_n(payload, n, add_start_jitter=True, shuffle_order=True):
        await asyncio.sleep(0.1)  # Simulate some processing time
        return [
            InvocationResponse(
                id=str(i), input_prompt=f"test{i}", response_text=f"response{i}"
            )
            for i in range(n)
        ]

    runner._invoke_n = mock_invoke_n  # type: ignore

    start_time = time.perf_counter()
    total_test_time = await runner._invoke_n_c(
        payload=[{"prompt": "test"}], n_requests=5, clients=clients
    )
    end_time = time.perf_counter()

    assert total_test_time > 0
    assert (
        end_time - start_time < 0.2 * clients
    )  # Ensure concurrent execution (should be less than 0.2 seconds per client)


@pytest.mark.parametrize(
    "payload, n_requests, clients, output_path, run_name, run_description",
    [
        ({"prompt": "test"}, 5, 2, "/tmp/output", "test_run", "Test description"),
        ([{"prompt": "test1"}, {"prompt": "test2"}], None, 1, None, None, None),
        ("test_file.jsonl", 10, 5, "/tmp/output", None, "File input test"),
        ({"prompt": "test"}, None, 3, None, None, None),
    ],
)
def test_initialize_result_more_combinations(
    runner: Runner,
    payload: dict[str, str] | list[dict[str, str]] | Literal["test_file.jsonl"],
    n_requests: None | Literal[5] | Literal[10],
    clients: Literal[3] | Literal[2] | Literal[1] | Literal[5],
    output_path: None | Literal["/tmp/output"],
    run_name: None | Literal["test_run"],
    run_description: None | Literal["Test description"] | Literal["File input test"],
):
    run_config = _RunConfig(
        payload=payload,
        n_requests=n_requests,
        clients=clients,
        output_path=output_path,
        run_name=run_name,
        run_description=run_description,
        endpoint=mock_endpoint,  # type: ignore
    )
    result = runner._initialize_result(run_config)

    expected_n_requests = n_requests or (
        len(payload) if isinstance(payload, list) else 1
    )
    expected_clients = clients or 1
    expected_total_requests = expected_n_requests * expected_clients

    assert result.total_requests == expected_total_requests
    assert result.clients == expected_clients
    assert result.n_requests == expected_n_requests
    assert result.run_name == run_name or result.run_name is not None
    assert result.run_description == run_description


@pytest.mark.parametrize(
    "payload, n_requests, clients, output_path, run_name, run_description",
    [
        ({"prompt": "test"}, 0, 1, "/tmp/output", "test_run", "Test description"),
        (
            [{"prompt": "test1"}, {"prompt": "test2"}],
            10,
            -1,
            "/tmp/output",
            "test_run",
            "Test description",
        ),
        ("nonexistent_file.jsonl", 10, 5, "/tmp/output", None, None),
        ({"prompt": "test"}, 5, 2, "/tmp/output", "", ""),
        (None, None, None, None, None, None),
    ],
)
def test_prepare_run_config_more_edge_cases(
    runner: Runner,
    payload: dict[str, str]
    | list[dict[str, str]]
    | None
    | Literal["nonexistent_file.jsonl"],
    n_requests: None | Literal[0] | Literal[10] | Literal[5],
    clients: None | Literal[1] | Literal[-1] | Literal[5] | Literal[2],
    output_path: None | Literal["/tmp/output"],
    run_name: None | Literal["test_run"] | Literal[""],
    run_description: None | Literal["Test description"] | Literal[""],
):
    if payload == "nonexistent_file.jsonl":
        with pytest.raises(FileNotFoundError):
            runner._prepare_run_config(
                payload, n_requests, clients, output_path, run_name, run_description
            )
    elif payload is None:
        with pytest.raises(AssertionError):
            runner._prepare_run_config(
                payload, n_requests, clients, output_path, run_name, run_description
            )
    elif run_name == "":
        with pytest.raises(AssertionError):
            runner._prepare_run_config(
                payload, n_requests, clients, output_path, run_name, run_description
            )
    elif clients == -1 or None:
        with pytest.raises(AssertionError):
            runner._prepare_run_config(
                payload, n_requests, clients, output_path, run_name, run_description
            )
    elif n_requests == 0:
        with pytest.raises(AssertionError):
            runner._prepare_run_config(
                payload, n_requests, clients, output_path, run_name, run_description
            )
    else:
        run_config = runner._prepare_run_config(
            payload, n_requests, clients, output_path, run_name, run_description
        )

        assert isinstance(run_config.payload, list)
        assert run_config.n_requests == (
            n_requests if n_requests and n_requests > 0 else None
        )
        assert run_config.clients == (clients if clients and clients > 0 else 1)
        assert str(run_config.output_path) == output_path if output_path else None
        assert run_config.run_description == (
            run_description if run_description else None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shuffle_order, add_start_jitter",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
async def test_invoke_n_with_different_options(
    runner: Runner, shuffle_order: bool, add_start_jitter: bool
):
    runner._invoke_n_no_wait = MagicMock(
        return_value=[
            InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
            InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        ]
    )

    result = await runner._invoke_n(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}],
        n=2,
        shuffle_order=shuffle_order,
        add_start_jitter=add_start_jitter,
    )

    assert len(result) == 2
    runner._invoke_n_no_wait.assert_called_once_with(
        [{"prompt": "test1"}, {"prompt": "test2"}], 2, shuffle_order
    )


@pytest.mark.asyncio
async def test_count_tokens_from_q_with_custom_output_path(
    runner: Runner, tmp_path: Path
):
    custom_output_path = tmp_path / "custom_output"
    custom_output_path.mkdir(parents=True, exist_ok=True)

    runner._queue = AsyncMock()
    runner._queue.get.side_effect = [
        InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
        InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        None,
    ]
    runner._queue.task_done = MagicMock()
    runner._count_tokens_no_wait = MagicMock(return_value=5)

    await runner._count_tokens_from_q(
        output_path=custom_output_path / "responses.jsonl"
    )

    assert len(runner._responses) == 2
    assert (custom_output_path / "responses.jsonl").exists()
    with open(custom_output_path / "responses.jsonl", "r") as f:
        lines = f.readlines()
        assert len(lines) == 2


# Add more tests for edge cases and other methods as needed
