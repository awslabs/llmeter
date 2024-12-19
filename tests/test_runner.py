# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import time
from typing import Literal
from unittest.mock import AsyncMock, call, MagicMock, patch

import pytest
from upath import UPath as Path

import llmeter.endpoints
from llmeter.endpoints.base import Endpoint, InvocationResponse
from llmeter.runner import Runner, _Run, _RunConfig
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
def run(mock_endpoint: MagicMock, mock_tokenizer: MagicMock) -> _Run:
    mock_run = _Run(
        payload={"dummy": "payload"},
        run_name="dummy-run",
        endpoint=mock_endpoint,
        tokenizer=mock_tokenizer,
    )
    mock_run._responses = []
    mock_run._progress_bar = MagicMock()
    mock_run._queue = AsyncMock()
    mock_run._queue.task_done = MagicMock()

    return mock_run


@pytest.fixture
def runner(mock_endpoint: MagicMock, mock_tokenizer: MagicMock) -> Runner:
    mock_runner = Runner(endpoint=mock_endpoint, tokenizer=mock_tokenizer)

    return mock_runner


def test_run_initialization(run: _Run):
    assert isinstance(run, _Run)
    assert isinstance(run, _RunConfig)


@pytest.mark.asyncio
async def test_count_tokens_from_q(run: _Run):
    run._queue = AsyncMock()
    run._queue.get.side_effect = [
        InvocationResponse(
            id="1",
            input_prompt="test",
            response_text="response",
            num_tokens_input=5,
            num_tokens_output=5,
        ),
        None,
    ]
    run._queue.task_done = MagicMock()

    await run._process_results_from_q()

    assert len(run._responses) == 1
    assert run._responses[0].num_tokens_input == 5
    assert run._responses[0].num_tokens_output == 5


@pytest.mark.asyncio
async def test_invoke_n(run: _Run):
    run._invoke_n_no_wait = MagicMock(
        return_value=[
            InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
            InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        ]
    )

    result = await run._invoke_n(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}], n=2
    )

    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"


@pytest.mark.asyncio
async def test_run(runner: Runner):
    result = await runner.run(payload={"prompt": "test"}, n_requests=1, clients=1)

    assert result.total_requests == 1
    assert result.clients == 1
    assert result.n_requests == 1


@pytest.mark.asyncio
async def test_invoke_n_no_wait(run: _Run):
    run._endpoint.invoke.side_effect = [
        InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
        InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
    ]

    run._queue = AsyncMock()
    run._queue._loop.call_soon_threadsafe = MagicMock()

    result = run._invoke_n_no_wait(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}], n=2
    )

    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"


@pytest.mark.asyncio
async def test_invoke_n_c(run: _Run):
    run._invoke_n = AsyncMock(
        return_value=[
            InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
            InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        ]
    )

    total_test_time, _, _ = await run._invoke_n_c(
        payload=[{"prompt": "test"}], n_requests=2, clients=1
    )

    assert isinstance(total_test_time, float)
    assert run._invoke_n.call_count == 1


@pytest.mark.asyncio
async def test_run_with_file_payload(runner: Runner, tmp_path: Path):
    payload_file = tmp_path / "payload.jsonl"
    payload_file.write_text('{"prompt": "test1"}\n{"prompt": "test2"}')

    result = await runner.run(payload=str(payload_file), n_requests=2, clients=1)

    assert result.total_requests == 2
    assert result.clients == 1
    assert result.n_requests == 2


@pytest.mark.asyncio
async def test_run_with_output_path(runner: Runner, tmp_path: Path):
    # Implicit path from runner gets run name appended:
    runner.output_path = Path(tmp_path)
    result = await runner.run(
        payload={"prompt": "test"}, n_requests=1, clients=1, run_name="run_1"
    )
    assert result.output_path is not None
    assert (tmp_path / result.run_name / "run_config.json").exists()  # type: ignore

    # Explicit path in run() is used as-is:
    result = await runner.run(
        payload={"prompt": "test"},
        n_requests=1,
        clients=1,
        run_name="run_2",
        output_path=Path(tmp_path) / "custom_path",
    )
    assert result.output_path is not None
    assert (tmp_path / "custom_path" / "run_config.json").exists()  # type: ignore


@pytest.mark.asyncio
async def test_run_error_handling(run: _Run):
    run._invoke_n_c = AsyncMock(side_effect=Exception("Test error"))
    run._process_results_from_q = AsyncMock()

    with pytest.raises(Exception, match="Test error"):
        await run._run()


@pytest.mark.asyncio
async def test_run_callback_triggered(run: _Run):
    callback_mock = MagicMock()
    callback_mock.before_invoke = AsyncMock()
    callback_mock.after_invoke = AsyncMock()
    callback_mock.before_run = AsyncMock()
    callback_mock.after_run = AsyncMock()

    run.callbacks = [callback_mock]
    result = await run._run()

    callback_mock.before_run.assert_called_once_with(run)
    callback_mock.after_invoke.assert_has_calls([call(res) for res in result.responses])
    callback_mock.after_run.assert_called_once_with(result)
    callback_mock.before_invoke.assert_has_calls([call(p) for p in run.payload])


@pytest.mark.asyncio
async def test_run_callbacks_triggered_in_order(run: _Run):
    payloads = []

    class DummySequenceCheckerCallback:
        def __init__(self, my_index, field="callbacksequencecheckerindex"):
            self.my_index = my_index
            self.field = field

        async def before_run(self, run_config):
            if hasattr(run_config, self.field):
                setattr(run_config, self.field, getattr(run_config, self.field) + 1)
            else:
                setattr(run_config, self.field, 0)
            current_index = getattr(run_config, self.field)
            if current_index != self.my_index:
                raise AssertionError(
                    f"Callback index {self.my_index} before_run called at index {current_index}"
                )

        async def before_invoke(self, payload: dict):
            if self.my_index == 0:
                payloads.append(payload)
            if self.field in payload:
                payload[self.field] += 1
            else:
                payload[self.field] = 0
            current_index = payload[self.field]
            if current_index != self.my_index:
                raise AssertionError(
                    f"Callback index {self.my_index} before_invoke called at index {current_index}"
                )

        async def after_invoke(self, response):
            if hasattr(response, self.field):
                setattr(response, self.field, getattr(response, self.field) + 1)
            else:
                setattr(response, self.field, 0)
            current_index = getattr(response, self.field)
            if current_index != self.my_index:
                raise AssertionError(
                    f"Callback index {self.my_index} after_invoke called at index {current_index}"
                )

        async def after_run(self, result):
            if hasattr(result, self.field):
                setattr(result, self.field, getattr(result, self.field) + 1)
            else:
                setattr(result, self.field, 0)
            current_index = getattr(result, self.field)
            if current_index != self.my_index:
                raise AssertionError(
                    f"Callback index {self.my_index} after_run called at index {current_index}"
                )

    run.callbacks = [
        DummySequenceCheckerCallback(0),
        DummySequenceCheckerCallback(1),
        DummySequenceCheckerCallback(2),
    ]
    result = await run._run()

    assert run.callbacksequencecheckerindex == 2
    assert result.callbacksequencecheckerindex == 2
    for response in result.responses:
        assert response.callbacksequencecheckerindex == 2
    assert len(payloads) == len(result.responses)
    for payload in payloads:
        assert payload.callbacksequencecheckerindex == 2


@pytest.mark.asyncio
async def test_prepare_run(runner: Runner):
    payload = {"prompt": "test"}
    n_requests = 5
    clients = 2
    output_path = "/tmp/output"
    run_name = "test_run"
    run_description = "Test run description"
    callbacks = []

    run_config = runner._prepare_run(
        payload=payload,
        n_requests=n_requests,
        clients=clients,
        output_path=output_path,
        run_name=run_name,
        run_description=run_description,
        callbacks=callbacks,
    )

    assert run_config.payload == [payload]
    assert run_config.n_requests == n_requests
    assert run_config.clients == clients
    assert str(run_config.output_path) == output_path
    assert run_config.run_name == run_name
    assert run_config.run_description == run_description


def test_validate_and_prepare_payload(run: _Run, mock_endpoint: Endpoint):
    # Test with dict payload
    run.payload = {"prompt": "test"}
    run.endpoint = mock_endpoint
    run._validate_and_prepare_payload()
    assert isinstance(run.payload, list)
    assert len(run.payload) == 1

    # Test with list payload
    run.payload = [{"prompt": "test1"}, {"prompt": "test2"}]
    run._validate_and_prepare_payload()
    run._validate_and_prepare_payload()
    assert isinstance(run.payload, list)
    assert len(run.payload) == 2

    # Test with file payload (mock)
    with pytest.raises(FileNotFoundError):
        run.payload = "nonexistent_file.jsonl"
        run._validate_and_prepare_payload()


def test_run_output_path(runner: Runner, tmp_path: Path):
    # Test with provided run_name
    run = runner._prepare_run(
        payload={"prompt": "hi"}, output_path=str(tmp_path), run_name="test_run"
    )
    assert run.run_name == "test_run"
    assert isinstance(run.output_path, Path)

    # Test without provided run_name (should generate one)
    run = runner._prepare_run(
        payload={"prompt": "hi"},
        output_path=str(tmp_path),
    )
    assert run.run_name is not None
    assert isinstance(run.output_path, Path)


@pytest.mark.asyncio
async def test_invoke_n_edge_cases(run: _Run):
    # Test with empty payload
    result = await run._invoke_n(payload=[], n=5)
    assert not result

    # Test with n=None (should use all payloads)
    run._invoke_n_no_wait = MagicMock(
        return_value=[
            InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
            InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        ]
    )
    result = await run._invoke_n(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}], n=None
    )
    assert len(result) == 2


@pytest.mark.asyncio
async def test_run_with_sequence_payload(runner: Runner):
    result = await runner.run(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}], n_requests=2, clients=1
    )

    assert result.total_requests == 2
    assert result.clients == 1
    assert result.n_requests == 2


@pytest.mark.asyncio
async def test_count_tokens_from_q_timeout(run: _Run):
    run._queue = AsyncMock()
    run._queue.get.side_effect = asyncio.TimeoutError()

    await run._process_results_from_q()

    run._queue.assert_awaited_once
    run._queue.task_done = MagicMock()

    # assert len(run._responses) == 0


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
    loaded_config = Runner.load(tmp_path)

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
def test_prepare_run_combinations(
    runner: Runner,
    payload: dict[str, str] | list[dict[str, str]] | Literal["test_file.jsonl"],
    n_requests: None | Literal[5] | Literal[10],
    clients: Literal[2] | Literal[1] | Literal[5],
    output_path: None | Literal["/tmp/output"],
    run_name: None | Literal["test_run"],
    run_description: None | Literal["Test description"] | Literal["File input test"],
    callbacks=[],
):
    if payload == "test_file.jsonl":
        payload_file = Path(payload)
        payload_file.write_text('{"prompt": "test1"}\n{"prompt": "test2"}')

    run = runner._prepare_run(
        payload=payload,
        n_requests=n_requests,
        clients=clients,
        output_path=output_path,
        run_name=run_name,
        run_description=run_description,
        callbacks=callbacks,
    )

    assert isinstance(run.payload, list)
    assert run.n_requests == n_requests
    assert run.clients == clients
    assert run.output_path == (Path(output_path) if output_path else None)
    assert run.run_name is not None
    assert run.run_description == (run_description if run_description else None)


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

    result = await runner.run(payload=payload, n_requests=2, clients=1)

    assert result.total_requests == 2
    assert result.clients == 1
    assert result.n_requests == 2


@pytest.mark.asyncio
async def test_invoke_n_c_concurrent_execution(run: _Run):
    async def mock_invoke_n(payload, n, add_start_jitter=True, shuffle_order=True):
        await asyncio.sleep(0.1)  # Simulate some processing time
        return [
            InvocationResponse(
                id=str(i), input_prompt=f"test{i}", response_text=f"response{i}"
            )
            for i in range(n)
        ]

    run._invoke_n = mock_invoke_n  # type: ignore

    start_time = time.perf_counter()
    total_test_time, _, _ = await run._invoke_n_c(
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
def test_prepare_run_invalid_inputs(
    runner: Runner,
    payload: None | dict[str, str],
    n_requests: Literal[5] | Literal[-1],
    clients: Literal[2] | Literal[0],
):
    with pytest.raises(AssertionError):
        runner._prepare_run(
            payload=payload,
            n_requests=n_requests,
            clients=clients,
            output_path=None,
            run_name=None,
            run_description=None,
            callbacks=None,
        )


@pytest.mark.asyncio
async def test_count_tokens_from_q_different_scenarios(run: _Run):
    # Scenario 1: Queue with multiple items
    run._queue = AsyncMock()
    run._queue.task_done = MagicMock()
    run._queue.get.side_effect = [
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

    await run._process_results_from_q()

    assert len(run._responses) == 2
    assert run._responses[0].num_tokens_input == 5
    assert run._responses[0].num_tokens_output == 5
    assert run._responses[1].num_tokens_input == 5
    assert run._responses[1].num_tokens_output == 5

    # Scenario 2: Queue with exception
    run._queue.reset_mock()
    run._responses = []
    run._queue.get.side_effect = Exception("Test exception")
    run._queue.task_done = MagicMock()

    await run._process_results_from_q()

    # assert len(run._responses) == 0
    assert not run._responses


@pytest.mark.parametrize(
    "payload, n_requests, clients, output_path, run_name, run_description",
    [
        # (None, 5, 2, "/tmp/output", "test_run", "Test description"),
        ({"prompt": "test"}, -1, 2, "/tmp/output", "test_run", "Test description"),
        ({"prompt": "test"}, 5, 0, "/tmp/output", "test_run", "Test description"),
        ({"prompt": "test"}, 5, 2, "/tmp/output", "", "Test description"),
    ],
)
def test_prepare_run_edge_cases(
    runner: Runner,
    payload: None | dict[str, str],
    n_requests: Literal[5] | Literal[-1],
    clients: Literal[2] | Literal[0],
    output_path: Literal["/tmp/output"],
    run_name: Literal["test_run"] | Literal[""],
    run_description: Literal["Test description"],
    callbacks=[],
):
    with pytest.raises((AssertionError, ValueError)):
        runner._prepare_run(
            payload=payload,
            n_requests=n_requests,
            clients=clients,
            output_path=output_path,
            run_name=run_name,
            run_description=run_description,
            callbacks=callbacks,
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
def test_prepare_run_more_edge_cases(
    runner: Runner,
    payload: dict[str, str] | list[dict[str, str]] | Literal["test_file.jsonl"],
    n_requests: None | Literal[10],
    clients: None | Literal[3] | Literal[5],
    output_path: None | Literal["/tmp/output"],
    run_name: None | Literal["custom_run"],
    run_description: None | Literal["Custom description"],
    callbacks=[],
):
    run_config = runner._prepare_run(
        payload=payload,
        n_requests=n_requests,
        clients=clients,
        output_path=output_path,
        run_name=run_name,
        run_description=run_description,
        callbacks=callbacks,
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
    if output_path:
        assert result.output_path is not None
    if run_name:
        assert result.run_name == run_name
    if run_description:
        assert result.run_description == run_description


@pytest.mark.asyncio
@pytest.mark.parametrize("clients", [1, 3, 5, 10])
async def test_invoke_n_c_with_different_clients(
    run: _Run, clients: Literal[1] | Literal[3] | Literal[5] | Literal[10]
):
    async def mock_invoke_n(payload, n, add_start_jitter=True, shuffle_order=True):
        await asyncio.sleep(0.1)  # Simulate some processing time
        return [
            InvocationResponse(
                id=str(i), input_prompt=f"test{i}", response_text=f"response{i}"
            )
            for i in range(n)
        ]

    run._invoke_n = mock_invoke_n  # type: ignore

    start_time = time.perf_counter()
    total_test_time, _, _ = await run._invoke_n_c(
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
def test_prepare_run_more_edge_cases(
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
    callbacks=[],
):
    if payload == "nonexistent_file.jsonl":
        with pytest.raises(FileNotFoundError):
            runner._prepare_run(
                payload=payload,
                n_requests=n_requests,
                clients=clients,
                output_path=output_path,
                run_name=run_name,
                run_description=run_description,
                callbacks=callbacks,
            )
    elif payload is None:
        with pytest.raises(AssertionError):
            runner._prepare_run(
                payload=payload,
                n_requests=n_requests,
                clients=clients,
                output_path=output_path,
                run_name=run_name,
                run_description=run_description,
                callbacks=callbacks,
            )
    elif run_name == "":
        with pytest.raises(AssertionError):
            runner._prepare_run(
                payload=payload,
                n_requests=n_requests,
                clients=clients,
                output_path=output_path,
                run_name=run_name,
                run_description=run_description,
                callbacks=callbacks,
            )
    elif clients == -1 or None:
        with pytest.raises(AssertionError):
            runner._prepare_run(
                payload=payload,
                n_requests=n_requests,
                clients=clients,
                output_path=output_path,
                run_name=run_name,
                run_description=run_description,
                callbacks=callbacks,
            )
    elif n_requests == 0:
        with pytest.raises(AssertionError):
            runner._prepare_run(
                payload=payload,
                n_requests=n_requests,
                clients=clients,
                output_path=output_path,
                run_name=run_name,
                run_description=run_description,
                callbacks=callbacks,
            )
    else:
        run_config = runner._prepare_run(
            payload=payload,
            n_requests=n_requests,
            clients=clients,
            output_path=output_path,
            run_name=run_name,
            run_description=run_description,
            callbacks=callbacks,
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
    run: _Run, shuffle_order: bool, add_start_jitter: bool
):
    run._invoke_n_no_wait = MagicMock(
        return_value=[
            InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
            InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        ]
    )

    result = await run._invoke_n(
        payload=[{"prompt": "test1"}, {"prompt": "test2"}],
        n=2,
        shuffle_order=shuffle_order,
        add_start_jitter=add_start_jitter,
    )

    assert len(result) == 2
    run._invoke_n_no_wait.assert_called_once_with(
        [{"prompt": "test1"}, {"prompt": "test2"}], 2, shuffle_order
    )


@pytest.mark.asyncio
async def test_count_tokens_from_q_with_custom_output_path(run: _Run, tmp_path: Path):
    custom_output_path = tmp_path / "custom_output"
    custom_output_path.mkdir(parents=True, exist_ok=True)

    run._queue = AsyncMock()
    run._queue.get.side_effect = [
        InvocationResponse(id="1", input_prompt="test1", response_text="response1"),
        InvocationResponse(id="2", input_prompt="test2", response_text="response2"),
        None,
    ]
    run._queue.task_done = MagicMock()
    # run._count_tokens_no_wait = MagicMock(return_value=5)

    await run._process_results_from_q(
        output_path=custom_output_path / "responses.jsonl"
    )

    assert len(run._responses) == 2
    assert (custom_output_path / "responses.jsonl").exists()
    with open(custom_output_path / "responses.jsonl", "r") as f:
        lines = f.readlines()
        assert len(lines) == 2


# Add more tests for edge cases and other methods as needed
