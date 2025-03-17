# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import InitVar, asdict, dataclass, fields, replace
from datetime import datetime
from itertools import cycle
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tqdm.auto import tqdm, trange
from upath import UPath as Path

from llmeter.utils import now_utc

if TYPE_CHECKING:
    # Avoid circular import: We only need typing for Callback
    from .callbacks.base import Callback

from .endpoints.base import Endpoint, InvocationResponse
from .prompt_utils import load_payloads, save_payloads
from .results import Result
from .tokenizers import DummyTokenizer, Tokenizer

logger = logging.getLogger(__name__)

# using a custom env variable because the TQDM one (https://github.com/tqdm/tqdm/issues/612#issuecomment-2015702344) doesn't work reliably
_disable_tqdm = False
if os.getenv("LLMETER_DISABLE_ALL_PROGRESS_BARS") == "1":
    logger.info("Disabling tqdm progress bars")
    _disable_tqdm = True


@dataclass
class _RunConfig:
    """A class to store the configuration for a test run.

    Both Runner (which provides a base configuration from which multiple runs can be triggered),
    and _Run (which represents a specific individual test run) inherit from this shared base - for
    consistency and simplicity of configuration de/serialization.

    See public `Runner` docstring for fields documentation, since this class is internal.
    """

    endpoint: Endpoint | dict | None = None
    output_path: str | Path | None = None
    tokenizer: Tokenizer | Any | None = None
    clients: int = 1
    n_requests: int | None = None
    payload: dict | list[dict] | os.PathLike | str | None = None
    run_name: str | None = None
    run_description: str | None = None
    timeout: int | float = 60
    callbacks: list[Callback] | None = None
    disable_per_client_progress_bar: InitVar[bool] = True
    disable_clients_progress_bar: InitVar[bool] = True

    def __post_init__(self, disable_client_progress_bar, disable_clients_progress_bar):
        self._disable_per_client_progress_bar = disable_client_progress_bar
        self._disable_clients_progress_bar = disable_clients_progress_bar
        self._random_seed = 0

        if self.n_requests is not None:
            assert self.n_requests > 0, "Number of requests must be a positive integer"

        assert self.clients > 0, "Number of clients must be a positive integer"

        if self.run_name is not None:
            assert len(self.run_name) > 0, "Run name must be a non-empty string"

        assert self.endpoint is not None, "Endpoint cannot be None"
        if isinstance(self.endpoint, dict):
            self._endpoint: Endpoint = Endpoint.load(self.endpoint)
        else:
            self._endpoint = self.endpoint

        if self.output_path is not None:
            self.output_path = Path(self.output_path)

        if self.tokenizer is None:
            self.tokenizer = DummyTokenizer()
        if isinstance(self.tokenizer, dict):
            self._tokenizer: Tokenizer = Tokenizer.load(self.tokenizer)
        else:
            self._tokenizer = self.tokenizer

    def save(
        self,
        output_path: os.PathLike | str | None = None,
        file_name: str = "run_config.json",
    ):
        """Save the configuration to a disk or cloud storage.

        Args:
            output_path: Optional override for output folder. By default, self.output_path is used.
            file_name: File name to create under `output_path`.
        """
        output_path = Path(output_path or self.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        run_config_path = output_path / file_name

        config_copy = replace(self)

        if self.payload and (not isinstance(self.payload, (os.PathLike, str))):
            payload_path = save_payloads(self.payload, output_path)
            config_copy.payload = payload_path

        assert self.endpoint is not None, "Endpoint cannot be None"
        if not isinstance(self.endpoint, dict):
            config_copy.endpoint = self.endpoint.to_dict()

        if not isinstance(self.tokenizer, dict):
            config_copy.tokenizer = Tokenizer.to_dict(self.tokenizer)

        with run_config_path.open("w") as f:
            f.write(json.dumps(asdict(config_copy), default=str, indent=4))

    @classmethod
    def load(cls, load_path: Path | str, file_name: str = "run_config.json"):
        """Load a configuration from a (local or cloud-stored) JSON file.

        Args:
            output_path: Folder under which the configuration is stored
            file_name: File name within `output_path` for the run configuration JSON.
        """
        load_path = Path(load_path)
        with open(load_path / file_name) as f:
            config = json.load(f)
        config["endpoint"] = Endpoint.load(config["endpoint"])
        config["tokenizer"] = Tokenizer.load(config["tokenizer"])
        return cls(**config)


@dataclass
class _Run(_RunConfig):
    """Class to manage one specific test run

    This class is not intended to be used directly: Instead create a `Runner` to define a default
    configuration, then call `runner.run()` with optional run-specific overrides.
    """

    def __post_init__(self, disable_client_progress_bar, disable_clients_progress_bar):
        assert (
            self.run_name is not None
        ), "Test Run must be created with an explicit run_name"

        super().__post_init__(disable_client_progress_bar, disable_clients_progress_bar)

        assert (
            self.endpoint is not None
        ), "Test Run must be created with an explicit Endpoint"

        self._validate_and_prepare_payload()
        self._responses = []

    def _validate_and_prepare_payload(self):
        """Validate and prepare the payload for the test run and update n_requests

        This method ensures that the payload is valid and prepared for the test run.
        """
        assert self.payload, "No payload provided"
        if isinstance(self.payload, (os.PathLike, str)):
            self.payload = list(load_payloads(self.payload))
        if isinstance(self.payload, dict):
            self.payload = [self.payload]
        self._n_requests = self.n_requests or len(self.payload)

    @staticmethod
    async def _compute_time_per_output_token(response: InvocationResponse):
        """
        Compute the time per output token for the given response.

        Args:
            response (InvocationResponse): The response to compute time per output token for.
        """
        if response.time_per_output_token is None:
            if (
                response.time_to_last_token
                and response.num_tokens_output
                and response.time_to_first_token
            ):
                response.time_per_output_token = (
                    response.time_to_last_token - response.time_to_first_token
                ) / (response.num_tokens_output - 1)

    @staticmethod
    async def _update_token_counts(tokenizer: Tokenizer, response: InvocationResponse):
        """
        Update the token counts for the given response.

        Args:
            response (InvocationResponse): The response to update token counts for.
        """
        if response.error is not None:
            return

        if response.num_tokens_input is None:
            text = response.input_prompt
            if text is None:
                response.num_tokens_input = None
            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception:
                    raise ValueError("provided input can't be converted to string")
            response.num_tokens_input = len(tokenizer.encode(text))

        if response.num_tokens_output is None:
            text = response.response_text
            if text is None:
                response.num_tokens_output = None
            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception:
                    raise ValueError("generated output can't be converted to string")
            response.num_tokens_output = len(tokenizer.encode(text))

    async def _process_results_from_q(self, output_path: Path | None = None):
        logger.info("Starting token counting from queue")
        while True:
            try:
                response: InvocationResponse | None = await asyncio.wait_for(
                    self._queue.get(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for response")
                break
            except Exception as e:
                logger.exception(e)
                break

            if response is None:
                logger.debug("Got None from queue, stopping")
                break

            logger.debug(f"Got {response.id} from queue")

            await self._update_token_counts(self._tokenizer, response)
            await self._compute_time_per_output_token(response)
            if self.callbacks is not None:
                [await cb.after_invoke(response) for cb in self.callbacks]

            self._responses.append(response)
            if self._progress_bar:
                self._progress_bar.update(1)

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("a") as f:
                    f.write(response.to_json() + "\n")

            self._queue.task_done()

    def _invoke_n_no_wait(
        self,
        payload: list[dict],
        n: int | None = None,
        shuffle_order=True,
    ) -> list[InvocationResponse]:
        """
        Generate multiple invocations for the given payload.

        This method generates `n` invocations for the given payload(s) by sending
        requests to the endpoint in a loop. If a sequence of payloads is provided,
        the payloads are cycled through until `n` invocations are generated. If a
        single payload is provided, it is used for all `n` invocations.

        Args:
            payload: The input payload to generate invocations for.
            n (int|None, optional): The number of invocations to generate.
                If not specified, every element in the payload is used once.
            shuffle_order (bool, optional): Whether to shuffle the order of payloads
                before generating invocations. Defaults to True.

        Returns:
            List[EndpointResponse]: A list of response objects.
        """

        # ToDo: replace with an async method to prepare payloads, including possible callbacks,
        #  and feed them to the endpoint as needed
        if shuffle_order:
            self._random_seed += random.randint(1, 1000)
            random.seed(0)
            payload = random.sample(payload, k=len(payload))

        responses = []
        if n is None:
            n = len(payload)
        for p, _ in zip(
            cycle(payload),
            trange(
                n,
                leave=False,
                desc="Requests",
                disable=_disable_tqdm or self._disable_per_client_progress_bar,
            ),
        ):
            try:
                p = asyncio.run(process_before_invoke_callbacks(self.callbacks, p))
                response = self._endpoint.invoke(p)

            except Exception as e:
                logger.exception(f"Error with invocation with payload {p}: {e}")
                response = InvocationResponse.error_output(
                    id=uuid4().hex,
                    error=str(e),
                )
            responses.append(response)
            if self._queue:
                # fix for thread-aware sync, from https://stackoverflow.com/a/57316517/2109965
                self._queue._loop.call_soon_threadsafe(  # type: ignore
                    self._queue.put_nowait, response
                )
        return responses

    async def _invoke_n(
        self,
        payload: list[dict],
        n: int | None = None,
        add_start_jitter=True,
        shuffle_order=True,
    ) -> list[InvocationResponse]:
        """
        Asynchronously generate multiple invocations for the given payload.

        This method generates `n` invocations for the given payload(s) by sending
        requests to the endpoint asynchronously. If a sequence of payloads is provided,
        the payloads are cycled through until `n` invocations are generated. If a
        single payload is provided, it is used for all `n` invocations.

        Args:
            payload (Dict[str, str] | Sequence[Dict[str, str]]): The input payload(s) to generate invocations for.
            n (int | None, optional): The number of invocations to generate. Defaults to None.
            add_start_jitter (bool, optional): Whether to add a random delay before
                starting the invocations loop to avoid batch bunching when using
                multiple clients. Defaults to True.
            shuffle_order (bool, optional): Whether to shuffle the order of payloads
                before generating invocations. Defaults to True.

        Returns:
            List[EndpointResponse]: A list of response objects.
        """

        if add_start_jitter:
            await asyncio.sleep(random.random() * 0.01)

        if shuffle_order:
            self._random_seed = random.randint(0, 2**16 - 1)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._invoke_n_no_wait, payload, n, shuffle_order),
                timeout=self.timeout * (n or len(payload)),
            )
        except asyncio.TimeoutError:
            logger.error("client timeout!")
            return []

        return response

    async def _invoke_n_c(
        self,
        payload: list[dict],
        n_requests: int | None = None,
        clients: int = 1,
    ) -> tuple[float, float, float]:
        """
        Asynchronously generates multiple invocations for a given payload.

        Args:
            payload (dict): The input data for generating invocations.
            queue (asyncio.Queue): The queue to store the generated responses.
            n_requests (int | None, optional): The number of invocations to generate per connection. Defaults to None.
            clients (int, optional): The number of concurrent connections to generate invocations. Defaults to 1.

        Returns:
            None

        Raises:
            None
        """
        logger.info(
            f"Generating {clients} connections with {n_requests} invocations each"
        )
        start_t = time.perf_counter()
        await tqdm.gather(
            *[self._invoke_n(payload, n_requests) for _ in range(clients)],
            leave=False,
            desc="Clients",
            disable=_disable_tqdm or self._disable_clients_progress_bar,
        )
        end_t = time.perf_counter()
        total_test_time = end_t - start_t
        logger.info(
            f"Generated {clients} connections with {n_requests} invocations each in {total_test_time*1000:.2f} seconds"
        )

        # Signal the token counting task to exit
        if self._queue:
            await self._queue.put(None)
            logger.debug("Signaling token counting task to exit")
        return total_test_time, start_t, end_t

    async def _run(self):
        """Run the test with the given configuration

        This method is expected to be called *exactly once* after the _Run object is created.
        Attempting to re-use a _Run object may result in undefined behavior.
        """
        result = Result(
            responses=[],
            total_test_time=None,
            total_requests=self._n_requests * self.clients,
            clients=self.clients,
            n_requests=self._n_requests,
            output_path=self.output_path,  # type: ignore
            model_id=self._endpoint.model_id,
            provider=self._endpoint.provider,
            endpoint_name=self._endpoint.endpoint_name,
            run_name=self.run_name,
            run_description=self.run_description,
        )

        if self.callbacks is not None:
            [await cb.before_run(self) for cb in self.callbacks]

        if self.output_path:
            self.save()  # Save run config & payload (after any callback transformations)

        # Address default threads limit in asyncio
        # https://stackoverflow.com/questions/75885213/how-to-increase-asyncio-thread-limits-in-an-existing-co-routine)
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=self.clients + 5))
        logger.info("Starting test")
        self._queue = asyncio.Queue()
        self._progress_bar = tqdm(
            total=self.clients * self._n_requests,
            leave=False,
            desc="Total requests",
            disable=_disable_tqdm,
        )

        try:
            run_start_time = now_utc()
            _, (total_test_time, start_time, end_time) = await asyncio.gather(
                self._process_results_from_q(
                    output_path=Path(self.output_path) / "responses.jsonl"
                    if self.output_path
                    else None,
                ),
                self._invoke_n_c(
                    payload=self.payload,  # type: ignore
                    n_requests=self._n_requests,
                    clients=self.clients,
                ),
            )
            run_end_time = now_utc()

        except asyncio.CancelledError:
            logger.error(
                f"Waited {self.timeout} seconds, but received no response. Test failed."
            )
            return result

        self._progress_bar.close()
        logger.info(f"Test completed in {total_test_time*1000:.2f} seconds.")

        result = replace(
            result,
            responses=self._responses,
            total_test_time=total_test_time,
            start_time=run_start_time,
            end_time=run_end_time,
        )

        if self.callbacks is not None:
            [await cb.after_run(result) for cb in self.callbacks]

        if result.output_path:
            result.save()

        return result


async def process_before_invoke_callbacks(
    callbacks: list[Callback] | None, payload: dict
) -> dict:
    """
    Process the `before_run` callbacks for a Run.

    This method is expected to be called *exactly once* after the _Run object is created.
    Attempting to re-use a _Run object may result in undefined behavior.

    Args:
        callbacks (list[Callback]): The list of callbacks to process.
    """
    if callbacks is not None:
        p = deepcopy(payload)

        [await cb.before_invoke(p) for cb in callbacks]
        return p
    return payload


@dataclass
class Runner(_RunConfig):
    """
    Run (one or more) LLM test sets using a base configuration.

    First create a `Runner` with base configuration for your test(s), then call `.run()` with
    optional run-specific overrides. This pattern allows you to group related runs together for
    organizing experiments (like ramping load tests) that might use more than one Run in total.

    All attributes of this class may be unset (as you may choose to set them only at the Run
    level), but some are "Mandatory" to be set *either* at the Runner or individual-run level, as
    described below.

    Attributes:
        endpoint (Endpoint | dict | None): The LLM endpoint to be tested. **Must be set** at either
            the Runner or specific Run level.
        output_path (os.PathLike | str | None): The (cloud or local) base folder under which run
            outputs and configurations should be stored. By default, outputs will not be saved to
            file.
        tokenizer (Tokenizer | Any | None): Optional tokenizer used to estimate input and output
            token counts for endpoints that don't report exact information. By default, LLMeter's
            `DummyTokenizer` will be used if needed.
        clients (int): The number of concurrent clients to use for sending requests. Defaults to 1.
        n_requests (int | None): The number of LLM invocations to generate *per client*. By
            default, each request in `payload` will be sent once by each client.
        payload (dict | list[dict] | os.PathLike | str | None): The request data to send to the
            endpoint under test. You can provide a single JSON payload (dict), a list of payloads
            (list[dict]), or a path to one or more JSON/JSON-Lines files to be loaded by
            `llmeter.prompt_utils.load_payloads()`. **Must be set** at either the Runner or
            specific Run level.
        run_name (str | None): Name to use for a specific test Run. This is *ignored* if set at the
            Runner level, and should instead be set in `Runner.run()` to name a specific run. By
            default, runs are named with the date and time they're requested in format:
            `%Y%m%d-%H%M`
        run_description (str | None): A natural-language description for the test Run. Can be set
            either at the Runner level (in which case the same description will be shared across
            all Runs), or individually in `Runner.run()`.
        timeout (int | float): The maximum time (in seconds) to wait for each response from the
            endpoint. Defaults to 60 seconds.
        callbacks (list[Callback] | None): Optional callbacks to enable during the test Run. See
            `llmeter.callbacks` for more information.
        disable_per_client_progress_bar (bool): Set `True` to disable per-client progress bars
            from showing during the run. Default `False` (each client's progress will be shown).
        disable_clients_progress_bar (bool): Set `True` to disable overall progress bar from
            showing during the run. Default `False` (overall requests progress will be shown).
    """

    def _prepare_run(self, **kwargs) -> _Run:
        """Create an individual run based on this runner's configurations, with optional overrides

        Args:
            **kwargs: Overrides to the Runner's default config for this specific run
        """
        run_params = {
            **{f.name: getattr(self, f.name) for f in fields(_RunConfig)},
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        if kwargs.get("run_name") is None:
            # Runner's own self.run_name is explicitly *not* used, as it would share between runs
            run_params["run_name"] = f"{datetime.now():%Y%m%d-%H%M}"
        if self.output_path and not kwargs.get("output_path"):
            # Run output path is nested under run name subfolder unless explicitly set:
            run_params["output_path"] = Path(self.output_path) / run_params["run_name"]
        # Validate that clients parameter is set and is a positive integer
        clients = run_params.get("clients")
        if clients is None:
            print(run_params)
            raise ValueError("Number of clients must be set")
        if not isinstance(clients, int) or clients <= 0:
            raise ValueError("Number of clients must be a positive integer")

        return _Run(**run_params)

    async def run(
        self,
        *,  # Prevent mistakes with this long arg list by allowing only keyword-arg based passing
        # Explicitly name and re-document the args for ease of use of this important public method
        endpoint: Endpoint | dict | None = None,
        output_path: Path | None = None,
        tokenizer: Tokenizer | Any | None = None,
        clients: int | None = None,
        n_requests: int | None = None,
        payload: dict | list[dict] | os.PathLike | str | None = None,
        run_name: str | None = None,
        run_description: str | None = None,
        timeout: int | float | None = None,
        callbacks: list[Callback] | None = None,
        disable_per_client_progress_bar: bool | None = None,
        disable_clients_progress_bar: bool | None = None,
    ) -> Result:
        """
        Run a test against an LLM endpoint

        This method tests the performance of the endpoint by sending multiple concurrent requests
        with the given payload(s). It measures the total time taken to complete the test, generates
        invocations for the given payload(s), and optionally saves the results and metrics.

        For arguments that are not specified, the Runner's attributes will be used by default.

        Args:
            endpoint (Endpoint | dict | None): The LLM endpoint to be tested. **Must be set** at
                either the Runner or specific Run level.
            output_path (os.PathLike | str | None): The (cloud or local) base folder under which
                run outputs and configurations should be stored. By default, a new `run_name`
                sub-folder will be created under the Runner's `output_path` if set - otherwise
                outputs will not be saved to file.
            tokenizer (Tokenizer | Any | None): Optional tokenizer used to estimate input and
                output token counts for endpoints that don't report exact information.
            clients (int): The number of concurrent clients to use for sending requests.
            n_requests (int | None): The number of LLM invocations to generate *per client*.
            payload (dict | list[dict] | os.PathLike | str | None): The request data to send to the
                endpoint under test. You can provide a single JSON payload (dict), a list of
                payloads (list[dict]), or a path to one or more JSON/JSON-Lines files to be loaded
                by `llmeter.prompt_utils.load_payloads()`. **Must be set** at either the Runner or
                specific Run level.
            run_name (str | None): Name to use for a specific test Run. By default, runs are named
                with the date and time they're requested in format: `%Y%m%d-%H%M`
            run_description (str | None): A natural-language description for the test Run.
            timeout (int | float): The maximum time (in seconds) to wait for each response from the
                endpoint.
            callbacks (list[Callback] | None): Optional callbacks to enable during the test Run. See
                `llmeter.callbacks` for more information.
            disable_per_client_progress_bar (bool): Set `True` to disable per-client progress bars
                from showing during the run.
            disable_clients_progress_bar (bool): Set `True` to disable overall progress bar from
                showing during the run.

        Returns:
            Result: An object containing the test results, including the generated
            response texts, total test time, total requests, number of clients,
            number of requests per client, and other relevant metrics.

        Raises:
            Exception: If there's an error during the test execution or if the
            endpoint cannot be reached.

        Note:
            - This method uses asyncio for concurrent processing.
            - Progress is displayed using tqdm if not disabled.
            - Responses are collected and processed asynchronously.
            - If an output_path is provided, results are saved to files.
        """

        run = self._prepare_run(
            endpoint=endpoint,
            output_path=output_path,
            tokenizer=tokenizer,
            clients=clients,
            n_requests=n_requests,
            payload=payload,
            run_name=run_name,
            run_description=run_description,
            timeout=timeout,
            callbacks=callbacks,
            disable_per_client_progress_bar=disable_per_client_progress_bar,
            disable_clients_progress_bar=disable_clients_progress_bar,
        )
        assert isinstance(run.payload, list)
        assert isinstance(run.run_name, str)
        return await run._run()

    def add_callback(self, callback: Callback):
        """
        Add a callback to the runner's list of callbacks.

        Args:
            callback (Callback): The callback to be added.
        """
        if self.callbacks is None:
            self.callbacks = [callback]
        else:
            self.callbacks.append(callback)
