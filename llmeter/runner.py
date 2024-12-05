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
from dataclasses import InitVar, asdict, dataclass, fields, replace
from datetime import datetime
from itertools import cycle
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from tqdm.auto import tqdm, trange
from upath import UPath as Path

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
    """A class to store the configuration for a test run."""

    endpoint: Endpoint | dict
    output_path: os.PathLike | str | None = None
    tokenizer: Tokenizer | Any | None = None
    n_requests: int | None = None
    clients: int = 1
    payload: dict | list[dict] | os.PathLike | str | None = None
    run_name: str | None = None
    run_description: str | None = None
    timeout: int | float = 60
    disable_per_client_progress_bar: InitVar[bool] = True
    disable_clients_progress_bar: InitVar[bool] = True
    callbacks: list[Callback] | None = None

    def __post_init__(self, disable_client_progress_bar, disable_clients_progress_bar):
        self._disable_per_client_progress_bar = disable_client_progress_bar
        self._disable_clients_progress_bar = disable_clients_progress_bar
        self._random_seed = 0

        if self.n_requests is not None:
            assert self.n_requests > 0, "Number of requests must be a positive integer"

        assert self.clients > 0, "Number of clients must be a positive integer"

        if self.run_name is not None:
            assert len(self.run_name) > 0, "Run name must be a non-empty string"

        if isinstance(self.endpoint, dict):
            self._endpoint: Endpoint = Endpoint.load(self.endpoint)
        else:
            self._endpoint = self.endpoint

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
        """Save the configuration to a disk or could storage."""
        output_path = Path(output_path or self.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        if self.run_name:
            output_path = output_path / self.run_name
        run_config_path = output_path / file_name

        config_copy = replace(self)

        if self.payload and (not isinstance(self.payload, (os.PathLike, str))):
            payload_path = save_payloads(self.payload, output_path)
            config_copy.payload = payload_path

        if not isinstance(self.endpoint, dict):
            config_copy.endpoint = self.endpoint.to_dict()

        if not isinstance(self.tokenizer, dict):
            config_copy.tokenizer = Tokenizer.to_dict(self.tokenizer)

        with run_config_path.open("w") as f:
            f.write(json.dumps(asdict(config_copy), default=str, indent=4))

    @classmethod
    def load(cls, load_path: Path | str, file_name: str = "run_config.json"):
        """Load a configuration from a JSON file."""
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
        super().__post_init__(disable_client_progress_bar, disable_clients_progress_bar)
        self._validate_and_prepare_payload()
        self._prepare_output_path()
        self._responses = []
        self._n_requests = self.n_requests or len(self.payload)

    async def _update_token_counts(self, response: InvocationResponse):
        """
        Update the token counts for the given response.

        Args:
            response (InvocationResponse): The response to update token counts for.
        """
        if response.num_tokens_input is None:
            text = response.input_prompt
            if text is None:
                response.num_tokens_input = 0
            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception:
                    raise ValueError("provided input can't be converted to string")
            response.num_tokens_input = len(self._tokenizer.encode(text))

        if response.num_tokens_output is None:
            text = response.response_text
            if text is None:
                response.num_tokens_output = 0
            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception:
                    raise ValueError("generated output can't be converted to string")
            response.num_tokens_output = len(self._tokenizer.encode(text))

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

            await self._update_token_counts(response)
            if self.callbacks is not None:
                [await cb.after_invoke(response) for cb in self.callbacks]

            self._responses.append(response)
            if self._progress_bar:
                self._progress_bar.update(1)

            if output_path:
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

    def _validate_and_prepare_payload(self):
        assert self.payload, "No payload provided"
        if isinstance(self.payload, (os.PathLike, str)):
            self.payload = list(load_payloads(self.payload))
        if isinstance(self.payload, dict):
            self.payload = [self.payload]

    def _prepare_output_path(self):
        if self.run_name is None:
            self.run_name = f"{datetime.now():%Y%m%d-%H%M}"
        if self.output_path:
            self.output_path = Path(self.output_path)
            logger.info(f"Saving results to {self.output_path}")

    async def _run(self):
        result = Result(
            responses=[],
            total_test_time=None,
            total_requests=self._n_requests * self.clients,
            clients=self.clients,
            n_requests=self._n_requests,
            output_path=self.output_path / self.run_name,
            model_id=self._endpoint.model_id,
            provider=self._endpoint.provider,
            endpoint_name=self._endpoint.endpoint_name,
            run_name=self.run_name,
            run_description=self.run_description,
        )

        if self.callbacks is not None:
            [await cb.before_run(self) for cb in self.callbacks]

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
            _, (total_test_time, start_time, end_time) = await asyncio.gather(
                self._process_results_from_q(
                    output_path=self.output_path / "responses.jsonl"
                    if self.output_path
                    else None,
                ),
                self._invoke_n_c(
                    payload=self.payload,
                    n_requests=self._n_requests,
                    clients=self.clients,
                ),
            )

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
            start_time=start_time,
            end_time=end_time,
        )

        if self.callbacks is not None:
            [await cb.after_run(result) for cb in self.callbacks]

        if result.output_path:
            result.save()

        return result


@dataclass
class Runner(_RunConfig):
    """
    A class for running and managing LLM inference tasks.

    This class provides methods for token counting,
    invocation handling, and asynchronous processing of LLM requests.

    Attributes:
        endpoint (Endpoint | dict): TODO Docstring - Here since this is the public class
        output_path (os.PathLike | str | None): TODO Docstring
        tokenizer (Tokenizer | Any | None): TODO Docstring
        n_requests (int | None): TODO Docstring
        clients (int): TODO Docstring
        payload (dict | list[dict] | os.PathLike | str | None): TODO Docstring
        run_name (str | None): TODO Docstring
        run_description (str | None): TODO Docstring
        timeout (int | float): TODO Docstring
        disable_per_client_progress_bar (bool): TODO Docstring
        disable_clients_progress_bar (bool): TODO Docstring
        callbacks (list[Callback] | None): TODO Docstring
    """

    def _prepare_run(self, **kwargs) -> _Run:
        """Create an individual run based on this runner's configurations, with optional overrides

        Args:
            **kwargs: Overrides to the Runner's default config for this specific run
        """
        return _Run(
            **{
                **{f.name: getattr(self, f.name) for f in fields(_RunConfig)},
                **{k: v for k, v in kwargs.items() if v is not None},
            }
        )

    async def run(
        self,
        payload: dict | list[dict] | os.PathLike | str | None,
        n_requests: int | None = None,
        clients: int = 1,
        output_path: os.PathLike | str | None = None,
        run_name: str | None = None,
        run_description: str | None = None,
        callbacks: list[Callback] | None = None,
    ) -> Result:
        """
        Tests the the endpoint latency and throughput for a fixed payload.

        This method tests the performance of the endpoint by sending multiple
        concurrent requests with the given payload(s). It measures the total time
        taken to complete the test, generates invocations for the given payload(s),
        and optionally saves the results and metrics.
        The arguments are optional, and if provided they take precedence over
        the values of the `Runner` class.

        Args:
            payload (Dict | Sequence[Dict] | str | UPath | None): The input payload(s)
                to generate invocations for. Can be a single dictionary, a sequence of
                dictionaries, a string (file path), or a UPath object.
            n_requests (int | None, optional): The number of invocations to generate per
                client. If None, uses the length of the payload. Defaults to None.
            clients (int, optional): The number of concurrent clients to use for
                sending requests. Defaults to 1.
            output_path (UPath | str | None, optional): The path to save responses,
                results, and test metrics. If None, results are not saved to files.
            run_name (str | None, optional): A name for this specific test run.
                Defaults to None.
            run_description (str | None, optional): A description of this test run.
                Defaults to None.

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
            payload=payload,
            n_requests=n_requests,
            clients=clients,
            output_path=output_path,
            run_name=run_name,
            run_description=run_description,
            callbacks=callbacks,
        )
        assert isinstance(run.payload, list)
        assert isinstance(run.run_name, str)
        return run._run()
