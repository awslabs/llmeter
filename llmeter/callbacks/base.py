# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across `callbacks` submodules"""

from __future__ import annotations

from abc import ABC

from ..endpoints.base import InvocationResponse
from ..results import Result
from ..runner import _RunConfig
from ..serialization import Serializable


class Callback(Serializable, ABC):
    """Base class for a callback in LLMeter

    Callbacks support extending LLMeter functionality by running additional code at defined points
    in the test Run lifecycle: For example, logging experiments to MLFlow, or estimating costs
    associated with test runs or individual model invocations.

    A Callback object may implement multiple of the defined lifecycle hooks (such as
    `before_invoke`, `after_run`, etc) - which have no-op implementations by default. Serialization
    to/from file is inherited from `llmeter.serialization.Serializable`, and is necessary so your
    callback(s) can be saved to file (and restored) as part of a Run configuration. Any custom
    callback class that is *not* LLMeter-serializable will raise a `TypeError` when the run config
    it belongs to is saved.
    """

    async def before_invoke(self, payload: dict) -> None:
        """Lifecycle hook called before every `Endpoint.invoke()` request in a Run.

        Args:
            payload: The payload to be sent to the endpoint.
        Returns:
            None: If you'd like to modify the request `payload`, edit the dictionary in-place.
        """
        pass

    async def after_invoke(self, response: InvocationResponse) -> None:
        """Lifecycle hook called after every `Endpoint.invoke()` request in a Run.

        Args:
            response: The InvocationResponse (already annotated with initial information e.g.
                timing and token counts)
        Returns:
            None: If you'd like to add information to the `response` logged in the Run, modify it
                in-place. To attach **extra custom fields** that you want preserved when responses
                are saved to file, store them in the `response.annotations` map.
        """
        pass

    async def before_run(self, run_config: _RunConfig) -> None:
        """Lifecycle hook called at the start of each `Runner.run()`

        This function will be called after the initial Runner configuration is prepared, and before
        creating clients or starting to send requests.

        Args:
            run_config: The configuration that will be used to run the test.
        Returns:
            None: If you'd like to modify the current run's configuration, edit it in-place.
        """
        pass

    async def after_run(self, result: Result) -> None:
        """Lifecycle hook called at the end of each `Runner.run()`

        Args:
            result: The Result of the overall run (including all individual model invocations)
        Returns:
            None: If you'd like to modify the run `result`, edit the argument in-place.
        """
        pass
