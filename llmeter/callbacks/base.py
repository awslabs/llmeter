# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from abc import ABC
from typing import final

from ..endpoints.base import InvocationResponse
from ..results import Result
from ..runner import _RunConfig


class Callback(ABC):
    """Base class for a callback in LLMeter

    Callbacks support extending LLMeter functionality by running additional code at defined points
    in the test Run lifecycle: For example, logging experiments to MLFlow, or estimating costs
    associated with test runs or individual model invocations.

    A Callback object may implement multiple of the defined lifecycle hooks (such as
    `before_invoke`, `after_run`, etc). Callbacks must support serializing their configuration to
    a file (by implementing `save_to_file`), and loading back (via `load_from_file`).
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
                in-place.
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

    def save_to_file(self, path: os.PathLike | str) -> None:
        """Save this Callback to file

        Individual Callbacks implement this method to save their configuration to a file that will
        be re-loadable with the equivalent `_load_from_file()` method.

        Args:
            path: (Local or Cloud) path where the callback is saved
        """
        raise NotImplementedError("TODO: Callback.save_to_file is not yet implemented!")

    @staticmethod
    @final
    def load_from_file(path: os.PathLike | str) -> Callback:
        """Load (any type of) Callback from file

        `Callback.load_from_file()` attempts to detect the type of Callback saved in a given file,
        and use the relevant implementation's `_load_from_file` method to load it.

        Args:
            path: (Local or Cloud) path where the callback is saved
        Returns:
            callback: A loaded Callback - for example an `MlflowCallback`.
        """
        raise NotImplementedError(
            "TODO: Callback.load_from_file is not yet implemented!"
        )

    @classmethod
    def _load_from_file(cls, path: os.PathLike | str) -> Callback:
        """Load this Callback from file

        Individual Callbacks implement this method to define how they can be loaded from files
        created by the equivalent `save_to_file()` method.

        Args:
            path: (Local or Cloud) path where the callback is saved
        Returns:
            callback: The loaded Callback object
        """
        raise NotImplementedError(
            "TODO: Callback._load_from_file is not yet implemented!"
        )
