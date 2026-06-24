# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across `callbacks` submodules"""

from __future__ import annotations

import json
from abc import ABC
from typing import final

from upath.types import ReadablePathLike, WritablePathLike

from ..endpoints.base import InvocationResponse
from ..json_utils import llmeter_default_serializer
from ..results import Result
from ..runner import _RunConfig
from ..serialization import Serializable, dump_object, load_object
from ..utils import ensure_path


class Callback(Serializable, ABC):
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

    def save_to_file(self, path: WritablePathLike) -> None:
        """Save this Callback to a JSON file.

        Uses the ``__getstate__`` protocol. Override ``__getstate__`` (not this method)
        if custom serialization is needed.

        Args:
            path: (Local or Cloud) path where the callback will be saved.
        """
        path = ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dump_object(self)
        with path.open("w") as f:
            json.dump(data, f, indent=4, default=llmeter_default_serializer)

    @staticmethod
    @final
    def load_from_file(path: ReadablePathLike) -> Callback:
        """Load (any type of) Callback from a JSON file.

        Detects the callback type from the ``__llmeter_class__`` field and reconstructs it.

        Args:
            path: (Local or Cloud) path where the callback was saved.
        Returns:
            callback: A loaded Callback instance.
        """
        path = ensure_path(path)
        with path.open("r") as f:
            data = json.load(f)
        return load_object(data)
