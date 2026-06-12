# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across `callbacks` submodules"""

from __future__ import annotations

from abc import ABC
from typing import final

from upath.types import ReadablePathLike, WritablePathLike

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

    Best practices for implementing callbacks:

    - **Serializability**: The Runner calls ``dataclasses.asdict()`` on its run configuration
      (which includes the callbacks list) when saving ``run_config.json``. This internally performs
      a ``copy.deepcopy()`` on all field values. If your callback uses threading primitives (locks,
      events, threads) or other non-picklable objects, you **must** implement ``__getstate__`` and
      ``__setstate__`` to exclude them from serialization and reinitialize them on restore. Example::

          def __getstate__(self):
              return {"my_config": self.my_config}

          def __setstate__(self, state):
              self.my_config = state["my_config"]
              self._lock = threading.Lock()  # Reinitialize non-picklable state

    - **Reusability**: A callback instance may be reused across multiple ``Runner.run()`` calls.
      Reset any accumulated state in ``before_run()`` so each run starts fresh.

    - **Contributing stats**: Use ``result._update_contributed_stats(dict)`` in ``after_run()`` to
      add custom metrics to ``result.stats``. These will be persisted in ``stats.json`` and survive
      save/load round-trips automatically.

    - **Thread safety**: If your callback spawns background threads (e.g. for monitoring), use
      daemon threads and ensure they are joined in ``after_run()`` to avoid resource leaks.
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

    def live_stats(self) -> dict[str, float | int] | None:
        """Optional hook returning live metrics for the progress display.

        If implemented, the Runner will call this method on each display refresh cycle
        and merge the returned values into the live stats table. This allows callbacks
        to surface real-time information (e.g. CPU usage, memory) during a run.

        Returns:
            A dict of ``{display_key: numeric_value}`` to show in the live display,
            or ``None``/empty dict if nothing to show. Keys should use short, descriptive
            names (they appear as-is in the progress table).
        """
        return None

    def save_to_file(self, path: WritablePathLike) -> None:
        """Save this Callback to file

        Individual Callbacks implement this method to save their configuration to a file that will
        be re-loadable with the equivalent `_load_from_file()` method.

        Args:
            path: (Local or Cloud) path where the callback is saved
        """
        raise NotImplementedError("TODO: Callback.save_to_file is not yet implemented!")

    @staticmethod
    @final
    def load_from_file(path: ReadablePathLike) -> Callback:
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
    def _load_from_file(cls, path: ReadablePathLike) -> Callback:
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
