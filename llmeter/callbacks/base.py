"""Base class for extending LLMeter with callback functions"""

from __future__ import annotations
from abc import ABC, abstractmethod
import os
from typing import final, TYPE_CHECKING

from ..endpoints.base import InvocationResponse
from ..results import Result

if TYPE_CHECKING:
    from ..runner import Runner


class Callback(ABC):
    """Base class for a callback in LLMeter

    Callbacks support extending LLMeter functionality by running additional code at defined points
    in the test Run lifecycle: For example, logging experiments to MLFlow, or estimating costs
    associated with test runs or individual model invocations.

    A Callback object may implement multiple of the defined LLMeter lifecycle hooks (such as
    `before_invoke`, `after_run`, etc). Callbacks must support serializing their configuration to
    a file (by implementing `save_to_file`), and loading back (via `load_from_file`).
    """

    async def before_invoke(self, payload: dict) -> None:
        """Lifecycle hook run before every `Endpoint.invoke()` request in a Run.

        Args:
            payload: The payload to be sent to the endpoint.
        Returns:
            None: If you'd like to modify the request `payload`, edit the dictionary in-place.
        """
        pass

    async def after_invoke(self, response: InvocationResponse) -> None:
        """Lifecycle hook run after every `Endpoint.invoke()` request in a Run.

        Args:
            response: The InvocationResponse (already annotated with initial information e.g.
                timing and token counts)
        Returns:
            None: If you'd like to add information to the `response` logged in the Run, modify it
                in-place.
        """
        pass

    async def before_run(self, runner: Runner) -> None:
        """Lifecycle hook run at the start of each `Runner.run()`

        This function will be called after the initial Runner configuration is prepared, and before
        creating clients or starting to send requests.

        Args:
            runner: The configured `Runner` object that will be used to run the test.
        Returns:
            None: If you'd like to modify the `runner`'s configuration, edit the argument in-place.
        """
        pass

    async def after_run(self, result: Result) -> None:
        """Lifecycle hook run at the end of each `Runner.run()`

        Args:
            result: The Result of the overall run (including all individual model invocations)
        Returns:
            None: If you'd like to modify the run `result`, edit the argument in-place.
        """
        pass

    @abstractmethod
    def save_to_file(self, path: os.PathLike | str) -> None:
        """Save this Callback to file

        Individual Callbacks implement this method to save their configuration to a file that will
        be re-loadable with the equivalent `_load_from_file()` method.

        Args:
            path: (Local or Cloud) path where the callback is saved
        """
        pass

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
        # check if it's one of built-in modules
        # use the _load_from_file method to load the configuration
        pass

    @classmethod
    @abstractmethod
    def _load_from_file(cls, path: os.PathLike | str) -> Callback:
        """Load this Callback from file

        Individual Callbacks implement this method to define how they can be loaded from files
        created by the equivalent `save_to_file()` method.

        Args:
            path: (Local or Cloud) path where the callback is saved
        Returns:
            callback: The loaded Callback object
        """
        pass
