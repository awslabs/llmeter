# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across `callbacks` submodules"""

from __future__ import annotations

import json
import logging
from abc import ABC
from typing import Any, final

from upath.types import ReadablePathLike, WritablePathLike

from ..endpoints.base import InvocationResponse
from ..json_utils import (
    _resolve_class,
    llmeter_default_deserializer,
    llmeter_default_serializer,
)
from ..results import Result
from ..runner import _RunConfig
from ..utils import ensure_path

logger = logging.getLogger(__name__)


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

    # -- Serialization -----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize this Callback's configuration to a JSON-compatible dictionary.

        The returned dict includes an ``__llmeter_class__`` key with the fully-qualified
        ``module:ClassName`` path so that :func:`~llmeter.json_utils.llmeter_default_deserializer`
        (or :meth:`Callback.from_dict`) can reconstruct the correct subclass.

        Subclasses with additional state should override this method, call ``super().to_dict()``,
        and merge in their own fields.

        Returns:
            A JSON-serializable dictionary representation of this callback.
        """
        cls = type(self)
        qualified_name = f"{cls.__module__}:{cls.__name__}"
        return {"__llmeter_class__": qualified_name}

    @classmethod
    def from_dict(cls, raw: dict, **kwargs: Any) -> Callback:
        """Reconstruct a Callback from a dictionary produced by :meth:`to_dict`.

        If *raw* contains an ``__llmeter_class__`` key the referenced class is dynamically
        imported and its own ``from_dict`` is called (or its constructor, if it does not
        override ``from_dict``).  When called directly on a concrete subclass (not the
        ``Callback`` base), the class marker is optional and the subclass constructor is
        used directly.

        Args:
            raw: A plain Python dict, typically loaded from JSON.
            **kwargs: Extra keyword arguments forwarded to the subclass constructor.

        Returns:
            A Callback instance.
        """
        data = {**raw}
        qualified_name = data.pop("__llmeter_class__", None)

        if qualified_name is not None:
            target_cls = _resolve_class(qualified_name)
            # If the resolved class has its own from_dict (not the base Callback one),
            # delegate to it so subclass-specific logic runs.
            if target_cls is not cls and hasattr(target_cls, "from_dict"):
                return target_cls.from_dict(data, **kwargs)
            return target_cls(**data, **kwargs)

        # No class marker — we must be on a concrete subclass already.
        if cls is Callback:
            raise ValueError(
                "Cannot instantiate Callback base class from dict without "
                "'__llmeter_class__' key. Provide the class marker or call "
                "from_dict on a concrete subclass."
            )
        return cls(**data, **kwargs)

    def save_to_file(self, path: WritablePathLike) -> None:
        """Save this Callback to a JSON file.

        Args:
            path: (Local or Cloud) path where the callback configuration is saved.
        """
        path = ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, default=llmeter_default_serializer, indent=4)

    @staticmethod
    @final
    def load_from_file(path: ReadablePathLike) -> Callback:
        """Load (any type of) Callback from a JSON file.

        The file must contain a JSON object with an ``__llmeter_class__`` key that identifies
        the concrete Callback subclass to instantiate.

        Args:
            path: (Local or Cloud) path where the callback was saved.
        Returns:
            callback: A loaded Callback — for example a ``CostModel`` or ``MlflowCallback``.
        """
        path = ensure_path(path)
        with path.open("r") as f:
            data = json.load(f, object_hook=llmeter_default_deserializer)

        if isinstance(data, Callback):
            # object_hook already reconstructed the object
            return data

        # Fallback: the top-level dict wasn't resolved (e.g. subclass needed special handling)
        if isinstance(data, dict):
            return Callback.from_dict(data)

        raise ValueError(f"Unexpected data type loaded from {path}: {type(data)}")

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
        return Callback.load_from_file(path)
