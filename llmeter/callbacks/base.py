# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes used across `callbacks` submodules"""

from __future__ import annotations

import importlib
import json
import logging
from abc import ABC
from typing import Any, final

from upath.types import ReadablePathLike, WritablePathLike

from ..endpoints.base import InvocationResponse
from ..json_utils import llmeter_default_serializer
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
    `before_invoke`, `after_run`, etc). Callbacks support serializing their configuration via
    ``to_dict()`` / ``from_dict()`` (and the convenience wrappers ``save_to_file()`` /
    ``load_from_file()``).

    Serialization uses a ``_callback_type`` marker (``"module:ClassName"``) so that
    ``Callback.from_dict()`` can dynamically import and reconstruct the correct subclass
    without a hardcoded registry. This means third-party callbacks round-trip through JSON
    automatically, as long as the defining module is importable at load time.

    Subclasses with complex nested state (like ``CostModel``) can override ``to_dict()`` and
    ``from_dict()`` while preserving the type marker by calling ``super()``.
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
        """Serialize this callback's configuration to a JSON-safe dict.

        The returned dict includes a ``_callback_type`` key with the fully-qualified
        class path (``"module:ClassName"``), enabling ``Callback.from_dict`` to
        reconstruct the correct subclass without a hardcoded registry.

        By default, all public (non-underscore-prefixed) instance attributes are
        included. Subclasses with richer state should override this method and call
        ``super().to_dict()`` to preserve the type marker.

        Returns:
            dict: A JSON-serializable dictionary representation of this callback.

        Example::

            >>> from llmeter.callbacks import CostModel
            >>> from llmeter.callbacks.cost.dimensions import InputTokens
            >>> model = CostModel(request_dims=[InputTokens(price_per_million=3.0)])
            >>> d = model.to_dict()
            >>> d["_callback_type"]
            'llmeter.callbacks.cost.model:CostModel'
        """
        cls = self.__class__
        data: dict[str, Any] = {
            "_callback_type": f"{cls.__module__}:{cls.__qualname__}",
        }
        data.update({k: v for k, v in vars(self).items() if not k.startswith("_")})
        return data

    @classmethod
    def from_dict(cls, raw: dict, **kwargs: Any) -> Callback:
        """Reconstruct a Callback from a dict produced by ``to_dict()``.

        Uses the ``_callback_type`` field to dynamically import and instantiate
        the correct subclass. If called on a concrete subclass (e.g.
        ``CostModel.from_dict(...)``), the ``_callback_type`` is still respected
        so that the dict always controls which class is created.

        Args:
            raw: A dictionary previously produced by ``to_dict()`` (or loaded from
                JSON). Must contain a ``_callback_type`` key.
            **kwargs: Extra keyword arguments forwarded to the resolved class
                constructor (or its own ``from_dict`` if it overrides this method).

        Returns:
            Callback: An instance of the appropriate Callback subclass.

        Raises:
            ValueError: If ``_callback_type`` is missing from *raw*.
            ImportError: If the module referenced by ``_callback_type`` cannot be
                imported.
            AttributeError: If the class name cannot be found in the referenced
                module.

        Example::

            >>> from llmeter.callbacks.base import Callback
            >>> d = {
            ...     "_callback_type": "llmeter.callbacks.mlflow:MlflowCallback",
            ...     "step": 1,
            ...     "nested": False,
            ... }
            >>> cb = Callback.from_dict(d)  # returns an MlflowCallback instance
        """
        raw = dict(raw)  # shallow copy — don't mutate caller's dict
        callback_type = raw.pop("_callback_type", None)
        if callback_type is None:
            raise ValueError(
                "Cannot deserialize Callback: dict is missing '_callback_type' key. "
                f"Got keys: {list(raw.keys())}"
            )

        module_path, class_name = callback_type.rsplit(":", 1)
        module = importlib.import_module(module_path)
        callback_cls = getattr(module, class_name)

        # If the resolved class has its own from_dict (e.g. CostModel), delegate to it
        # so that subclass-specific deserialization logic is honoured.
        if callback_cls is not cls and "from_dict" in callback_cls.__dict__:
            # Re-inject _callback_type so the subclass from_dict can pop it if needed
            return callback_cls.from_dict(raw, **kwargs)

        # Remove any keys the constructor doesn't expect (e.g. _type from JSONableBase)
        raw.pop("_type", None)
        return callback_cls(**raw, **kwargs)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize this callback to a JSON string.

        Args:
            **kwargs: Extra keyword arguments forwarded to ``json.dumps``
                (e.g. ``indent``).

        Returns:
            str: JSON representation of this callback.
        """
        kwargs.setdefault("default", llmeter_default_serializer)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, json_string: str, **kwargs: Any) -> Callback:
        """Reconstruct a Callback from a JSON string produced by ``to_json()``.

        Args:
            json_string: A valid JSON string.
            **kwargs: Extra keyword arguments forwarded to ``from_dict``.

        Returns:
            Callback: An instance of the appropriate Callback subclass.
        """
        return cls.from_dict(json.loads(json_string), **kwargs)

    def save_to_file(self, path: WritablePathLike) -> None:
        """Save this Callback's configuration to a JSON file.

        The file can be loaded back with ``Callback.load_from_file(path)``.

        Args:
            path: (Local or Cloud) path where the callback should be saved.
        """
        path = ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(self.to_json(indent=4))

    @staticmethod
    @final
    def load_from_file(path: ReadablePathLike) -> Callback:
        """Load (any type of) Callback from a JSON file.

        The ``_callback_type`` field inside the file determines which subclass is
        instantiated, so callers don't need to know the concrete type in advance.

        Args:
            path: (Local or Cloud) path to a JSON file previously created by
                ``save_to_file()``.

        Returns:
            Callback: The deserialized callback instance.
        """
        path = ensure_path(path)
        with path.open("r") as f:
            return Callback.from_dict(json.load(f))
