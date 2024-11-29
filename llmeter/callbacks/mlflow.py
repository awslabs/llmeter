# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from ..results import Result
from ..utils import DeferredError
from .base import Callback

try:
    import mlflow
except ImportError:
    mlflow = DeferredError(
        "Please install mlflow (or mlflow-skinny) to use the MlflowCallback"
    )


class MlflowCallback(Callback):
    """LLMeter callback to log test run parameters and metrics to an MLFlow tracking server

    TODO: Short code example showing usage in Runner with the MLFlow `with` context?
    """

    step: int | None
    nested: bool
    parameters_names = [
        "total_requests",
        "clients",
        "n_requests",
        "model_id",
        "output_path",
        "endpoint_name",
        "provider",
        "run_name",
        "run_description",
    ]

    def __init__(self, step: int | None = None, nested: bool = False) -> None:
        """Create an MlflowCallback

        Args:
            step: Passed through to `mlflow.log_metrics`
            nested: By default (`False`) the externally-provided mlflow run context will be used.
                Set `True` to log under a new nested run with the name of the LLMeter
                `Result.run_name`.
        """
        super().__init__()
        self.step = step
        self.nested = nested
        # Check MLFlow is installed by polling any attribute on the module to trigger DeferredError
        mlflow.__version__

    @classmethod
    async def _load_from_file(cls, path: str):
        raise NotImplementedError(
            "TODO: MlflowCallback does not yet support loading from file"
        )

    def save_to_file(self) -> str | None:
        raise NotImplementedError(
            "TODO: MlflowCallback does not yet support saving to file"
        )

    async def _log_llmeter_run(self, result: Result):
        mlflow.log_params(
            {
                k: getattr(result, k)
                for k in self.parameters_names
                if hasattr(result, k)
            },
            synchronous=False,
        )
        mlflow.log_metrics(
            {
                k: v
                for k, v in result.stats.items()
                if k not in self.parameters_names
                if v is not None
            },
            step=self.step,
            synchronous=False,
        )

    async def _log_nested_run(self, result: Result):
        with mlflow.start_run(run_name=result.run_name, nested=True):
            await self._log_llmeter_run(result)

    async def after_run(self, result: Result):
        if self.nested:
            await self._log_nested_run(result)
            return
        await self._log_llmeter_run(result)
