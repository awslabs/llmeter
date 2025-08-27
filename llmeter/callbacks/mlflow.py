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
    """LLMeter callback to log test run parameters and metrics to an MLFlow tracking server.
    
    This callback integrates with MLflow to track and log parameters and metrics from LLMeter test runs.
    It can operate either in the current MLflow run context or create nested runs for each test.
    
    Example:
        ```python
        import mlflow
        from llmeter.callbacks import MlflowCallback
        
        with mlflow.start_run():
            runner = Runner(
                endpoint=endpoint,
                callbacks=[MlflowCallback()]
            )
            results = await runner.run()
        ```
    
    Attributes:
        step (int | None): Step number for MLflow metrics logging
        nested (bool): Whether to create nested runs for each test
        parameters_names (list): List of parameter names to log to MLflow
    """
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
            step (int | None, optional): Passed through to `mlflow.log_metrics`
            nested (bool, optional): By default (`False`) the externally-provided mlflow run context will be used.
                Set `True` to log under a new nested run with the name of the LLMeter
                `Result.run_name`.

        Raises:
            ImportError: If MLflow is not installed                
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
        """Log parameters and metrics from an LLMeter run to MLflow.

        Args:
            result (Result): LLMeter test run result containing parameters and metrics to log
        """

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
                if isinstance(v, (int, float))
            },
            step=self.step,
            synchronous=False,
        )

    async def _log_nested_run(self, result: Result):
        """Create and log to a nested MLflow run.

        Args:
            result (Result): LLMeter test run result containing parameters and metrics to log
        """

        with mlflow.start_run(run_name=result.run_name, nested=True):
            await self._log_llmeter_run(result)

    async def after_run(self, result: Result):
        """Callback method executed after an LLMeter test run completes.

        Logs parameters and metrics to MLflow, either in the current run context
        or in a new nested run depending on the `nested` setting.

        Args:
            result (Result): LLMeter test run result containing parameters and metrics to log
        """
        if self.nested:
            await self._log_nested_run(result)
            return
        await self._log_llmeter_run(result)
