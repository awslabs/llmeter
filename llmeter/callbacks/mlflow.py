import mlflow

from ..results import Result
from .base import Callback


class MlflowCallback(Callback):
    def __init__(self, step=None, nested=False) -> None:
        super().__init__()
        self.step = step
        self.nested = nested

    @classmethod
    async def _load_from_file(cls, path: str):
        # load the configuration from the file
        pass

    def save_to_file(self) -> str | None:
        # save the configuration to the file
        pass

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
        with mlflow.start_run(nested=True):
            await self._log_llmeter_run(result)

    async def after_run(self, result: Result):
        if self.nested:
            await self._log_nested_run(result)
            return
        await self._log_llmeter_run(result)
