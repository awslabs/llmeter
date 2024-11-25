import mlflow

from ..results import Result
from .base import Callback


class MlflowCallback(Callback):
    def __init__(self, step=None) -> None:
        super().__init__()
        self.step = step

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

    async def after_run(self, result: Result):
        mlflow.log_params(
            {k: getattr(result, k) for k in self.parameters_names if hasattr(result, k)}
        )
        mlflow.log_metrics(
            {
                k: v
                for k, v in result.stats.items()
                if k not in self.parameters_names
                if v is not None
            },
            step=self.step,
        )