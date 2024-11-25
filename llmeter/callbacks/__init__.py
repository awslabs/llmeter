import importlib.util

spec = importlib.util.find_spec("mlflow")
if spec:
    from .mlflow import MlflowCallback  # noqa: F401