# Track Experiments with MLflow

When running many different tests or experiments, you'll likely be interested to set up a tool in which to compare, sort, search, plot, and log the results.

[MLflow](https://mlflow.org/) is one tool already used by many data science teams for experiment tracking in general, and AWS offers [fully-managed, serverless MLflow on Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html).

Wherever your MLflow tracking server is hosted, you can use LLMeter's built-in [`MlflowCallback`](../../reference/callbacks/mlflow/#llmeter.callbacks.mlflow.MlflowCallback) to log your Run input parameters and output metrics to MLflow experiment runs.

Just set up MLflow as usual in your script, before running your LLMeter `Runner`. You can find client setup guidance for MLflow on Amazon SageMaker [in its developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html).

!!! tip "Logging callback-contributed stats"
    If you're using some other callback that contributes statistics to the Run Result via an `after_run` hook, and you want those statistics to be reflected in your MLflow experiments, remember to include the `MlflowCallback` **after** your other stat-contributing callbacks - not before!
