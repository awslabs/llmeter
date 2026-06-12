# Key Concepts

To break down the complex task of performance testing in a modular way, LLMeter defines the following key abstractions:

## [Endpoint](connect_endpoints.md): An instrumented LLM/API

An `Endpoint` is the Python interface through which LLMeter connects to whatever model or API you want to evaluate. It provides an `invoke()` method which calls the model, and automatically captures metadata like request timing, token counts, and error information in an [`InvocationResponse`](../reference/endpoints/base.md#llmeter.endpoints.base.InvocationResponse).

The base `Endpoint` class handles common concerns (error handling, timing, metadata) automatically via an invoke lifecycle:

1. **`prepare_payload(payload, **kwargs)`** — merges caller kwargs and injects provider-specific fields (model ID, streaming options, etc.)
2. **`invoke(payload)`** — makes the API call and delegates to `parse_response()`
3. **`parse_response(raw_response, start_t)`** — extracts text, token counts, and other metadata from the provider's raw response

Subclasses implement these three methods with their provider-specific logic. The `@llmeter_invoke` decorator wraps `invoke` with standardized error handling, timing (`time_to_last_token`), and metadata back-fill (`input_payload`, `input_prompt`, `id`), so individual endpoints don't need to duplicate that boilerplate.

LLMeter provides a [range of built-in Endpoint connectors](connect_endpoints.md) for different types of Cloud-deployed or local LLM, or you can also define your own custom integrations.

## [`Runner`](../reference/runner.md#llmeter.runner.Runner): Low-level concurrent request runner

While an individual [`InvocationResponse`](../reference/endpoints/base.md#llmeter.endpoints.base.InvocationResponse)'s latency information may be useful, performance tests will usually need to run a batch of multiple requests (perhaps in parallel) to explore how response times **vary** over different parameters and repeated runs.

The `Runner` is a low-level API to run a batch of (concurrent) requests through your `Endpoint`, calculate **summary statistics** over the results, and optionally save both the underlying invocation data and summary statistics to file.

## [Experiment](run_experiments.md): High-level test procedure

An `Experiment` is a high-level, pre-defined analysis to explore a particular aspect of latency or performance - which might run one or more Runs under the hood.

LLMeter's pre-built Experiments are designed based on evaluation best-practices and feedback from our users, but you can always build your own custom Experiments from the lower-level Runner API if needed.

## [Callbacks](../reference/callbacks/index.md): Extensibility hooks

Callbacks let you extend LLMeter's behavior at defined points in the run lifecycle. A callback can run code before/after each request, or before/after an entire run. Built-in callbacks include:

- **[`CostModel`](../reference/callbacks/cost/model.md)**: Estimate costs based on token counts, endpoint time, or custom pricing dimensions.
- **[`SystemMetricsMonitor`](../reference/callbacks/system_metrics.md)**: Monitor CPU, memory, and network I/O during runs, with live display integration and persisted statistics.
- **[`MlflowCallback`](../reference/callbacks/mlflow.md)**: Log run parameters and results to MLflow for experiment tracking.

Callbacks are passed to `Runner` or `Experiment` via the `callbacks` parameter:

```python
from llmeter.callbacks import CostModel
from llmeter.callbacks.system_metrics import SystemMetricsMonitor

runner = Runner(
    endpoint=endpoint,
    callbacks=[SystemMetricsMonitor(), CostModel(...)],
)
```

You can also create custom callbacks by subclassing `Callback` — see the [API reference](../reference/callbacks/base.md) for the lifecycle hooks and best practices.
