# Extend LLMeter with Callbacks

There's a lot more to performance testing beyond calling your LLM lots of times and capturing the speed of the responses... But to avoid bloating the performance-sensitive core of the library, LLMeter handles many of these more advanced or optional concerns via a **callback mechanism**.

See the dedicated sections in this guide on how [built-in callbacks](../../reference/callbacks/) and standard patterns can help you to, for example:

- [Avoid prompt caching](cache_buster.md) on your target Endpoint
- [Model and compare the costs](cost.md) of different Endpoints
- [Track your experiments](mlflow.md) with MLflow


## Build your own custom callbacks

You can implement custom callbacks to add your own functionality to LLMeter, by extending from the [`Callback`](../../reference/callbacks/base/#llmeter.callbacks.base.Callback) base class and implementing one or multiple of its standard methods:

- [`before_run`](../../reference/callbacks/base/#llmeter.callbacks.base.Callback.before_run) is called before each test Run starts, and has the opportunity to inspect or modify the Run configuration.
- [`before_invoke`](../../reference/callbacks/base/#llmeter.callbacks.base.Callback.before_invoke) is called before each individual model invocation, and can inspect or modify the request payload.
- [`after_invoke`](../../reference/callbacks/base/#llmeter.callbacks.base.Callback.after_invoke) is called after each model invocation, and can inspect or modify the `InvocationResponse`.
- [`after_run`](../../reference/callbacks/base/#llmeter.callbacks.base.Callback.after_run) is called after each test Run completes, and can inspect or modify the Run `Result`.

Callbacks are processed outside of the timing of invocations, and `before_run` and `after_run` callbacks are processed outside of the timing of the overall Run. However, it's important to remember that slow callbacks could still affect the maximum volume of traffic that LLMeter is able to drive - and that the backlog of `after_invoke` callbacks must be cleared before a Run is considered ended.

You can specify one or multiple callbacks when setting up your Runner, as below:

```python
runner = Runner(
    endpoint,
    output_path=f"outputs/{endpoint.model_id}",
    callbacks=[MlflowCallback(), MyCoolCallback()],
)
results = await runner.run(
    payload=sample_payloads,
    n_requests=10,
    clients=10,
)
```

Each callback will be processed in the **same order** as you provide them to the runner. This is important to remember and configure properly, if you're stacking multiple callbacks that access the same data (for example - transforming an invocation response *then* logging/exporting it somewhere, both using `after_invoke`).

### Storing extra data on responses and results

If your callback needs to attach **extra fields** on your `InvocationResponse`s (for example, a computed cost or a custom label), store them in the response's `annotations` dictionary:

```python
async def after_invoke(self, response):
    response.annotations["my_metric"] = compute_something(response)
```

Any extra fields added directly to the `InvocationResponse` object itself will *not* be preserved when the responses are saved to file. For legacy compatability, any unrecognized fields found in older responses.jsonl files are *currently* collected back to `annotations`, but this behaviour may be dropped in future.

Likewise for extra **run-level** data, `after_run` can add numeric statistics to the `Result` via [`_update_contributed_stats`](../../reference/results/#llmeter.results.Result), which are persisted alongside the run's other stats. (This is how the built-in [cost modelling callback](cost.md) records run-level cost metrics.)
