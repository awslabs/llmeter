# Key Concepts

To break down the complex task of performance testing in a modular way, LLMeter defines the following key abstractions:

## [Endpoint](../connect_endpoints/): An instrumented LLM/API

An `Endpoint` is the Python interface through which LLMeter connects to whatever model or API you want to evaluate. It provides an `invoke()` method which calls the model, but also stores metadata like the time the request took to process and number of input/output tokens consumed.

LLMeter provides a [range of built-in Endpoint connectors](../connect_endpoints/) for different types of Cloud-deployed or local LLM, or you can also define your own custom class implementing the [Endpoint API](../../reference/endpoints/base/).

## [`Runner`](../../reference/runner/#llmeter.runner.Runner): Low-level concurrent request runner

While an individual [`InvocationResponse`](../../reference/endpoints/base/#llmeter.endpoints.base.InvocationResponse)'s latency information may be useful, performance tests will usually need to run a batch of multiple requests (perhaps in parallel) to explore how response times **vary** over different parameters and repeated runs.

The `Runner` is a low-level API to run a batch of (concurrent) requests through your `Endpoint`, calculate **summary statistics** over the results, and optionally save both the underlying invocation data and summary statistics to file.

## [Experiment](../run_experiments/): High-level test procedure

An `Experiment` is a high-level, pre-defined analysis to explore a particular aspect of latency or performance - which might run one or more Runs under the hood.

LLMeter's pre-built Experiments are designed based on evaluation best-practices and feedback from our users, but you can always build your own custom Experiments from the lower-level Runner API if needed.
