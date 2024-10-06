<div align="center">
<img alt="LLMeter (Logo)" src="https://github.com/awslabs/llmeter/blob/main/docs/llmeter-logotype-192px.png?raw=true" height="96px" width="396px"/>

**Measuring large language models latency and throughput**

[![Latest Version](https://img.shields.io/pypi/v/llmeter.svg)](https://pypi.python.org/pypi/llmeter)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/llmeter)](https://pypi.python.org/pypi/llmeter)
[![Code Style: Ruff](https://img.shields.io/badge/code_style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

LLMeter is a pure-python library for simple latency and throughput testing of large language models (LLMs). It's designed to be lightweight to install; straightforward to run standard tests; and versatile to integrate - whether in notebooks, CI/CD, or other workflows.

## 🛠️ Installation

LLMeter requires `python>=3.10`, please make sure your current version of python is compatible.

To install the basic metering functionalities, you can install the minimum package using pip install:

```terminal
pip install llmeter
```

LLMeter also offers extra features that require additional dependencies. Currently these extras include:

- **plotting**: Add methods to generate charts and heatmaps to summarize the results
- **openai**: Enable testing endpoints offered by OpenAI
- **litellm**: Enable testing a range of different models through [LiteLLM](https://github.com/BerriAI/litellm)

You can install one or more of these extra options using pip:

```terminal
pip install llmeter[plotting, openai, litellm]
```

## 🚀 Quick-start

At a high level, you'll start by configuring an LLMeter "Endpoint" for whatever type of LLM you're connecting to:

```python
# For example with Amazon Bedrock...
from llmeter.endpoints import BedrockConverse
endpoint = BedrockConverse(model_id="...")

# ...or OpenAI...
from llmeter.endpoints import OpenAIEndpoint
endpoint = OpenAIEndpoint(model_id="...", api_key="...")

# ...or via LiteLLM...
from llmeter.endpoints import LiteLLM
endpoint = LiteLLM("{provider}/{model_id}")

# ...and so on
```

You can then run the high-level "experiments" offered by LLMeter:

```python
# For example a heatmap of latency by input & output token count:
from llmeter.experiments import LatencyHeatmap
latency_heatmap = LatencyHeatmap(
    endpoint=endpoint,
    clients=10,
    source_file="examples/MaryShelleyFrankenstein.txt",
    ...
)
heatmap_results = await latency_heatmap.run()
latency_heatmap.plot_heatmap()

# ...Or testing how throughput varies with concurrent request count:
from llmeter.experiments import LoadTest
sweep_test = LoadTest(
    endpoint=endpoint,
    payload={...},
    sequence_of_clients=[1, 5, 20, 50, 100, 500],
)
sweep_results = await sweep_test.run()
sweep_test.plot_sweep_results()
```

Alternatively, you can use the low-level `llmeter.runner.Runner` class to run and analyze request
batches - and build your own custom experiments.

For more details, check out our selection of end-to-end code examples in the [examples](https://github.com/awslabs/llmeter/tree/main/examples) folder!

## Security

See [CONTRIBUTING](https://github.com/awslabs/llmeter/tree/main/CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
