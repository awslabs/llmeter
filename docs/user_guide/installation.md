# Installation

LLMeter requires Python version 3.10 or higher.

To install the basic metering functionalities, you can install the minimal package using pip install:

```bash
pip install llmeter
```

LLMeter also offers optional extra features that require additional dependencies. Currently these extras include:

- **plotting**: Add methods to generate charts and heatmaps to summarize the results
- **openai**: Enable testing endpoints offered by OpenAI
- **litellm**: Enable testing a range of different models through [LiteLLM](https://github.com/BerriAI/litellm)

You can install one or more of these extra options using pip:

```bash
pip install llmeter[plotting, openai, litellm]
```

## *Where* to install and use LLMeter

LLMeter measures **end-to-end latencies** and uses pure-Python ([asyncio](https://docs.python.org/3/library/asyncio.html)-based) concurrency to parallelize requests.

✅ Remember that **network latency** from the environment where you run LLMeter to the LLM under test will be included in results.

> You may want to run LLMeter on the Cloud if that's where your application and LLM will be deployed.

✅ Check your **network bandwidth and compute capacity** are sufficient to avoid bottlenecking your highest-concurrency tests.

> If hosting LLMeter on [burstable](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-performance-instances.html) Cloud instance types (e.g. AWS `t3`, `t4g`, etc), be aware that sustained network and compute limits are lower than short-term burst resources.

Because generative AI inference is a compute-intensive task, load testing LLMs has not typically required very high-powered (compute, network) resources from the client side. This led us to a deliberate decision to keep LLMeter simple to install and use by avoiding more complex distributed computing frameworks, in favour of plain Python+asyncio. If you have a use-case with such a fast LLM or large request volume that this approach is limiting to you - please share your feedback!
