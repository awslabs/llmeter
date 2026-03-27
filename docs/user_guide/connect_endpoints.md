# Connect to Endpoints

LLMeter provides a range of optimized `Endpoint` connectors for different types of LLM deployment, and supports 100+ more through [LiteLLM](https://docs.litellm.ai/).

## Streaming vs non-streaming endpoints

Since generating text responses from LLMs (especially long responses) can take significant time, many LLM deployment platforms and services offer **response streaming**: in which the model starts returning chunks of the response as soon as they're generated - rather than waiting for the whole thing before responding.

Response streaming can reduce perceived solution latency by allowing consumers to start reading (or processing) the response before generation is completed... But not always applicable, if some other component in the overall architecture doesn't support it.

## Supported endpoint types

The [endpoints](../reference/endpoints/index.md) section of the API reference lists the range of built-in endpoint types currently offered by LLMeter.

You can also **create your own integrations** by extending the [`Endpoint`](../reference/endpoints/base.md#llmeter.endpoints.base.Endpoint) class interface, if your target isn't already supported by the built-in endpoints or through the [LiteLLM Endpoint](../reference/endpoints/litellm.md) and [LiteLLM Python SDK](https://docs.litellm.ai/#basic-usage).

Note that [Amazon Bedrock](https://aws.amazon.com/bedrock/) supports several different APIs for accessing Foundation Models. Depending on your target API, you can use LLMeter's:

- [`bedrock`](../reference/endpoints/bedrock.md) endpoints for connecting to Bedrock's [Converse](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html) or [ConverseStream](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html) APIs
- [`bedrock_invoke`](../reference/endpoints/bedrock_invoke.md) endpoints for connecting to Bedrock's [InvokeModel](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html) or [InvokeModelWithResponseStream](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html) APIs
- [`openai`](../reference/endpoints/openai.md) endpoints for connecting to Bedrock's OpenAI-compatible [Mantle APIs](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html)
