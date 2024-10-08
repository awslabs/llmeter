# Connect to Endpoints

LLMeter provides a range of optimized `Endpoint` connectors for different types of LLM deployment, and supports 100+ more through [LiteLLM](https://docs.litellm.ai/).

## Streaming vs non-streaming endpoints

Since generating text responses from LLMs (especially long responses) can take significant time, many LLM deployment platforms and services offer **response streaming**: in which the model starts returning chunks of the response as soon as they're generated - rather than waiting for the whole thing before responding.

Response streaming can reduce perceived solution latency by allowing consumers to start reading (or processing) the response before generation is completed... But not always applicable, if some other component in the overall architecture doesn't support it.
