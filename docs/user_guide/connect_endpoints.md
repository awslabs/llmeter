# Connect to Endpoints

LLMeter provides a range of optimized `Endpoint` connectors for different types of LLM deployment, and supports 100+ more through [LiteLLM](https://docs.litellm.ai/).

## Streaming vs non-streaming endpoints

Since generating text responses from LLMs (especially long responses) can take significant time, many LLM deployment platforms and services offer **response streaming**: in which the model starts returning chunks of the response as soon as they're generated - rather than waiting for the whole thing before responding.

Response streaming can reduce perceived solution latency by allowing consumers to start reading (or processing) the response before generation is completed... But not always applicable, if some other component in the overall architecture doesn't support it.

## Supported endpoint types

The [endpoints](../reference/endpoints/index.md) section of the API reference lists the range of built-in endpoint types currently offered by LLMeter.

You can also **create your own integrations** by extending the [`Endpoint`](../reference/endpoints/base.md#llmeter.endpoints.base.Endpoint) class interface, if your target isn't already supported by the built-in endpoints or through the [LiteLLM Endpoint](../reference/endpoints/litellm.md) and [LiteLLM Python SDK](https://docs.litellm.ai/#basic-usage).

### Custom endpoint example

To create a custom endpoint, implement three methods:

- **`invoke(payload)`** — call the API and pass the raw response to `parse_response()`
- **`parse_response(raw_response, start_t)`** — extract text, token counts, and metadata into an `InvocationResponse`
- **`prepare_payload(payload, **kwargs)`** *(optional)* — transform the caller's payload before invocation (merge kwargs, inject model ID, etc.)

The base class automatically wraps `invoke` with error handling, timing, and metadata back-fill, so your implementation only needs the happy path:

```python
from llmeter.endpoints.base import Endpoint, InvocationResponse

class MyEndpoint(Endpoint):
    def __init__(self, model_id: str, api_key: str):
        super().__init__(
            endpoint_name="my-service",
            model_id=model_id,
            provider="my-provider",
        )
        self._api_key = api_key

    def invoke(self, payload: dict) -> InvocationResponse:
        raw_response = my_api_call(payload, api_key=self._api_key)
        return self.parse_response(raw_response, self._start_t)

    def parse_response(self, raw_response, start_t: float) -> InvocationResponse:
        return InvocationResponse(
            response_text=raw_response["text"],
            num_tokens_input=raw_response.get("input_tokens"),
            num_tokens_output=raw_response.get("output_tokens"),
        )

    def prepare_payload(self, payload, **kwargs):
        return {**payload, **kwargs, "model": self.model_id}

    @staticmethod
    def create_payload(user_message: str, max_tokens: int = 256) -> dict:
        return {"prompt": user_message, "max_tokens": max_tokens}
```

You don't need to handle errors, set `time_to_last_token`, `input_payload`, `input_prompt`, or `id` — the base class does all of that. If your `invoke` or `parse_response` raises an exception, it's caught and converted to an error `InvocationResponse` with the prepared payload attached.

Note that [Amazon Bedrock](https://aws.amazon.com/bedrock/) supports several different APIs for accessing Foundation Models. Depending on your target API, you can use LLMeter's:

- [`bedrock`](../reference/endpoints/bedrock.md) endpoints for connecting to Bedrock's [Converse](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html) or [ConverseStream](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html) APIs
- [`bedrock_invoke`](../reference/endpoints/bedrock_invoke.md) endpoints for connecting to Bedrock's [InvokeModel](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html) or [InvokeModelWithResponseStream](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html) APIs
- [`openai`](../reference/endpoints/openai.md) endpoints for connecting to [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat)-compatible APIs, including Chat Completions API endpoints on [Amazon Bedrock Mantle](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html)
- [`openai_response`](../reference/endpoints/openai_response.md) endpoints for connecting to [OpenAI Responses](https://platform.openai.com/docs/api-reference/responses)-compatible APIs, including Responses API endpoints on [Amazon Bedrock Mantle](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html). Responses is a newer OpenAI API, with additional features including structured outputs, instructions-based system prompts, and improved multi-turn conversation handling

## Multi-modal content

Yes, LLMeter supports sending requests with multi-modal content - for profiling use-cases like analysis of documents, images, audio and video, as well as text.

## Generating request payloads

Each Endpoint class offers a `create_payload()` function to help you build test request payloads, in case you're not sure of the underlying JSON format for your target LLM/API.

For help building **multi-modal** requests, we recommend to install the optional `multimodal` extra:

```terminal
pip install 'llmeter[multimodal]'
```

Or with uv:

```terminal
uv pip install 'llmeter[multimodal]'
```

This installs the `puremagic` library for detecting media file formats from magic bytes in their content. Without it, format detection falls back to file extensions and the `.from_bytes(...)` functions mentioned below won't work.

The particular API of `create_payload()` may vary between providers, but should be generally similar. Here are some examples with [BedrockConverse.create_payload](../reference/endpoints/bedrock/#llmeter.endpoints.bedrock.BedrockConverse):

```python
from llmeter.endpoints import BedrockConverse
from llmeter.prompt_utils import DocumentContent, ImageContent, VideoContent

# Single image from file
payload = BedrockConverse.create_payload(
    user_message=[
        "What is in this image?",
        ImageContent.from_path("photo.jpg"),
    ],
    max_tokens=256
)

# Multiple images
payload = BedrockConverse.create_payload(
    user_message=[
        "Compare these images:",
        ImageContent.from_path("image1.jpg"),
        ImageContent.from_path("image2.jpg"),
    ],
    max_tokens=512
)

# Image from bytes (requires puremagic for format detection)
with open("photo.jpg", "rb") as f:
    image_bytes = f.read()

payload = BedrockConverse.create_payload(
    user_message=[
        "What is in this image?",
        ImageContent.from_bytes(image_bytes),
    ],
    max_tokens=256
)

# Mixed content types
payload = BedrockConverse.create_payload(
    user_message=[
        "Analyze this presentation and supporting materials",
        DocumentContent.from_path("slides.pdf"),
        ImageContent.from_path("chart.png"),
    ],
    max_tokens=1024
)

# Video analysis
payload = BedrockConverse.create_payload(
    user_message=[
        "Describe what happens in this video",
        VideoContent.from_path("clip.mp4"),
    ],
    max_tokens=1024
)
```

Actual media format support varies by target model and API provider. LLMeter's payload building helpers attempt to automatically detect content of the following types, and represent them according to the requirements of your target Endpoint:
- **Images**: JPEG, PNG, GIF, WebP
- **Documents**: PDF
- **Videos**: MP4, MOV, AVI
- **Audio**: MP3, WAV, OGG

### ⚠️ Security warning: Format detection is NOT input validation

**IMPORTANT**: The content-based format detection in this library is for testing and development convenience only and **should NOT be used** for with untrusted files without proper validation: Including for example files uploaded by end users; files from untrusted URLs, email attachments, or third-party systems.

LLMeter's multimodal prompt content utilities **DO:**

- Detect likely file format from magic bytes (puremagic) or extension (mimetypes)
- Read binary content from files
- Package content for API endpoints
- Provide type checking (bytes vs strings)

They **DO NOT:**

- ❌ Validate file content safety or integrity
- ❌ Scan for malicious content or malware
- ❌ Sanitize or clean file data
- ❌ Protect against malformed or exploited files
- ❌ Guarantee format correctness beyond detection heuristics
- ❌ Validate file size or prevent memory exhaustion
- ❌ Check for embedded scripts or exploits
- ❌ Verify file authenticity or source

#### Recommended security practices for untrusted files

When working with untrusted files (user uploads, external sources, etc.), you MUST implement proper security measures:

1. **Validate file sources**: Only accept files from trusted, authenticated sources
2. **Scan for malware**: Use antivirus/malware scanning (e.g., ClamAV) before processing
3. **Validate file integrity**: Verify checksums, digital signatures, or other integrity mechanisms
4. **Sanitize content**: Use specialized libraries to validate and sanitize file content:
   - Images: Re-encode with PIL/Pillow to strip metadata and validate structure
   - PDFs: Use PDF sanitization libraries to remove scripts and validate structure
   - Videos: Re-encode with ffmpeg to validate and sanitize
5. **Limit file sizes**: Enforce maximum file size limits before reading into memory
6. **Sandbox processing**: Process untrusted files in isolated environments (containers, VMs)
7. **Validate API responses**: Check that API endpoints successfully processed the content
8. **Implement rate limiting**: Prevent abuse through excessive file uploads
9. **Log and monitor**: Track file processing for security auditing
