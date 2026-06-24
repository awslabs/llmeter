# Avoid Prompt Caching

Many inference APIs implement some form of **implicit prompt caching**. With this feature, repeated calls whose prompt inputs are the same or share a common prefix can *re-use* part of the computation of previous ones - for faster responses.

While that's great for performance in general, it could become an issue if the cache hit rate *for your test dataset is very different* than what you expect to see in production - as your test results would no longer be representative. This is especially likely if, for example, you're running a test that will repeat a very **small dataset** (maybe even a single payload) many times.

If you're not able to generate a larger dataset, you could consider using a **cache busting callback** similar to the example below, to insert a randomised prefix before each invocation and avoid hitting your endpoint's implicit prompt cache:

```python
from uuid import uuid4
from llmeter.callbacks import Callback

class CacheBusterCallback(Callback):
    """Add a unique prefix to each request to defeat prompt caching."""

    @staticmethod
    def _generate_prefix() -> str:
        return f"[req-{uuid4()}] "

    async def before_invoke(self, payload: dict) -> None:
        tag = self._generate_prefix()

        messages = payload.get("messages", [])
        if not messages:
            return

        first_msg = messages[0]
        content = first_msg.get("content")

        # Messages API format: content is a plain string
        if isinstance(content, str):
            first_msg["content"] = tag + content
            return

        # Converse format: content is a list of blocks
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    block["text"] = tag + block["text"]
                    return

        # TODO: Other formats too?
```

This callback is not natively included in LLMeter today, because the payload manipulation logic is dependent on which Endpoint's being targeted. If you have ideas or a draft implementation that balances ease of use and maintainability across APIs, please let us know on GitHub! For now, you can use the example to implement a callback that suits your needs.
