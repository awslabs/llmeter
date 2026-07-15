# Serialization

LLMeter provides a unified serialization layer in a single module —
`llmeter.serialization` — that handles JSON encoding, datetime conversion,
binary content round-tripping, and full object persistence.

## Module overview

| Symbol | Purpose |
|--------|---------|
| `Serializable` | Mixin giving any class automatic `__getstate__`/`__setstate__` |
| `dump_object` / `load_object` | Full round-trip persistence via a type-tagged envelope |
| `json_default` | `json.dumps` fallback for bytes, datetime, PathLike |
| `bytes_decoder` | `json.loads` object hook to restore `__llmeter_bytes__` markers |
| `datetime_to_str` / `str_to_datetime` | UTC ISO-8601 with `Z` suffix, both directions |

```python
from llmeter.serialization import (
    dump_object, load_object, json_default, bytes_decoder,
    datetime_to_str, str_to_datetime,
)
```

## JSON encoding

Use `json_default` as the `default` argument whenever you call `json.dumps` or
`json.dump` with LLMeter objects:

```python
import json
from llmeter.serialization import json_default

json.dump(my_data, f, default=json_default, indent=4)
```

It handles:

- **bytes** — wrapped in `{"__llmeter_bytes__": "<base64>"}` markers
- **datetime** — UTC ISO-8601 string via `datetime_to_str`
- **date / time** — `.isoformat()`
- **os.PathLike** — POSIX path string
- **anything else** — `str()` fallback

To decode bytes markers back on load:

```python
data = json.load(f, object_hook=bytes_decoder)
```

## Datetime handling

All datetime serialization goes through two functions:

```python
from llmeter.serialization import datetime_to_str, str_to_datetime

datetime_to_str(datetime(2024, 1, 1, tzinfo=timezone.utc))
# → "2024-01-01T00:00:00Z"

str_to_datetime("2024-01-01T00:00:00Z")
# → datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
```

Timezone-aware datetimes are normalized to UTC. Naive datetimes are serialized
as-is. The `Z` suffix is canonical.

## Persisting runtime objects

Objects with runtime state (boto3 clients, SDK connections, compiled
tokenizers) can't be naively serialized. LLMeter handles this via the
`__getstate__`/`__setstate__` protocol:

```python
from llmeter.serialization import dump_object, load_object

# Save an endpoint (config only — not the boto3 client)
data = dump_object(endpoint)
# → {"__llmeter_class__": "llmeter.endpoints.bedrock.BedrockConverse",
#    "__llmeter_state__": {"model_id": "claude-3", "region": "us-west-2"}}

# Restore it (boto3 client recreated automatically via __init__)
restored = load_object(data)
```

### How it works

1. `dump_object(obj)` calls `obj.__getstate__()` to get a JSON-safe state
   dict, then wraps it with the fully-qualified class path.
2. `load_object(data)` imports the class, creates an empty instance via
   `cls.__new__`, and calls `obj.__setstate__(state)` to reconstruct it.

### The `Serializable` mixin (zero boilerplate)

Inherit from `Serializable` to get automatic `__getstate__`/`__setstate__`:

```python
from llmeter.serialization import Serializable

class MyEndpoint(Serializable, Endpoint):
    def __init__(self, model_id: str, region: str = "us-east-1"):
        super().__init__(endpoint_name=model_id, model_id=model_id, provider="custom")
        self.region = region
        self._client = create_client(region)  # runtime — not serialized
```

The mixin introspects `__init__` parameters, matches them to instance
attributes (`self.name` or `self._name`), and serializes only what's needed to
recreate the object. Private attributes (prefixed with `_`) that don't
correspond to `__init__` params are skipped — they're assumed to be derived
runtime state.

Nested `Serializable` objects are recursively handled: `__getstate__` wraps
them via `dump_object`, and `__setstate__` restores them via `load_object`.

### When to override

Override `__getstate__`/`__setstate__` only when:

- An `__init__` parameter is consumed without being stored
- Reconstruction needs special logic beyond `__init__(**state)`
- You want to exclude large transient data from persistence

```python
class SpecialEndpoint(Serializable, Endpoint):
    def __getstate__(self) -> dict:
        return {"model_id": self.model_id, "region": self.region}

    def __setstate__(self, state: dict):
        self.__init__(**state)
```

## Callback persistence

All callbacks support `save_to_file()` / `load_from_file()`:

```python
from llmeter.callbacks.base import Callback
from llmeter.callbacks.mlflow import MlflowCallback

cb = MlflowCallback(step=5, nested=True)
cb.save_to_file("/tmp/callback.json")

# Polymorphic load — detects the type from __llmeter_class__
restored = Callback.load_from_file("/tmp/callback.json")
# → MlflowCallback(step=5, nested=True)
```

## Runner config persistence

`_RunConfig.save()` and `_RunConfig.load()` use `dump_object`/`load_object`
for all callable fields (endpoint, tokenizer, callbacks):

```python
runner = Runner(endpoint=BedrockConverse(...), callbacks=[MlflowCallback(step=1)])
runner.save(output_path="/tmp/run")

# Full reconstruction:
restored = _RunConfig.load("/tmp/run")
```

When loading, the runner detects both formats:

- **New format** (`__llmeter_class__` key) — uses `load_object`
- **Legacy format** (`endpoint_type` / `tokenizer_module` keys) — uses
  `Endpoint.load()` / `Tokenizer.load()` for backward compatibility

## Dataclass compatibility

`@dataclass` classes work seamlessly with `Serializable`. The mixin
introspects the `__init__` that `@dataclass` generates:

```python
@dataclass
class InputTokens(Serializable):
    price_per_million: float
    granularity: int = 1

data = dump_object(InputTokens(price_per_million=3.0))
restored = load_object(data)
```

## Security

`load_object` imports and instantiates whatever class path is found in the
`__llmeter_class__` field. This is the same trust model as Python's `pickle` —
**never load configs from untrusted sources**.

This is appropriate for LLMeter's use case: developer-generated configs stored
on local disk or controlled cloud storage (S3 with IAM).
