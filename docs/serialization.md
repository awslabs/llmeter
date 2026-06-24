# Serialization

LLMeter provides a unified serialization layer that works consistently across all object types — Pydantic models, dataclasses, plain classes, and callable objects with runtime state.

## Two operations, two purposes

| Function | Returns | Use case |
|----------|---------|----------|
| `to_dict(obj)` | Python dict with **native types** (datetime, bytes, UPath) | In-memory access, stats computation, jmespath queries |
| `serialize(obj)` | **JSON-safe** dict (strings, numbers, lists, dicts, None) | Writing to disk, JSON output, network transfer |

```python
from llmeter.serialization import to_dict, serialize

response = InvocationResponse(response_text="hi", request_time=datetime.now(timezone.utc))

to_dict(response)["request_time"]      # → datetime(2024, 6, 15, 10, 30, ...)
serialize(response)["request_time"]    # → "2024-06-15T10:30:00Z"
```

Both functions work on **any** LLMeter object — the user never needs to know whether something is a Pydantic model, a dataclass, or a plain class.

## Object methods

Every LLMeter object exposes `to_dict()` as a method for convenience:

```python
response.to_dict()       # same as to_dict(response)
result.to_dict()         # excludes responses by default
endpoint.to_dict()       # endpoint config as dict
cost_model.to_dict()     # includes dimension metadata
```

## Persisting runtime objects

Objects that hold runtime state (boto3 clients, SDK connections, compiled tokenizers) can't be naively serialized to JSON. LLMeter handles this via the `__getstate__`/`__setstate__` protocol:

```python
from llmeter.serialization import dump_object, load_object

# Save an endpoint (including its config, but not the boto3 client)
data = dump_object(endpoint)
# → {"__llmeter_class__": "llmeter.endpoints.bedrock.BedrockConverse",
#    "__llmeter_state__": {"model_id": "claude-3", "region": "us-west-2"}}

# Restore it (boto3 client is recreated automatically)
restored = load_object(data)
```

### How it works

1. `dump_object(obj)` calls `serialize(obj)` to get a JSON-safe state dict, then wraps it with the class path
2. `load_object(data)` imports the class, creates an empty instance, and calls `__setstate__(state)` to reconstruct it

### Default behavior (zero boilerplate)

Base classes (`Endpoint`, `Tokenizer`, `Callback`) provide default `__getstate__`/`__setstate__` implementations that:

- **`__getstate__`**: Introspects `__init__` parameters, matches them to instance attributes (`self.name` or `self._name`), and returns only what's needed to recreate the object
- **`__setstate__`**: Calls `__init__(**state)` which recreates runtime resources (clients, connections) from the config

This means most subclasses work without any custom serialization code:

```python
class MyEndpoint(Endpoint):
    def __init__(self, model_id: str, region: str = "us-east-1"):
        super().__init__(endpoint_name=model_id, model_id=model_id, provider="custom")
        self.region = region
        self._client = create_client(region)  # runtime — not serialized

# Just works:
data = dump_object(MyEndpoint(model_id="my-model"))
restored = load_object(data)  # _client recreated via __init__
```

### When to override

Override `__getstate__`/`__setstate__` only when:

- An `__init__` parameter is consumed without being stored (uncommon)
- Reconstruction needs special logic beyond `__init__(**state)`
- You want to exclude large transient data from persistence

```python
class SpecialEndpoint(Endpoint):
    def __getstate__(self) -> dict:
        # Only save what matters — skip the 10MB cache
        return {"model_id": self.model_id, "region": self.region}

    def __setstate__(self, state: dict):
        self.__init__(**state)
```

## Callback persistence

All callbacks support `save_to_file()` / `load_from_file()` via this protocol:

```python
from llmeter.callbacks.base import Callback
from llmeter.callbacks.mlflow import MlflowCallback

# Save
cb = MlflowCallback(step=5, nested=True)
cb.save_to_file("/tmp/callback.json")

# Load (polymorphic — detects the type automatically)
restored = Callback.load_from_file("/tmp/callback.json")
# → MlflowCallback(step=5, nested=True)
```

## Runner config persistence

`Runner.save()` and `Runner.load()` use `dump_object`/`load_object` for all callable fields:

```python
runner = Runner(endpoint=BedrockConverse(...), callbacks=[MlflowCallback(step=1)])
runner.save(output_path="/tmp/run")

# Saved JSON includes:
# {"endpoint": {"__llmeter_class__": "...BedrockConverse", "__llmeter_state__": {...}},
#  "tokenizer": {"__llmeter_class__": "...DummyTokenizer", "__llmeter_state__": {}},
#  "callbacks": [{"__llmeter_class__": "...MlflowCallback", "__llmeter_state__": {"step": 1, "nested": false}}]}

# Full reconstruction:
restored = Runner.load("/tmp/run")
```

## Security

`load_object` will import and instantiate whatever class path is in the `_class` field. This is the same trust model as Python's `pickle` — **never load configs from untrusted sources**.

This is appropriate for LLMeter's use case: developer-generated configs stored on local disk or controlled cloud storage (S3 with IAM).

## Dataclass compatibility

`@dataclass` classes work seamlessly with this system. The `default_getstate` introspects the `__init__` that `@dataclass` generates:

```python
@dataclass
class InputTokens(RequestCostDimensionBase):
    price_per_million: float
    granularity: int = 1

# __getstate__ and __setstate__ inherited from base — no custom code needed
data = dump_object(InputTokens(price_per_million=3.0))
restored = load_object(data)
```
