# Save (and Reload) Results

!!! warning "Important security warning"
    Do not use LLMeter file load functions on data from untrusted sources! For more details on why, see the following section.

LLMeter can save your test configurations and results to files - whether locally or on the Cloud - and load past runs back into Python for analysis later. For example:

```python
from llmeter.results import Result
from llmeter.runner import Runner

# Provide s3:// URI or local path:
base_output_path = f"s3://doc-example-bucket/llmeter/outputs/{endpoint.model_id}"

runner = Runner(
    endpoint,
    # Configure the output path when creating a Runner...
    output_path=base_output_path,
)
results = await runner.run(
    payload=sample_payload,
    n_requests=3,
    clients=5,
    # ...*Optionally* specify a specific path for an individual run:
    # (Otherwise a subfolder will be created by run name, automatically)
    output_path=f"{base_output_path}/my-cool-run"
)

# At some later date, load your result back:
results = Result.load(f"{base_output_path}/my-cool-run")
```

!!! note "A note on performance"
    While it's *possible* for LLMeter to write run outputs directly to Cloud object stores like
    [Amazon S3](https://aws.amazon.com/s3/), remember it might reduce the maximum throughput you
    can drive in *high-volume* tests, since it will consume network bandwidth.

For the most part, generated files are JSON-based (or [JSON Lines](https://jsonlines.org/), for the
individual responses). However, LLMeter also handles some more complex data types including:

- Binary data in request/response payloads (such as images)
- [Callbacks](./callbacks) configured on the test Run, including LLMeter built-ins as well as *your custom* callback classes

This core de/serialization functionality is implemented in the [`llmeter.serialization`](../reference/serialization) module.


## How complex data types are represented and loaded

Objects that are not natively JSON serializable (but support LLMeter's serialization protocol) are saved to JSON-based formats something like the below:

```json
{
    "__llmeter_class__": "llmeter.endpoints.bedrock.BedrockConverse",
    "__llmeter_state__": {"model_id": "claude-3", "region": "us-west-2"}
}
```

- `datetime` objects are stored as ISO-8601 format strings in UTC timezone, like `2024-01-01T00:00:00Z`.
- `bytes` objects are serialized to base64 strings in a special `{"__llmeter_bytes__": "<base64>"}` wrapper.
- Objects implementing LLMeter's [`Serializable`](../reference/serialization.md#llmeter.serialization.Serializable) interface (including for example Endpoints and Callbacks) are represented as dicts with the `__llmeter_{class/state}__` properties as shown above

When **loading** these objects back from file, LLMeter will try to import and instantiate **any** class path saved in the `__llmeter_class__` field.

This is the same trust model as Python's native [pickle](https://docs.python.org/3/library/pickle.html) library: It is possible to construct malicious data which will run arbitrary code during loading, so **only load data that you trust**.

## Under the hood: Serialization API components

In many cases you'll be working with high-level classes like `Runner` and `Result` that already provide convenience methods to save to and load from file. However, building custom LLMeter extensions may require understanding how our [`llmeter.serialization`](../reference/serialization) components fit together:

| Symbol | Purpose |
|--------|---------|
|  [`Serializable`](../reference/serialization.md#llmeter.serialization.Serializable) | Mixin enabling your class to be serialized and loaded by LLMeter |
| [`dump_object`](../reference/serialization.md#llmeter.serialization.dump_object) / [`load_object`](../reference/serialization.md#llmeter.serialization.load_object) | Full round-trip persistence via a type-tagged envelope |
| [`json_default`](../reference/serialization.md#llmeter.serialization.json_default) | `json.dumps` fallback for bytes, datetime, PathLike |
| [`bytes_decoder`](../reference/serialization.md#llmeter.serialization.bytes_decoder) | `json.loads` object hook to restore `__llmeter_bytes__` markers |
| [`datetime_to_str`](../reference/serialization.md#llmeter.serialization.datetime_to_str) / [`str_to_datetime`](../reference/serialization.md#llmeter.serialization.str_to_datetime) | UTC ISO-8601 with `Z` suffix, both directions |

```python
from llmeter.serialization import (
    dump_object, load_object, json_default, bytes_decoder,
    datetime_to_str, str_to_datetime,
)
```

### Make custom classes serializable by LLMeter with the `Serializable` mixin

The `Serializable` mixin provides two default methods:

- `_get_llmeter_state`: Construct a JSON-ready dictionary of the state your class needs to be re-initialized
    - The default implementation inspects the arguments of your `__init__` constructor and attempts to fetch those from the current object's fields, or with an `_` underscore prefix if the raw parameter name isn't present.
- `_set_llmeter_state`: Initialise an instance of your class, based on a state dictionary
    - The default implementation calls your `__init__` with the arguments stored in the dictionary.

Nested `Serializable` objects are recursively handled by default: `_get_llmeter_state` wraps them via `dump_object`, and `_set_llmeter_state` restores them via `load_object`.

You'd only need to **override** these implementations if, for example:

- An `__init__` parameter is consumed without being stored, or
- Reconstruction needs special logic beyond `__init__(**state)`, or
- You want to exclude large transient data from persistence


### Save and load your own objects

Once your class inherits `Serializable`, it gets `save_to_file()` and `load_from_file()` for free - no extra code required:

```python
from llmeter.callbacks.base import Callback
from llmeter.callbacks.mlflow import MlflowCallback

cb = MlflowCallback(step=5, nested=True)
cb.save_to_file("/tmp/callback.json")

# load_from_file is *polymorphic*: it reads the __llmeter_class__ recorded in the
# file and rebuilds the correct subclass, so you can call it on the base class.
restored = Callback.load_from_file("/tmp/callback.json")  # -> MlflowCallback(step=5, nested=True)
```

Under the hood, those methods use two functions you can also call directly if you're managing the JSON yourself:

- `dump_object(obj)` builds the type-tagged envelope (`{"__llmeter_class__": ..., "__llmeter_state__": ...}`). It reads the state from `obj._get_llmeter_state()`, or - for a plain `@dataclass` that *doesn't* inherit `Serializable` - from its fields.
- `load_object(data)` imports the class named in `__llmeter_class__`, creates a bare instance (via `__new__`, bypassing `__init__`), then repopulates it through `_set_llmeter_state()` - which by default re-runs `__init__` with the saved state.

```python
from llmeter.serialization import dump_object, load_object

data = dump_object(my_object)   # -> plain dict
my_object = load_object(data)   # -> reconstructed instance
```

Note the envelope produced by `dump_object` may still contain non-JSON values (like `bytes` or `datetime`) nested inside its state, so pair it with `json_default` when you actually write it out - see below.

### Reading and writing the JSON yourself

When you call `json.dump`/`json.dumps` directly on LLMeter data, pass `json_default` so the extra types are handled:

```python
import json
from llmeter.serialization import json_default, bytes_decoder

with open("my-file.json", "w") as f:
    json.dump(my_data, f, default=json_default, indent=4)
```

`json_default` converts, in order:

- **bytes** — wrapped in a `{"__llmeter_bytes__": "<base64>"}` marker
- **datetime** — UTC ISO-8601 string with a `Z` suffix
- **date / time** — `.isoformat()`
- **os.PathLike** — POSIX path string
- **anything else** — `str()` fallback

To turn the `bytes` markers back into `bytes` on the way in, pass `bytes_decoder` as the `object_hook`:

```python
with open("my-file.json") as f:
    data = json.load(f, object_hook=bytes_decoder)
```

## How LLMeter's built-ins use this

For everyday use you rarely touch the functions above directly, because the high-level classes wrap them for you.

### Endpoints: config, not connections

Endpoints are saved as *configuration only* - runtime state like a boto3 client is never written to file. When you load an endpoint back, `_set_llmeter_state` re-runs the constructor, which recreates that client for you:

```python
from llmeter.serialization import dump_object, load_object

data = dump_object(endpoint)
# → {"__llmeter_class__": "llmeter.endpoints.bedrock.BedrockConverse",
#    "__llmeter_state__": {"model_id": "claude-3", "region": "us-west-2"}}

restored = load_object(data)  # boto3 client rebuilt via __init__
```

### Runners: the whole run configuration

When you give a `Runner` an `output_path`, it saves its full configuration - endpoint, tokenizer and callbacks included - as `run_config.json` at the start of each run. You can also trigger this yourself:

```python
runner = Runner(endpoint=BedrockConverse(...), callbacks=[MlflowCallback(step=1)])
runner.save(output_path="/tmp/run")   # writes /tmp/run/run_config.json
```

If a callback (or any other configured object) can't be serialized because its class doesn't inherit `Serializable`, saving raises a `TypeError`. This is why custom callbacks and cost dimensions should subclass the relevant LLMeter base class - see [`Callback`](../reference/callbacks/base.md#llmeter.callbacks.base.Callback) and the [cost dimension base classes](../reference/callbacks/cost/dimensions.md).

### Dataclasses work automatically

Because the `Serializable` mixin introspects the constructor, any `@dataclass` that inherits it is serializable with no extra code - the generated `__init__` supplies the parameter names the mixin looks for. LLMeter's built-in cost dimensions are a good example:

```python
from llmeter.callbacks.cost.dimensions import InputTokens
from llmeter.serialization import dump_object, load_object

data = dump_object(InputTokens(price_per_million=3.0))
restored = load_object(data)  # -> InputTokens(price_per_million=3.0, granularity=1)
```
