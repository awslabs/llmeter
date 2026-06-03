# S3 Storage

LLMeter supports reading and writing experiment results to Amazon S3 using standard path operations. This works transparently through the [UPath](https://github.com/fsspec/universal_pathlib) integration — you can use `s3://` paths anywhere you'd use local file paths.

## How It Works

LLMeter includes a built-in S3 filesystem backend (`Boto3S3FileSystem`) that uses `boto3` directly. This replaces the previous `s3fs`/`aiobotocore` dependency chain, eliminating the version conflicts those packages caused with `boto3`.

The backend registers itself automatically when you import `llmeter`. No extra installation or configuration is needed beyond having valid AWS credentials.

```python
from upath import UPath as Path

# These just work — no extra packages needed
results_path = Path("s3://my-bucket/experiments/run-001/")
results_path.mkdir(parents=True, exist_ok=True)

# Write results
(results_path / "metrics.json").write_text('{"latency_p50": 0.42}')

# Read results
data = (results_path / "metrics.json").read_text()

# List files
for p in results_path.iterdir():
    print(p.name)

# Glob patterns
json_files = list(results_path.glob("**/*.json"))
```

## Credentials

The backend uses standard AWS credential resolution — the same mechanisms `boto3` uses:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS config files (`~/.aws/credentials`, `~/.aws/config`)
3. IAM instance profiles (on EC2/ECS/Lambda)
4. SSO/credential process configurations

If `boto3` works in your environment, S3 paths in LLMeter will too.

## Custom Configuration

For advanced use cases, you can instantiate the filesystem directly:

```python
import boto3
from llmeter.s3fs import Boto3S3FileSystem

# Custom session (e.g., specific profile)
session = boto3.Session(profile_name="my-profile")
fs = Boto3S3FileSystem(session=session)

# Custom region
fs = Boto3S3FileSystem(region_name="eu-west-1")

# S3-compatible services (MinIO, LocalStack)
fs = Boto3S3FileSystem(endpoint_url="http://localhost:9000")
```

## Fallback Behavior

If you prefer a different S3 backend, the built-in one steps aside:

- If `s3fs` is installed, LLMeter defers to it
- If `obstore` is installed, LLMeter defers to it
- Otherwise, LLMeter's built-in `Boto3S3FileSystem` handles S3 paths

This means existing setups with `s3fs` continue to work unchanged.

## Limitations

- **File size**: The backend uses `put_object` for writes, which has a 5 GiB limit. This is fine for LLMeter's typical use case (JSON/JSONL result files in the KB–MB range).
- **Append mode**: `open(mode="a")` uses read-modify-write semantics. Concurrent writers to the same key may cause data loss.
- **No multipart upload**: Files larger than 5 GiB require multipart upload, which is not implemented.

## Async Support

The backend is async-native, wrapping synchronous boto3 calls in `asyncio.to_thread()` for non-blocking execution. This means it works well in async code:

```python
import asyncio
from llmeter.s3fs import Boto3S3FileSystem

async def main():
    fs = Boto3S3FileSystem()

    # Concurrent reads
    results = await asyncio.gather(
        fs._cat_file("my-bucket/file1.json"),
        fs._cat_file("my-bucket/file2.json"),
        fs._cat_file("my-bucket/file3.json"),
    )

asyncio.run(main())
```
