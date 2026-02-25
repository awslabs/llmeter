# Bedrock Integration Tests

This directory contains integration tests that verify LLMeter's Bedrock endpoint implementations work correctly with actual AWS Bedrock services.

## Overview

The integration test suite validates three main integration paths:

1. **Bedrock Converse API** - Tests for BedrockConverse and BedrockConverseStream endpoints
2. **Bedrock Invoke API** - Tests for BedrockInvoke and BedrockInvokeStream endpoints
3. **OpenAI SDK with Bedrock** - Tests for using the OpenAI SDK to call Bedrock models via Bedrock's OpenAI-compatible endpoint (Bedrock Mantle)

## Running Integration Tests

Integration tests are **opt-in by default** to avoid AWS costs and credential requirements during regular development.

### Run Integration Tests

To run all integration tests:

```bash
uv run pytest -m integ
```

### Skip Integration Tests (Default)

By default, integration tests are skipped when running the regular test suite:

```bash
uv run pytest
```

This is configured in `pyproject.toml` with `addopts = "-m 'not integ'"`.

### Run Specific Test Files

To run tests from a specific file:

```bash
# Bedrock Converse API tests
uv run pytest -m integ tests/integ/test_bedrock_converse.py

# Bedrock Invoke API tests
uv run pytest -m integ tests/integ/test_bedrock_invoke.py

# OpenAI SDK with Bedrock tests
uv run pytest -m integ tests/integ/test_openai_bedrock.py

# Error handling tests
uv run pytest -m integ tests/integ/test_error_handling.py
```

### Run All Tests (Including Integration)

To run all tests including integration tests:

```bash
uv run pytest -m ""
```

## AWS Requirements

### Required AWS Permissions

Integration tests require the following IAM permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "arn:aws:bedrock:*::foundation-model/openai.gpt-oss-20b-1:0"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

### AWS Credentials

Integration tests use AWS credentials from one of the following sources (in order of precedence):

1. **Environment Variables**:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_SESSION_TOKEN` (optional, for temporary credentials)

2. **AWS CLI Configuration**:
   - Credentials from `~/.aws/credentials`
   - IAM role credentials (when running on EC2, ECS, Lambda, etc.)

3. **IAM Role** (when running in AWS environments)

If AWS credentials are not available, all integration tests will be automatically skipped with an appropriate message.

## Configuration

### Environment Variables

You can configure integration tests using the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for testing | `us-east-1` |
| `BEDROCK_TEST_MODEL` | Model ID for Converse/Invoke tests | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `BEDROCK_OPENAI_TEST_MODEL` | Model ID for OpenAI SDK tests | `openai.gpt-oss-20b-1:0` |

### Example Configuration

```bash
# Set custom region and model
export AWS_REGION=us-west-2
export BEDROCK_TEST_MODEL=us.anthropic.claude-3-5-sonnet-20241022-v2:0

# Run integration tests
uv run pytest -m integ
```

## AWS Costs

Integration tests make actual API calls to AWS Bedrock services and will incur costs.

### Cost Breakdown

| Test Type | Cost per Test | Description |
|-----------|---------------|-------------|
| Text-only tests | ~$0.0001 | Simple text prompts with minimal tokens |
| Image tests | ~$0.0002 | Multimodal tests with small 100x100 images |
| Error handling tests | ~$0.0001 | Tests with invalid inputs |

### Total Estimated Costs

- **Full test suite**: ~$0.0012 per run
- **Bedrock Converse tests**: ~$0.0006 (4 tests)
- **Bedrock Invoke tests**: ~$0.0002 (2 tests)
- **OpenAI SDK tests**: ~$0.0002 (2 tests)
- **Error handling tests**: ~$0.0002 (3 tests)

### Cost Minimization

The test suite is designed to minimize costs:

- Uses cost-effective models (Claude 3.5 Sonnet v2)
- Uses minimal token payloads (`max_tokens=100-150`)
- Keeps test messages short
- Uses small test images (100x100 pixels)
- Tests are opt-in by default

## Test Structure

### Test Files

- `conftest.py` - Shared fixtures and configuration
- `test_bedrock_converse.py` - Bedrock Converse API tests (Requirements 1, 2)
- `test_bedrock_invoke.py` - Bedrock Invoke API tests (Requirements 3, 4)
- `test_openai_bedrock.py` - OpenAI SDK with Bedrock tests (Requirements 5, 6)
- `test_error_handling.py` - Error handling tests (Requirement 8)

### Test Markers

All integration tests are marked with `@pytest.mark.integ` to enable selective execution.

### Test Isolation

Each integration test:

- Creates fresh endpoint instances (no shared state)
- Uses independent test payloads
- Doesn't depend on execution order
- Cleans up resources automatically (boto3 handles connection cleanup)

For detailed verification of resource cleanup, see [RESOURCE_CLEANUP_VERIFICATION.md](./RESOURCE_CLEANUP_VERIFICATION.md).

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on:
  schedule:
    - cron: '0 0 * * *'  # Run nightly
  workflow_dispatch:  # Allow manual trigger

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      
      - name: Install dependencies
        run: uv sync
      
      - name: Run Integration Tests
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: us-east-1
        run: uv run pytest -m integ
```

### Best Practices for CI/CD

- Store AWS credentials as CI/CD secrets
- Run integration tests on:
  - Scheduled nightly runs
  - Release branches
  - Manual triggers (not on every PR to control costs)
- Use separate AWS account for testing
- Monitor AWS costs from test runs
- Consider using AWS service quotas to limit costs

## Troubleshooting

### Tests are Skipped

If you see "AWS credentials not available" when running integration tests:

1. Verify AWS credentials are configured:
   ```bash
   aws sts get-caller-identity
   ```

2. Check that credentials have the required permissions (see AWS Requirements section)

3. Ensure you're running tests with the `-m integ` flag:
   ```bash
   uv run pytest -m integ
   ```

### Model Not Available

If you see errors about model availability:

1. Verify the model is available in your AWS region:
   ```bash
   aws bedrock list-foundation-models --region us-east-1
   ```

2. Request model access in the AWS Bedrock console if needed

3. Use a different model by setting environment variables:
   ```bash
   export BEDROCK_TEST_MODEL=us.anthropic.claude-3-5-sonnet-20241022-v2:0
   ```

### Rate Limiting

If you encounter rate limiting errors:

1. Tests use boto3's built-in retry logic
2. Consider adding delays between test runs
3. Use pytest-rerunfailures for automatic retries:
   ```bash
   uv run pytest -m integ --reruns 2 --reruns-delay 1
   ```

### OpenAI SDK Tests Skipped

If OpenAI SDK tests are skipped with "OpenAI SDK not installed":

1. Install the OpenAI SDK and AWS token generator:
   ```bash
   poetry add openai aws-bedrock-token-generator
   ```

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [LLMeter Documentation](../../README.md)
- [Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- [Bedrock Invoke API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html)
- [Bedrock OpenAI-compatible endpoint](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-openai.html)
