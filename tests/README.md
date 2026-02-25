# LLMeter Testing Guide

This directory contains the complete test suite for LLMeter, including unit tests, integration tests, and property-based tests.

## Quick Start

```bash
# Install test dependencies
uv sync --group test

# Run all unit tests (default)
uv run pytest

# Run all tests including integration tests
uv run pytest -m ""

# Run with coverage
uv run pytest --cov=llmeter

# Run specific test types
uv run pytest -m integ  # Integration tests only
uv run pytest tests/unit  # Unit tests only
```

## Test Structure

```terminal
tests/
├── unit/              # Unit tests (fast, no external dependencies)
│   ├── callbacks/     # Callback system tests
│   ├── endpoints/     # Endpoint implementation tests
│   ├── parsers/       # Response parser tests
│   └── *.py          # Core functionality tests
└── integ/            # Integration tests (AWS Bedrock)
    ├── callbacks/     # Integration callback tests
    └── *.py          # Bedrock API integration tests
```

## Test Types

### Unit Tests

Unit tests validate core functionality without external dependencies. They use mocking (via `moto`) for AWS services and run quickly.

**Location**: `tests/unit/`

**Run unit tests**:

```bash
uv run pytest tests/unit
```

**Coverage areas**:

- Experiment configuration and execution
- Result aggregation and statistics
- File I/O and serialization
- Plotting and visualization
- Tokenization
- Prompt utilities
- Endpoint implementations (mocked)
- Callback system
- Response parsers

### Integration Tests

Integration tests validate LLMeter works correctly with actual AWS Bedrock services. These tests are opt-in to avoid AWS costs and credential requirements.

**Location**: `tests/integ/`

**Run integration tests**:

```bash
uv run pytest -m integ
```

**Coverage areas**:

- Bedrock Converse API (text and multimodal)
- Bedrock Invoke API (streaming and non-streaming)
- OpenAI SDK with Bedrock compatibility
- Error handling and edge cases

See [Integration Tests](#integration-tests-aws-bedrock) section below for detailed information.

### Property-Based Tests

Property-based tests use Hypothesis to generate test cases and validate invariants across a wide range of inputs.

**Framework**: [Hypothesis](https://hypothesis.readthedocs.io/)

**Run property tests**:

```bash
uv run pytest tests/unit/test_property_*.py
```

**Coverage areas**:

- Save/load round-trip consistency
- Data structure invariants
- Edge case discovery

## Running Tests

### Basic Commands

```bash
# Run all unit tests (default, integration tests skipped)
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_runner.py

# Run specific test function
uv run pytest tests/unit/test_runner.py::test_runner_basic

# Run tests matching pattern
uv run pytest -k "test_plotting"

# Run with verbose output
uv run pytest -v

# Run with extra verbose output (show all test names)
uv run pytest -vv
```

### Coverage Reports

```bash
# Run with coverage report
uv run pytest --cov=llmeter

# Generate HTML coverage report
uv run pytest --cov=llmeter --cov-report=html

# View HTML report
open htmlcov/index.html
```

### Test Markers

Tests are organized using pytest markers:

```bash
# Run integration tests only
uv run pytest -m integ

# Run all tests (including integration)
uv run pytest -m ""

# Run tests NOT marked as integration
uv run pytest -m "not integ"
```

## Dependencies

### Test Dependencies

Test dependencies are defined in the `test` dependency group in `pyproject.toml`:

```bash
# Install test dependencies
uv sync --group test
```

**Core test dependencies**:

- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `pytest-reportlog` - Test result logging
- `hypothesis` - Property-based testing
- `moto` - AWS service mocking
- `pillow` - Image processing for multimodal tests
- `aws-bedrock-token-generator` - Bedrock authentication

### Optional Dependencies for Testing

Some tests require optional dependencies:

```bash
# Install with OpenAI SDK (for OpenAI-Bedrock integration tests)
uv sync --extra openai --group test

# Install with all extras (recommended for full test coverage)
uv sync --all-extras --group test
```

## Integration Tests (AWS Bedrock)

### Overview

Integration tests validate LLMeter's Bedrock endpoint implementations with actual AWS services:

1. **Bedrock Converse API** - BedrockConverse and BedrockConverseStream endpoints
2. **Bedrock Invoke API** - BedrockInvoke and BedrockInvokeStream endpoints  
3. **OpenAI SDK with Bedrock** - OpenAI SDK calling Bedrock via Mantle

### AWS Requirements

**Required IAM permissions**:

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
    }
  ]
}
```

**AWS credentials** (in order of precedence):

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS CLI configuration (`~/.aws/credentials`)
3. IAM role (when running in AWS environments)

If credentials are not available, integration tests are automatically skipped.

### Running Integration Tests

```bash
# Run all integration tests
uv run pytest -m integ

# Run specific integration test file
uv run pytest -m integ tests/integ/test_bedrock_converse.py

# Run with custom AWS region
AWS_REGION=us-west-2 uv run pytest -m integ
```

### Configuration

Configure integration tests via environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `AWS_REGION` | AWS region for testing | `us-east-1` |
| `BEDROCK_TEST_MODEL` | Model for Converse/Invoke tests | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `BEDROCK_OPENAI_TEST_MODEL` | Model for OpenAI SDK tests | `openai.gpt-oss-20b-1:0` |

### AWS Costs

Integration tests make real API calls and incur costs:

| Test Type | Cost per Test | Total |
| --- | --- | --- |
| Text-only tests | ~$0.0001 | - |
| Image tests | ~$0.0002 | - |
| Error handling tests | ~$0.0001 | - |
| **Full suite** | - | **~$0.0012** |

**Cost minimization**:

- Uses cost-effective models
- Minimal token payloads (max_tokens=100-150)
- Short test messages
- Small test images (100x100 pixels)
- Opt-in by default

### Troubleshooting Integration Tests

**Tests are skipped**:

1. Verify AWS credentials:

   ```bash
   aws sts get-caller-identity
   ```

2. Check required permissions (see AWS Requirements)

3. Run with integration marker:

   ```bash
   uv run pytest -m integ
   ```

**Model not available**:

1. Check model availability in your region:

   ```bash
   aws bedrock list-foundation-models --region us-east-1
   ```

2. Request model access in AWS Bedrock console

3. Use different model:

   ```bash
   export BEDROCK_TEST_MODEL=us.anthropic.claude-3-5-sonnet-20241022-v2:0
   ```

**OpenAI SDK tests skipped**:

Install OpenAI extra:

```bash
uv sync --extra openai --group test
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      
      - name: Install dependencies
        run: uv sync --group test
      
      - name: Run unit tests
        run: uv run pytest --cov=llmeter
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    # Run on schedule or manual trigger only
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      
      - name: Install dependencies
        run: uv sync --all-extras --group test
      
      - name: Run integration tests
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: us-east-1
        run: uv run pytest -m integ
```

### Best Practices

- Run unit tests on every commit
- Run integration tests on:
  - Scheduled nightly runs
  - Release branches
  - Manual triggers (to control costs)
- Store AWS credentials as CI/CD secrets
- Use separate AWS account for testing
- Monitor AWS costs from test runs
- Set AWS service quotas to limit costs

## Writing Tests

### Unit Test Example

```python
import pytest
from llmeter import Runner, Experiment

def test_runner_basic():
    """Test basic runner functionality."""
    experiment = Experiment(
        name="test",
        endpoint="mock-endpoint",
        payload={"prompt": "test"}
    )
    
    runner = Runner(experiment)
    results = runner.run()
    
    assert results is not None
    assert len(results) > 0
```

### Integration Test Example

```python
import pytest
from llmeter.endpoints import BedrockConverse

@pytest.mark.integ
def test_bedrock_converse_text():
    """Test Bedrock Converse with text input."""
    endpoint = BedrockConverse(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "inferenceConfig": {"maxTokens": 100}
    }
    
    response = endpoint.invoke(payload)
    
    assert response is not None
    assert "output" in response
```

### Property-Based Test Example

```python
from hypothesis import given, strategies as st
import pytest

@given(st.integers(min_value=1, max_value=1000))
def test_runner_with_various_counts(count):
    """Test runner with property-based testing."""
    experiment = Experiment(
        name="test",
        endpoint="mock-endpoint",
        payload={"prompt": "test"},
        count=count
    )
    
    runner = Runner(experiment)
    results = runner.run()
    
    assert len(results) == count
```

## Test Configuration

### pytest.ini Options

Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_default_fixture_loop_scope = "function"
filterwarnings = ["ignore::DeprecationWarning"]
markers = [
    "integ: marks tests as integration tests (enabled by default, disabled in CI)"
]
```

### Custom Fixtures

Shared fixtures are defined in:

- `tests/integ/conftest.py` - Integration test fixtures
- Individual test files - Test-specific fixtures

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [LLMeter Documentation](../README.md)
- [Contributing Guide](../CONTRIBUTING.md)
