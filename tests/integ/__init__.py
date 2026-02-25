"""Integration tests for LLMeter Bedrock endpoints.

This package contains integration tests that verify LLMeter's Bedrock endpoint
implementations work correctly with actual AWS Bedrock services.

These tests are marked with @pytest.mark.integ and are skipped by default to avoid
AWS costs and credential requirements during regular development.

To run integration tests:
    uv run pytest -m integ

Required AWS Permissions:
    - bedrock:InvokeModel
    - bedrock:InvokeModelWithResponseStream
    - sts:GetCallerIdentity

Estimated Cost: ~$0.0012 per full test run
"""
