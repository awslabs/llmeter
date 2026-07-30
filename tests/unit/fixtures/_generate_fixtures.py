#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot script to generate fixture data with mathematically consistent stats.

Run once from the repo root:
    uv run python tests/unit/fixtures/_generate_fixtures.py

This writes all fixture files under tests/unit/fixtures/result_snapshots/.
After running, verify the tests pass, then this script can be kept as documentation
of how the fixtures were generated (or deleted if preferred).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llmeter.endpoints.base import InvocationResponse
from llmeter.results import Result
from llmeter.serialization import json_default, restore_dataclass_types

FIXTURES_DIR = Path(__file__).parent / "result_snapshots"


def _write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=4, default=json_default) + "\n")


def _write_jsonl(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n")


def _compute_and_write_stats(
    responses: list[InvocationResponse], summary: dict, path: Path
):
    """Build a Result from the summary metadata, compute stats, write stats.json.

    Passing the full summary metadata (parsed to native types) makes the resulting
    stats.json mirror what a real ``Result.save()`` produces.
    """
    meta = {k: v for k, v in summary.items() if k not in ("total_requests",)}
    restore_dataclass_types(Result, meta)
    result = Result(
        responses=responses, total_requests=summary["total_requests"], **meta
    )
    stats = Result._compute_stats(result)
    _write_json(path, stats)
    return stats


# =============================================================================
# Scenario: base (modern format, OpenAI endpoint, CostModel + Mlflow callbacks)
# =============================================================================


def generate_base():
    out = FIXTURES_DIR / "base"

    responses = [
        InvocationResponse(
            id="synth-base-001",
            response_text="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            input_prompt="Summarize the key concepts of distributed systems.",
            input_payload={
                "messages": [
                    {
                        "role": "user",
                        "content": "Summarize the key concepts of distributed systems.",
                    }
                ],
                "max_tokens": 256,
                "model": "synthetic.test-model-v1",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            time_to_first_token=0.452,
            time_to_last_token=1.203,
            num_tokens_input=42,
            num_tokens_output=87,
            num_tokens_input_cached=None,
            num_tokens_output_reasoning=None,
            time_per_output_token=0.00863,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            annotations={},
        ),
        InvocationResponse(
            id="synth-base-002",
            response_text="Sed ut perspiciatis unde omnis iste natus error sit voluptatem.",
            input_prompt="Explain the CAP theorem briefly.",
            input_payload={
                "messages": [
                    {"role": "user", "content": "Explain the CAP theorem briefly."}
                ],
                "max_tokens": 256,
                "model": "synthetic.test-model-v1",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            time_to_first_token=0.389,
            time_to_last_token=0.951,
            num_tokens_input=38,
            num_tokens_output=64,
            num_tokens_input_cached=12,
            num_tokens_output_reasoning=None,
            time_per_output_token=0.00879,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 15, 10, 0, 1, tzinfo=timezone.utc),
            annotations={},
        ),
        InvocationResponse(
            id="synth-base-003",
            response_text="Nemo enim ipsam voluptatem quia voluptas sit aspernatur.",
            input_prompt="What is eventual consistency?",
            input_payload={
                "messages": [
                    {"role": "user", "content": "What is eventual consistency?"}
                ],
                "max_tokens": 256,
                "model": "synthetic.test-model-v1",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            time_to_first_token=0.521,
            time_to_last_token=1.445,
            num_tokens_input=35,
            num_tokens_output=102,
            num_tokens_input_cached=None,
            num_tokens_output_reasoning=8,
            time_per_output_token=0.00906,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 15, 10, 0, 2, tzinfo=timezone.utc),
            annotations={},
        ),
        InvocationResponse(
            id="synth-base-004",
            response_text="Ut enim ad minima veniam, quis nostrum exercitationem.",
            input_prompt="Define partition tolerance.",
            input_payload={
                "messages": [
                    {"role": "user", "content": "Define partition tolerance."}
                ],
                "max_tokens": 256,
                "model": "synthetic.test-model-v1",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            time_to_first_token=0.298,
            time_to_last_token=0.812,
            num_tokens_input=30,
            num_tokens_output=55,
            num_tokens_input_cached=None,
            num_tokens_output_reasoning=None,
            time_per_output_token=0.00934,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 15, 10, 0, 3, tzinfo=timezone.utc),
            annotations={},
        ),
        InvocationResponse(
            id="synth-base-005",
            response_text="Quis autem vel eum iure reprehenderit qui in ea voluptate.",
            input_prompt="Compare synchronous and asynchronous replication.",
            input_payload={
                "messages": [
                    {
                        "role": "user",
                        "content": "Compare synchronous and asynchronous replication.",
                    }
                ],
                "max_tokens": 256,
                "model": "synthetic.test-model-v1",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            time_to_first_token=0.415,
            time_to_last_token=1.102,
            num_tokens_input=40,
            num_tokens_output=78,
            num_tokens_input_cached=None,
            num_tokens_output_reasoning=None,
            time_per_output_token=0.00881,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 15, 10, 0, 4, tzinfo=timezone.utc),
            annotations={},
        ),
    ]

    # summary.json
    summary = {
        "total_requests": 5,
        "clients": 2,
        "n_requests": 3,
        "total_test_time": 4.521,
        "model_id": "synthetic.test-model-v1",
        "output_path": None,
        "endpoint_name": "synthetic-openai-stream",
        "provider": "openai",
        "run_name": "synthetic-base-test",
        "run_description": "Synthetic fixture for deserialization regression tests",
        "start_time": "2025-01-15T10:00:00Z",
        "first_request_time": "2025-01-15T10:00:00Z",
        "last_request_time": "2025-01-15T10:00:04Z",
        "end_time": "2025-01-15T10:00:04Z",
    }
    _write_json(out / "summary.json", summary)

    # stats.json
    _compute_and_write_stats(responses, summary, out / "stats.json")

    # responses.jsonl
    _write_jsonl(out / "responses.jsonl", [r.to_json() for r in responses])

    # run_config.json (modern __llmeter_class__ envelopes)
    run_config = {
        "endpoint": {
            "__llmeter_class__": "llmeter.endpoints.openai.OpenAICompletionStreamEndpoint",
            "__llmeter_state__": {
                "model_id": "synthetic.test-model-v1",
                "endpoint_name": "synthetic-openai-stream",
                "provider": "openai",
                "organization": None,
                "base_url": "https://api.synthetic-provider.example.com/v1/",
                "max_retries": 3,
                "timeout": {
                    "connect": 5.0,
                    "read": 60.0,
                    "write": 5.0,
                    "pool": 5.0,
                },
                "api_key": "sk-synthetic-test-key-not-real",
            },
        },
        "output_path": "stale/path/that/should/be/overridden",
        "tokenizer": {
            "__llmeter_class__": "llmeter.tokenizers.DummyTokenizer",
            "__llmeter_state__": {},
        },
        "clients": 2,
        "n_requests": 3,
        "run_duration": None,
        "payload": "tests/unit/fixtures/result_snapshots/base/payload.jsonl",
        "run_name": "synthetic-base-test",
        "run_description": "Synthetic fixture for deserialization regression tests",
        "timeout": 60,
        "callbacks": [
            {
                "__llmeter_class__": "llmeter.callbacks.cost.model.CostModel",
                "__llmeter_state__": {
                    "request_dims": {
                        "InputTokens": {
                            "__llmeter_class__": "llmeter.callbacks.cost.dimensions.InputTokens",
                            "__llmeter_state__": {
                                "price_per_million": 3.0,
                                "granularity": 1,
                            },
                        },
                        "OutputTokens": {
                            "__llmeter_class__": "llmeter.callbacks.cost.dimensions.OutputTokens",
                            "__llmeter_state__": {
                                "price_per_million": 15.0,
                                "granularity": 1,
                            },
                        },
                    },
                    "run_dims": {},
                },
            },
            {
                "__llmeter_class__": "llmeter.callbacks.mlflow.MlflowCallback",
                "__llmeter_state__": {
                    "step": None,
                    "nested": True,
                },
            },
        ],
        "low_memory": False,
        "progress_bar_stats": None,
    }
    _write_json(out / "run_config.json", run_config)

    # payload.jsonl
    payloads = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize the key concepts of distributed systems.",
                }
            ],
            "max_tokens": 256,
            "model": "synthetic.test-model-v1",
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        {
            "messages": [
                {"role": "user", "content": "Explain the CAP theorem briefly."}
            ],
            "max_tokens": 256,
            "model": "synthetic.test-model-v1",
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ]
    _write_jsonl(out / "payload.jsonl", [json.dumps(p) for p in payloads])

    print(f"  base/ done ({len(responses)} responses)")


# =============================================================================
# Scenario: legacy/v0_1_endpoint_type
# =============================================================================


def generate_legacy_endpoint_type():
    out = FIXTURES_DIR / "legacy" / "v0_1_endpoint_type"

    responses = [
        InvocationResponse(
            id="synth-legacy-001",
            response_text="Consectetur adipiscing elit, sed do eiusmod tempor.",
            input_prompt="What is a load balancer?",
            time_to_first_token=0.612,
            time_to_last_token=1.534,
            num_tokens_input=28,
            num_tokens_output=72,
            time_per_output_token=0.01280,
            error=None,
            request_time=datetime(2025, 1, 10, 14, 0, 0, tzinfo=timezone.utc),
        ),
        InvocationResponse(
            id="synth-legacy-002",
            response_text="Duis aute irure dolor in reprehenderit in voluptate velit.",
            input_prompt="Explain horizontal scaling.",
            time_to_first_token=0.548,
            time_to_last_token=1.289,
            num_tokens_input=24,
            num_tokens_output=65,
            time_per_output_token=0.01140,
            error=None,
            request_time=datetime(2025, 1, 10, 14, 0, 1, tzinfo=timezone.utc),
        ),
        InvocationResponse(
            id="synth-legacy-003",
            response_text="Excepteur sint occaecat cupidatat non proident.",
            input_prompt="What is auto-scaling?",
            time_to_first_token=0.701,
            time_to_last_token=1.678,
            num_tokens_input=22,
            num_tokens_output=58,
            time_per_output_token=0.01684,
            error=None,
            request_time=datetime(2025, 1, 10, 14, 0, 2, tzinfo=timezone.utc),
        ),
        InvocationResponse(
            id="synth-legacy-004",
            response_text="Sunt in culpa qui officia deserunt mollit anim id est laborum.",
            input_prompt="Define microservices architecture.",
            time_to_first_token=0.489,
            time_to_last_token=1.102,
            num_tokens_input=26,
            num_tokens_output=71,
            time_per_output_token=0.00863,
            error=None,
            request_time=datetime(2025, 1, 10, 14, 0, 3, tzinfo=timezone.utc),
        ),
        InvocationResponse(
            id="synth-legacy-005",
            response_text="Neque porro quisquam est qui dolorem ipsum.",
            input_prompt="What is service mesh?",
            time_to_first_token=0.555,
            time_to_last_token=1.345,
            num_tokens_input=20,
            num_tokens_output=48,
            time_per_output_token=0.01646,
            error=None,
            request_time=datetime(2025, 1, 10, 14, 0, 4, tzinfo=timezone.utc),
        ),
    ]

    # summary.json
    summary = {
        "total_requests": 5,
        "clients": 3,
        "n_requests": 2,
        "total_test_time": 5.102,
        "model_id": "synthetic.legacy-model-v1",
        "output_path": "stale/legacy/path",
        "endpoint_name": "synthetic-legacy-endpoint",
        "provider": "bedrock",
        "run_name": "synthetic-legacy-test",
        "run_description": None,
        "start_time": "2025-01-10T14:00:00Z",
        "first_request_time": "2025-01-10T14:00:00Z",
        "last_request_time": "2025-01-10T14:00:04Z",
        "end_time": "2025-01-10T14:00:05Z",
    }
    _write_json(out / "summary.json", summary)

    # stats.json
    _compute_and_write_stats(responses, summary, out / "stats.json")

    # responses.jsonl — legacy format with flat cost annotations as top-level keys
    legacy_responses = []
    costs = [
        {
            "cost_InputTokens": 0.000084,
            "cost_OutputTokens": 0.00108,
            "cost_total": 0.001164,
        },
        {
            "cost_InputTokens": 0.000072,
            "cost_OutputTokens": 0.000975,
            "cost_total": 0.001047,
        },
        {
            "cost_InputTokens": 0.000066,
            "cost_OutputTokens": 0.00087,
            "cost_total": 0.000936,
        },
        {
            "cost_InputTokens": 0.000078,
            "cost_OutputTokens": 0.001065,
            "cost_total": 0.001143,
        },
        {
            "cost_InputTokens": 0.00006,
            "cost_OutputTokens": 0.00072,
            "cost_total": 0.00078,
        },
    ]
    for resp, cost in zip(responses, costs):
        d = json.loads(resp.to_json())
        # Remove modern fields that didn't exist in v0.1
        d.pop("num_tokens_input_cached", None)
        d.pop("num_tokens_output_reasoning", None)
        d.pop("retries", None)
        d.pop("annotations", None)
        # Add legacy flat cost annotations
        d.update(cost)
        legacy_responses.append(json.dumps(d))
    _write_jsonl(out / "responses.jsonl", legacy_responses)

    # run_config.json — legacy format
    run_config = {
        "endpoint": {
            "endpoint_name": "synthetic-legacy-endpoint",
            "model_id": "synthetic.legacy-model-v1",
            "provider": "bedrock",
            "region": "us-west-2",
            "endpoint_type": "BedrockConverse",
        },
        "output_path": "stale/legacy/path",
        "tokenizer": {"tokenizer_module": "llmeter"},
        "clients": 3,
        "n_requests": 2,
        "payload": "stale/legacy/payload.jsonl",
        "run_name": "synthetic-legacy-test",
        "run_description": None,
        "timeout": 60,
        "callbacks": None,
    }
    _write_json(out / "run_config.json", run_config)

    print(f"  legacy/v0_1_endpoint_type/ done ({len(responses)} responses)")


# =============================================================================
# Scenario: legacy/v0_1_str_callbacks
# =============================================================================


def generate_legacy_str_callbacks():
    out = FIXTURES_DIR / "legacy" / "v0_1_str_callbacks"

    responses = [
        InvocationResponse(
            id=f"synth-strcb-{i:03d}",
            response_text=f"Synthetic response number {i} for string callback test.",
            input_prompt=f"Synthetic prompt {i}.",
            time_to_first_token=0.4 + 0.1 * i,
            time_to_last_token=1.0 + 0.2 * i,
            num_tokens_input=30 + 5 * i,
            num_tokens_output=50 + 10 * i,
            time_per_output_token=0.01,
            error=None,
            request_time=datetime(2025, 1, 12, 9, 0, i, tzinfo=timezone.utc),
        )
        for i in range(1, 6)
    ]

    # summary.json
    summary = {
        "total_requests": 5,
        "clients": 1,
        "n_requests": 5,
        "total_test_time": 6.5,
        "model_id": "synthetic.strcb-model-v1",
        "output_path": "stale/strcb/path",
        "endpoint_name": "synthetic-strcb-endpoint",
        "provider": "bedrock",
        "run_name": "synthetic-strcb-test",
        "run_description": None,
        "start_time": "2025-01-12T09:00:01Z",
        "first_request_time": "2025-01-12T09:00:01Z",
        "last_request_time": "2025-01-12T09:00:05Z",
        "end_time": "2025-01-12T09:00:06Z",
    }
    _write_json(out / "summary.json", summary)

    # stats.json
    _compute_and_write_stats(responses, summary, out / "stats.json")

    # responses.jsonl
    _write_jsonl(out / "responses.jsonl", [r.to_json() for r in responses])

    # run_config.json — legacy with string repr callbacks
    run_config = {
        "endpoint": {
            "endpoint_name": "synthetic-strcb-endpoint",
            "model_id": "synthetic.strcb-model-v1",
            "provider": "bedrock",
            "region": "us-east-1",
            "endpoint_type": "BedrockConverseStream",
        },
        "output_path": "stale/strcb/path",
        "tokenizer": {"tokenizer_module": "llmeter"},
        "clients": 1,
        "n_requests": 5,
        "payload": "stale/strcb/payload.jsonl",
        "run_name": "synthetic-strcb-test",
        "run_description": None,
        "timeout": 60,
        "callbacks": [
            "<__main__.SyntheticCallback object at 0x1234567890>",
            "<some_module.AnotherCallback object at 0xabcdef0123>",
        ],
        "low_memory": False,
        "progress_bar_stats": None,
    }
    _write_json(out / "run_config.json", run_config)

    print(f"  legacy/v0_1_str_callbacks/ done ({len(responses)} responses)")


# =============================================================================
# Scenario: interrupted_run (no summary.json)
# =============================================================================


def generate_interrupted_run():
    out = FIXTURES_DIR / "interrupted_run"

    responses = [
        InvocationResponse(
            id=f"synth-intr-{i:03d}",
            response_text=f"Interrupted run response {i}: tempor incididunt ut labore.",
            input_prompt=f"Interrupted test prompt {i}.",
            time_to_first_token=0.3 + 0.05 * i,
            time_to_last_token=0.8 + 0.15 * i,
            num_tokens_input=25 + 3 * i,
            num_tokens_output=40 + 8 * i,
            time_per_output_token=0.009,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 20, 16, 30, i * 2, tzinfo=timezone.utc),
        )
        for i in range(1, 6)
    ]

    # NO summary.json — this is the point of this scenario

    # responses.jsonl
    _write_jsonl(out / "responses.jsonl", [r.to_json() for r in responses])

    # run_config.json — modern __llmeter_class__ format (what a *current* interrupted run leaves)
    run_config = {
        "endpoint": {
            "__llmeter_class__": "llmeter.endpoints.bedrock.BedrockConverseStream",
            "__llmeter_state__": {
                # Mirrors real dump_object() output: fields the constructor accepts, with
                # derived values like `provider` deliberately absent (recomputed on load).
                "model_id": "synthetic.interrupted-model-v1",
                "endpoint_name": "synthetic-interrupted-endpoint",
                "region": "eu-west-1",
                "inference_config": None,
                "max_attempts": 3,
                "ttft_visible_tokens_only": True,
            },
        },
        "output_path": "stale/interrupted/path",
        "tokenizer": {
            "__llmeter_class__": "llmeter.tokenizers.DummyTokenizer",
            "__llmeter_state__": {},
        },
        "clients": 4,
        "n_requests": 20,
        "payload": "stale/interrupted/payload.jsonl",
        "run_name": "synthetic-interrupted-test",
        "run_description": "This run was interrupted before completion",
        "timeout": 120,
        "callbacks": None,
        "low_memory": False,
        "progress_bar_stats": None,
    }
    _write_json(out / "run_config.json", run_config)

    print(f"  interrupted_run/ done ({len(responses)} responses, no summary.json)")


# =============================================================================
# Scenario: legacy/v0_1_interrupted (legacy-format config, no summary.json)
# =============================================================================


def generate_legacy_interrupted_run():
    """A legacy-format interrupted run: no summary.json, top-level endpoint fields.

    Exercises the legacy branch of ``Result._recover_metadata`` - endpoint fields
    (including the derived ``provider``) live at the top level of the endpoint dict,
    and loading should recover them while warning that the format is deprecated.
    """
    out = FIXTURES_DIR / "legacy" / "v0_1_interrupted"

    responses = [
        InvocationResponse(
            id=f"synth-legacy-intr-{i:03d}",
            response_text=f"Legacy interrupted response {i}.",
            input_prompt=f"Legacy interrupted prompt {i}.",
            time_to_first_token=0.35 + 0.05 * i,
            time_to_last_token=0.9 + 0.1 * i,
            num_tokens_input=20 + 4 * i,
            num_tokens_output=45 + 6 * i,
            time_per_output_token=0.01,
            error=None,
            request_time=datetime(2025, 1, 5, 12, 0, i, tzinfo=timezone.utc),
        )
        for i in range(1, 6)
    ]

    # NO summary.json — recovery path

    # responses.jsonl (legacy: no retries/annotations/cached/reasoning fields)
    legacy_responses = []
    for resp in responses:
        d = json.loads(resp.to_json())
        d.pop("num_tokens_input_cached", None)
        d.pop("num_tokens_output_reasoning", None)
        d.pop("retries", None)
        d.pop("annotations", None)
        legacy_responses.append(json.dumps(d))
    _write_jsonl(out / "responses.jsonl", legacy_responses)

    # run_config.json — legacy format with top-level endpoint fields (incl. derived provider)
    run_config = {
        "endpoint": {
            "endpoint_name": "synthetic-legacy-intr-endpoint",
            "model_id": "synthetic.legacy-interrupted-v1",
            "provider": "bedrock",
            "region": "us-east-1",
            "endpoint_type": "BedrockConverse",
        },
        "output_path": "stale/legacy/interrupted/path",
        "tokenizer": {"tokenizer_module": "llmeter"},
        "clients": 2,
        "n_requests": 4,
        "payload": "stale/legacy/interrupted/payload.jsonl",
        "run_name": "synthetic-legacy-interrupted-test",
        "run_description": None,
        "timeout": 60,
        "callbacks": None,
    }
    _write_json(out / "run_config.json", run_config)

    print(
        f"  legacy/v0_1_interrupted/ done ({len(responses)} responses, no summary.json)"
    )


# =============================================================================
# Scenario: errors_and_annotations
# =============================================================================


def generate_errors_and_annotations():
    out = FIXTURES_DIR / "errors_and_annotations"

    responses = [
        InvocationResponse(
            id="synth-err-001",
            response_text="Successful response with custom annotations.",
            input_prompt="Test prompt with annotations.",
            time_to_first_token=0.350,
            time_to_last_token=0.920,
            num_tokens_input=32,
            num_tokens_output=56,
            time_per_output_token=0.01018,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 18, 8, 0, 0, tzinfo=timezone.utc),
            annotations={"custom_metric": 42.5, "experiment_tag": "baseline"},
        ),
        InvocationResponse(
            id="synth-err-002",
            response_text="",
            input_prompt="This request was throttled.",
            time_to_first_token=None,
            time_to_last_token=2.105,
            num_tokens_input=25,
            num_tokens_output=0,
            time_per_output_token=None,
            error="ThrottlingException: Rate exceeded",
            retries=2,
            request_time=datetime(2025, 1, 18, 8, 0, 1, tzinfo=timezone.utc),
            annotations={},
        ),
        InvocationResponse(
            id="synth-err-003",
            response_text="Another successful response after the error.",
            input_prompt="Normal prompt after throttling.",
            time_to_first_token=0.410,
            time_to_last_token=1.050,
            num_tokens_input=30,
            num_tokens_output=62,
            time_per_output_token=0.01032,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 18, 8, 0, 3, tzinfo=timezone.utc),
            annotations={},
        ),
        InvocationResponse(
            id="synth-err-004",
            response_text="",
            input_prompt="This request timed out.",
            time_to_first_token=None,
            time_to_last_token=None,
            num_tokens_input=22,
            num_tokens_output=0,
            time_per_output_token=None,
            error="ReadTimeoutError: timed out after 60s",
            retries=0,
            request_time=datetime(2025, 1, 18, 8, 0, 5, tzinfo=timezone.utc),
            annotations={},
        ),
        InvocationResponse(
            id="synth-err-005",
            response_text="Response with cached input tokens.",
            input_prompt="Prompt that leverages prompt caching.",
            time_to_first_token=0.180,
            time_to_last_token=0.650,
            num_tokens_input=45,
            num_tokens_output=38,
            num_tokens_input_cached=30,
            num_tokens_output_reasoning=None,
            time_per_output_token=0.01237,
            error=None,
            retries=1,
            request_time=datetime(2025, 1, 18, 8, 0, 7, tzinfo=timezone.utc),
            annotations={"cache_hit_ratio": 0.667},
        ),
    ]

    # summary.json
    summary = {
        "total_requests": 5,
        "clients": 1,
        "n_requests": 5,
        "total_test_time": 7.8,
        "model_id": "synthetic.errors-model-v1",
        "output_path": "stale/errors/path",
        "endpoint_name": "synthetic-errors-endpoint",
        "provider": "bedrock",
        "run_name": "synthetic-errors-test",
        "run_description": "Fixture with errors, retries, and annotations",
        "start_time": "2025-01-18T08:00:00Z",
        "first_request_time": "2025-01-18T08:00:00Z",
        "last_request_time": "2025-01-18T08:00:07Z",
        "end_time": "2025-01-18T08:00:07Z",
    }
    _write_json(out / "summary.json", summary)

    # responses.jsonl
    _write_jsonl(out / "responses.jsonl", [r.to_json() for r in responses])

    # stats.json — computed from the responses (including errors)
    _compute_and_write_stats(responses, summary, out / "stats.json")

    print(f"  errors_and_annotations/ done ({len(responses)} responses, 2 errors)")


# =============================================================================
# Scenario: load_test (multi-concurrency)
# =============================================================================


def generate_load_test():
    out = FIXTURES_DIR / "load_test"

    # 00001-clients
    responses_1 = [
        InvocationResponse(
            id=f"synth-lt1-{i:03d}",
            response_text=f"Load test single-client response {i}.",
            input_prompt=f"Load test prompt {i}.",
            time_to_first_token=0.200 + 0.05 * i,
            time_to_last_token=0.600 + 0.1 * i,
            num_tokens_input=20 + 2 * i,
            num_tokens_output=35 + 5 * i,
            time_per_output_token=0.008,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 22, 12, 0, i, tzinfo=timezone.utc),
        )
        for i in range(1, 6)
    ]

    out_1 = out / "00001-clients"
    summary_1 = {
        "total_requests": 5,
        "clients": 1,
        "n_requests": 5,
        "total_test_time": 5.5,
        "model_id": "synthetic.loadtest-model-v1",
        "output_path": "stale/loadtest/00001-clients",
        "endpoint_name": "synthetic-loadtest-endpoint",
        "provider": "bedrock",
        "run_name": "00001-clients",
        "run_description": None,
        "start_time": "2025-01-22T12:00:01Z",
        "first_request_time": "2025-01-22T12:00:01Z",
        "last_request_time": "2025-01-22T12:00:05Z",
        "end_time": "2025-01-22T12:00:05Z",
    }
    _write_json(out_1 / "summary.json", summary_1)
    _compute_and_write_stats(responses_1, summary_1, out_1 / "stats.json")
    _write_jsonl(out_1 / "responses.jsonl", [r.to_json() for r in responses_1])

    # 00003-clients
    responses_3 = [
        InvocationResponse(
            id=f"synth-lt3-{i:03d}",
            response_text=f"Load test triple-client response {i}.",
            input_prompt=f"Load test prompt {i}.",
            time_to_first_token=0.350 + 0.08 * i,
            time_to_last_token=0.900 + 0.15 * i,
            num_tokens_input=20 + 2 * i,
            num_tokens_output=35 + 5 * i,
            time_per_output_token=0.012,
            error=None,
            retries=0,
            request_time=datetime(2025, 1, 22, 12, 5, i, tzinfo=timezone.utc),
        )
        for i in range(1, 6)
    ]

    out_3 = out / "00003-clients"
    summary_3 = {
        "total_requests": 5,
        "clients": 3,
        "n_requests": 2,
        "total_test_time": 3.2,
        "model_id": "synthetic.loadtest-model-v1",
        "output_path": "stale/loadtest/00003-clients",
        "endpoint_name": "synthetic-loadtest-endpoint",
        "provider": "bedrock",
        "run_name": "00003-clients",
        "run_description": None,
        "start_time": "2025-01-22T12:05:01Z",
        "first_request_time": "2025-01-22T12:05:01Z",
        "last_request_time": "2025-01-22T12:05:05Z",
        "end_time": "2025-01-22T12:05:05Z",
    }
    _write_json(out_3 / "summary.json", summary_3)
    _compute_and_write_stats(responses_3, summary_3, out_3 / "stats.json")
    _write_jsonl(out_3 / "responses.jsonl", [r.to_json() for r in responses_3])

    print(
        f"  load_test/ done (2 subdirs, {len(responses_1)}+{len(responses_3)} responses)"
    )


# =============================================================================

if __name__ == "__main__":
    print("Generating fixture data...")
    generate_base()
    generate_legacy_endpoint_type()
    generate_legacy_str_callbacks()
    generate_interrupted_run()
    generate_legacy_interrupted_run()
    generate_errors_and_annotations()
    generate_load_test()
    print("Done!")
