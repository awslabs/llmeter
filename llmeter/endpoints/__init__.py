# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configure your target target LLM servers to test with LLMeter"""

import importlib.util

from .base import Endpoint, InvocationResponse  # noqa: F401
from .bedrock import BedrockConverse, BedrockConverseStream  # noqa: F401
from .bedrock_invoke import BedrockInvoke, BedrockInvokeStream  # noqa: F401
from .sagemaker import SageMakerEndpoint, SageMakerStreamEndpoint  # noqa: F401

spec = importlib.util.find_spec("openai")
if spec:
    from .openai import (  # noqa: F401
        OpenAIEndpoint,
        OpenAICompletionEndpoint,
        OpenAICompletionStreamEndpoint,
    )
    from .openai_response import (  # noqa: F401
        ResponseEndpoint,
        ResponseStreamEndpoint,
    )

spec = importlib.util.find_spec("litellm")
if spec:
    from .litellm import LiteLLM, LiteLLMStreaming  # noqa: F401
