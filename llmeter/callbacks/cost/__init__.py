# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tools for modelling the cost of FM deployments, invocations, and test runs

Cost is an important aspect to consider when comparing between Foundation Models or optimizing
solutions, but can often be complex to fully model: Some pricing dimensions may be applied at the
individual request level (such as tokens in and out, or request duration), while others could be at
the overall deployment level (such as fixed per-hour charges for infrastructure). Pricing
dimensions might differ between different solution options, and other factors like volume discounts
can further complicate comparison.

This model provides configurable tools for modelling the costs associated with individual LLM
requests or overall test runs/sessions. You can include `CostModel` as a *callback* in LLMeter runs
and experiments to annotate cost estimates on your results, or run CostModels on-demand against
existing `InvocationResponse`s or `Result`s to explore different pricing scenarios.
"""

from .model import CostModel  # noqa: F401
from .results import CalculatedCostWithDimensions  # noqa: F401
