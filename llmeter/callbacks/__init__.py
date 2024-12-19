# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Customize and extend LLMeter with callback hooks

Callbacks provide an API through which you can trigger custom code at defined points in the test
Run (or Experiment) lifecycle and augment the output results or statistics. Some built-in examples
include:

- Logging run details to MLFlow, with `MlflowCallback`
- Calculating cost estimates, with `CostModel`

For creating your own custom callbacks, see the `Callback` base class.
"""

import importlib.util

from .base import Callback  # noqa: F401
from .cost import CostModel  # noqa: F401

spec = importlib.util.find_spec("mlflow")
if spec:
    from .mlflow import MlflowCallback  # noqa: F401
