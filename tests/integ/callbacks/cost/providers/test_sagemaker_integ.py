# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests (depending on pricing & SageMaker admin APIs) for SageMaker cost models"""

# External Dependencies:
import pytest

# Local Dependencies:
from llmeter.callbacks.cost.providers.sagemaker import (
    SageMakerRTEndpointCompute,
    SageMakerRTEndpointStorage,
)
from llmeter.results import Result


@pytest.mark.asyncio
async def test_real_time_endpoint_compute_auto_price():
    """SageMakerRTEndpointCompute looks up prices from AWS Price List API

    (Test values taken from SageMaker AI pricing page)
    """
    result = Result(
        [],
        total_requests=0,
        clients=0,
        n_requests=0,
        total_test_time=3600,
    )

    dim = SageMakerRTEndpointCompute(
        instance_count=2,
        instance_type="ml.g5.4xlarge",
        region="us-east-1",
    )

    assert dim.price_per_hour == 2.03
    assert await dim.calculate(result) == 4.06

    dim = SageMakerRTEndpointCompute(
        instance_count=2,
        instance_type="ml.g5.8xlarge",
        region="ap-southeast-3",
    )

    assert dim.price_per_hour == 4.2828
    assert await dim.calculate(result) == 8.5656


@pytest.mark.asyncio
async def test_real_time_endpoint_storage_auto_price():
    """SageMakerRTEndpointCompute looks up prices from AWS Price List API

    (Test values taken from SageMaker AI pricing page)
    """
    result = Result(
        [],
        total_requests=0,
        clients=0,
        n_requests=0,
        total_test_time=3600,
    )

    dim = SageMakerRTEndpointStorage(
        gbs_provisioned=40,
        region="us-east-1",
    )

    assert dim.price_per_gb_hour == 0.14 / (30 * 24)
    assert await dim.calculate(result) == 40 * 0.14 / (30 * 24)

    dim = SageMakerRTEndpointStorage(
        gbs_provisioned=10,
        region="ap-southeast-3",
    )

    assert dim.price_per_gb_hour == 0.168 / (30 * 24)
    assert await dim.calculate(result) == 10 * 0.168 / (30 * 24)
