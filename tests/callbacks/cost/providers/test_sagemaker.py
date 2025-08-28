# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SageMaker-specific cost modelling utilities"""

# Python Built-Ins:
from unittest.mock import Mock, call, patch

# External Dependencies:
import boto3
import pytest
from moto import mock_aws

# Local Dependencies:
from llmeter.callbacks.cost.providers.sagemaker import (
    SageMakerRTEndpointCompute,
    SageMakerRTEndpointStorage,
)
from llmeter.results import Result


@pytest.mark.asyncio
async def test_real_time_endpoint_compute_explicit_price():
    dim = SageMakerRTEndpointCompute(
        instance_type="ml.doesnotexist",
        price_per_hour=30,
        region="us-nope-0",
    )

    result = Result(
        [],
        total_requests=0,
        clients=0,
        n_requests=0,
        total_test_time=60,
    )
    assert await dim.calculate(result) == 30 * 60 / 3600
    dim.price_per_hour = 6
    assert await dim.calculate(result) == 6 * 60 / 3600
    result.total_test_time = 120
    assert await dim.calculate(result) == 6 * 120 / 3600
    result.total_test_time = 0
    assert await dim.calculate(result) == 0
    result.total_test_time = None
    assert await dim.calculate(result) is None

    dim_ser = dim.to_dict()
    assert dim_ser == {
        "_type": "SageMakerRTEndpointCompute",
        "instance_count": 1,
        "instance_type": "ml.doesnotexist",
        "price_per_hour": 6,
        "region": "us-nope-0",
    }

    dim = SageMakerRTEndpointCompute.from_dict(dim_ser)
    result.total_test_time = 90
    assert await dim.calculate(result) == 6 * 90 / 3600


@pytest.mark.asyncio
async def test_real_time_endpoint_compute_missing_price():
    """SageMakerRTEndpointCompute tries to look up prices if not explicitly provided

    NOTE: moto doesn't support 'pricing' API yet, so this test just mocks out the whole
    `fetch_sm_hosting_od_price()` function. See `tests/integ` folder for an integration test that
    actually exercises it (but depends on having AWS credentials configured)
    """
    result = Result(
        [],
        total_requests=0,
        clients=0,
        n_requests=0,
        total_test_time=3600,
    )

    lookup_price_mock = Mock(return_value=42)
    with patch.object(
        SageMakerRTEndpointCompute,
        "fetch_sm_hosting_od_price",
        new=lookup_price_mock,
    ):
        dim = SageMakerRTEndpointCompute(
            instance_type="ml.g5.4xlarge", region="us-east-1"
        )

    lookup_price_mock.assert_called_once_with("ml.g5.4xlarge", "us-east-1")
    assert dim.price_per_hour == 42
    assert await dim.calculate(result) == 42


@pytest.mark.asyncio
async def test_compute_dimensions_from_endpoint_single_variant():
    """SageMakerRTEndpointCompute.from_endpoint handles single-variant endpoints"""
    # Price List API isn't in moto, so also need to mock out this:
    lookup_price_mock = Mock(return_value=9)
    with mock_aws(), patch.object(
        SageMakerRTEndpointCompute,
        "fetch_sm_hosting_od_price",
        new=lookup_price_mock,
    ):
        # Set up dummy environment in moto:
        sagemaker = boto3.client("sagemaker")
        sagemaker.create_model(
            ModelName="dummy-model",
            PrimaryContainer={
                "Image": "123456789012.dkr.ecr.ap-southeast-3.amazonaws.com/dummy-image:dummy-label",
                "ModelDataUrl": "s3://dummy-bucket/model.tar.gz",
            },
            ExecutionRoleArn="arn:aws:iam::123456789012:role/DummyRole",
        )
        sagemaker.create_endpoint_config(
            EndpointConfigName="dummy-endpoint-config",
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "dummy-model",
                    "InstanceType": "ml.g4dn.12xlarge",
                    "InitialInstanceCount": 1,
                }
            ],
        )
        sagemaker.create_endpoint(
            EndpointName="dummy-endpoint",
            EndpointConfigName="dummy-endpoint-config",
        )

        # Check cost dimension creation:
        cost_dims = SageMakerRTEndpointCompute.from_endpoint("dummy-endpoint")
        assert isinstance(cost_dims, dict)
        assert "SageMakerRTEndpointCompute" in cost_dims
        assert len(cost_dims) == 1
        cost_dim = cost_dims["SageMakerRTEndpointCompute"]
        assert cost_dim.instance_type == "ml.g4dn.12xlarge"
        assert cost_dim.instance_count == 1
        lookup_price_mock.assert_called_once_with(
            "ml.g4dn.12xlarge", sagemaker.meta.region_name
        )
        assert cost_dim.price_per_hour == lookup_price_mock.return_value
        assert cost_dim.region == sagemaker.meta.region_name


@pytest.mark.asyncio
async def test_compute_dimensions_from_endpoint_multi_variant():
    """SageMakerRTEndpointCompute.from_endpoint handles multi-variant endpoints"""
    # Price List API isn't in moto, so also need to mock out this:
    lookup_price_mock = Mock(return_value=9)
    with mock_aws(), patch.object(
        SageMakerRTEndpointCompute,
        "fetch_sm_hosting_od_price",
        new=lookup_price_mock,
    ):
        # Set up dummy environment in moto:
        sagemaker = boto3.client("sagemaker")
        sagemaker.create_model(
            ModelName="dummy-model",
            PrimaryContainer={
                "Image": "123456789012.dkr.ecr.ap-southeast-3.amazonaws.com/dummy-image:dummy-label",
                "ModelDataUrl": "s3://dummy-bucket/model.tar.gz",
            },
            ExecutionRoleArn="arn:aws:iam::123456789012:role/DummyRole",
        )
        sagemaker.create_endpoint_config(
            EndpointConfigName="dummy-endpoint-config",
            ProductionVariants=[
                {
                    "VariantName": "Blue",
                    "ModelName": "dummy-model",
                    "InstanceType": "ml.g4dn.12xlarge",
                    "InitialInstanceCount": 1,
                },
                {
                    "VariantName": "Green",
                    "ModelName": "dummy-model",
                    "InstanceType": "ml.g4dn.8xlarge",
                    "InitialInstanceCount": 1,
                },
            ],
        )
        sagemaker.create_endpoint(
            EndpointName="dummy-endpoint",
            EndpointConfigName="dummy-endpoint-config",
        )

        # Check cost dimension creation:
        cost_dims = SageMakerRTEndpointCompute.from_endpoint("dummy-endpoint")
        assert isinstance(cost_dims, dict)
        assert "Blue_SageMakerRTEndpointCompute" in cost_dims
        assert "Green_SageMakerRTEndpointCompute" in cost_dims
        assert len(cost_dims) == 2
        blue_dim = cost_dims["Blue_SageMakerRTEndpointCompute"]
        assert blue_dim.instance_type == "ml.g4dn.12xlarge"
        assert blue_dim.instance_count == 1
        assert blue_dim.price_per_hour == lookup_price_mock.return_value
        assert blue_dim.region == sagemaker.meta.region_name
        green_dim = cost_dims["Green_SageMakerRTEndpointCompute"]
        assert green_dim.instance_type == "ml.g4dn.8xlarge"
        assert green_dim.instance_count == 1
        assert green_dim.price_per_hour == lookup_price_mock.return_value
        assert green_dim.region == sagemaker.meta.region_name

        lookup_price_mock.assert_has_calls(
            [
                call("ml.g4dn.12xlarge", sagemaker.meta.region_name),
                call("ml.g4dn.8xlarge", sagemaker.meta.region_name),
            ]
        )


@pytest.mark.asyncio
async def test_real_time_endpoint_storage_explicit_price():
    dim = SageMakerRTEndpointStorage(
        gbs_provisioned=5,
        price_per_gb_hour=0.9,
        region="us-nope-0",
    )

    result = Result(
        [],
        total_requests=0,
        clients=0,
        n_requests=0,
        total_test_time=60,
    )
    assert await dim.calculate(result) == 0.9 * 5 * 60 / 3600
    dim.price_per_gb_hour = 9
    assert await dim.calculate(result) == 9 * 5 * 60 / 3600
    result.total_test_time = 120
    assert await dim.calculate(result) == 9 * 5 * 120 / 3600
    result.total_test_time = 0
    assert await dim.calculate(result) == 0
    result.total_test_time = None
    assert await dim.calculate(result) is None

    dim_ser = dim.to_dict()
    assert dim_ser == {
        "_type": "SageMakerRTEndpointStorage",
        "gbs_provisioned": 5,
        "price_per_gb_hour": 9,
        "region": "us-nope-0",
    }

    dim = SageMakerRTEndpointStorage.from_dict(dim_ser)
    result.total_test_time = 90
    assert await dim.calculate(result) == 9 * 5 * 90 / 3600


@pytest.mark.asyncio
async def test_real_time_endpoint_storage_missing_price():
    """SageMakerRTEndpointStorage tries to look up prices if not explicitly provided

    NOTE: moto doesn't support 'pricing' API yet, so this test just mocks out the whole
    `fetch_sm_hosting_od_price()` function. See `tests/integ` folder for an integration test that
    actually exercises it (but depends on having AWS credentials configured)
    """
    result = Result(
        [],
        total_requests=0,
        clients=0,
        n_requests=0,
        total_test_time=3600,
    )

    lookup_price_mock = Mock(return_value=42)
    with patch.object(
        SageMakerRTEndpointStorage,
        "fetch_sm_hosting_ebs_price",
        new=lookup_price_mock,
    ):
        dim = SageMakerRTEndpointStorage(gbs_provisioned=4, region="us-east-1")

    lookup_price_mock.assert_called_once_with("us-east-1")
    assert dim.price_per_gb_hour == 42
    assert await dim.calculate(result) == 4 * 42


@pytest.mark.asyncio
async def test_storage_dimensions_from_endpoint_single_variant():
    """SageMakerRTEndpointStorage.from_endpoint handles single-variant endpoints"""
    # Price List API isn't in moto, so also need to mock out this:
    lookup_price_mock = Mock(return_value=9)
    with mock_aws(), patch.object(
        SageMakerRTEndpointStorage,
        "fetch_sm_hosting_ebs_price",
        new=lookup_price_mock,
    ):
        # Set up dummy environment in moto:
        sagemaker = boto3.client("sagemaker")
        sagemaker.create_model(
            ModelName="dummy-model",
            PrimaryContainer={
                "Image": "123456789012.dkr.ecr.ap-southeast-3.amazonaws.com/dummy-image:dummy-label",
                "ModelDataUrl": "s3://dummy-bucket/model.tar.gz",
            },
            ExecutionRoleArn="arn:aws:iam::123456789012:role/DummyRole",
        )
        sagemaker.create_endpoint_config(
            EndpointConfigName="dummy-endpoint-config",
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "dummy-model",
                    "InstanceType": "ml.g4dn.12xlarge",
                    "InitialInstanceCount": 1,
                    "VolumeSizeInGB": 50,
                }
            ],
        )
        sagemaker.create_endpoint(
            EndpointName="dummy-endpoint",
            EndpointConfigName="dummy-endpoint-config",
        )

        # Check cost dimension creation:
        cost_dims = SageMakerRTEndpointStorage.from_endpoint("dummy-endpoint")
        assert isinstance(cost_dims, dict)
        assert "SageMakerRTEndpointStorage" in cost_dims
        assert len(cost_dims) == 1
        cost_dim = cost_dims["SageMakerRTEndpointStorage"]
        assert cost_dim.gbs_provisioned == 50
        lookup_price_mock.assert_called_once_with(sagemaker.meta.region_name)
        assert cost_dim.price_per_gb_hour == lookup_price_mock.return_value
        assert cost_dim.region == sagemaker.meta.region_name


@pytest.mark.asyncio
async def test_storage_dimensions_from_endpoint_no_ebs():
    """SageMakerRTEndpointStorage.from_endpoint handles endpoints with no EBS storage

    Instances like ml.g5.4xlarge (which have instance-attached storage) don't get EBS
    """
    # Price List API isn't in moto, so also need to mock out this:
    lookup_price_mock = Mock(return_value=9)
    with mock_aws(), patch.object(
        SageMakerRTEndpointStorage,
        "fetch_sm_hosting_ebs_price",
        new=lookup_price_mock,
    ):
        # Set up dummy environment in moto:
        sagemaker = boto3.client("sagemaker")
        sagemaker.create_model(
            ModelName="dummy-model",
            PrimaryContainer={
                "Image": "123456789012.dkr.ecr.ap-southeast-3.amazonaws.com/dummy-image:dummy-label",
                "ModelDataUrl": "s3://dummy-bucket/model.tar.gz",
            },
            ExecutionRoleArn="arn:aws:iam::123456789012:role/DummyRole",
        )
        sagemaker.create_endpoint_config(
            EndpointConfigName="dummy-endpoint-config",
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "dummy-model",
                    "InstanceType": "ml.g4dn.12xlarge",
                    "InitialInstanceCount": 1,
                }
            ],
        )
        sagemaker.create_endpoint(
            EndpointName="dummy-endpoint",
            EndpointConfigName="dummy-endpoint-config",
        )

        # Check cost dimension creation:
        cost_dims = SageMakerRTEndpointStorage.from_endpoint("dummy-endpoint")
        assert cost_dims == {}


@pytest.mark.asyncio
async def test_storage_dimensions_from_endpoint_multi_variant():
    """SageMakerRTEndpointStorage.from_endpoint handles multi-variant endpoints"""
    # Price List API isn't in moto, so also need to mock out this:
    lookup_price_mock = Mock(return_value=9)
    with mock_aws(), patch.object(
        SageMakerRTEndpointStorage,
        "fetch_sm_hosting_ebs_price",
        new=lookup_price_mock,
    ):
        # Set up dummy environment in moto:
        sagemaker = boto3.client("sagemaker")
        sagemaker.create_model(
            ModelName="dummy-model",
            PrimaryContainer={
                "Image": "123456789012.dkr.ecr.ap-southeast-3.amazonaws.com/dummy-image:dummy-label",
                "ModelDataUrl": "s3://dummy-bucket/model.tar.gz",
            },
            ExecutionRoleArn="arn:aws:iam::123456789012:role/DummyRole",
        )
        sagemaker.create_endpoint_config(
            EndpointConfigName="dummy-endpoint-config",
            ProductionVariants=[
                {
                    "VariantName": "Blue",
                    "ModelName": "dummy-model",
                    "InstanceType": "ml.g4dn.12xlarge",
                    "InitialInstanceCount": 1,
                    "VolumeSizeInGB": 9,
                },
                {
                    "VariantName": "Green",
                    "ModelName": "dummy-model",
                    "InstanceType": "ml.g4dn.8xlarge",
                    "InitialInstanceCount": 1,
                    "VolumeSizeInGB": 6,
                },
            ],
        )
        sagemaker.create_endpoint(
            EndpointName="dummy-endpoint",
            EndpointConfigName="dummy-endpoint-config",
        )

        # Check cost dimension creation:
        cost_dims = SageMakerRTEndpointStorage.from_endpoint("dummy-endpoint")
        assert isinstance(cost_dims, dict)
        assert "Blue_SageMakerRTEndpointStorage" in cost_dims
        assert "Green_SageMakerRTEndpointStorage" in cost_dims
        assert len(cost_dims) == 2
        blue_dim = cost_dims["Blue_SageMakerRTEndpointStorage"]
        assert blue_dim.gbs_provisioned == 9
        assert blue_dim.price_per_gb_hour == lookup_price_mock.return_value
        assert blue_dim.region == sagemaker.meta.region_name
        green_dim = cost_dims["Green_SageMakerRTEndpointStorage"]
        assert green_dim.gbs_provisioned == 6
        assert green_dim.price_per_gb_hour == lookup_price_mock.return_value
        assert green_dim.region == sagemaker.meta.region_name

        lookup_price_mock.assert_has_calls(
            [call(sagemaker.meta.region_name), call(sagemaker.meta.region_name)]
        )


@pytest.mark.asyncio
async def test_storage_dimensions_from_endpoint_multi_variant_partial_ebs():
    """SageMakerRTEndpointStorage.from_endpoint handles multi-variant endpoints with partial EBS

    When only one variant has EBS, the returned dimension should still be explicitly labelled with
    which variant yields it.
    """
    # Price List API isn't in moto, so also need to mock out this:
    lookup_price_mock = Mock(return_value=9)
    with mock_aws(), patch.object(
        SageMakerRTEndpointStorage,
        "fetch_sm_hosting_ebs_price",
        new=lookup_price_mock,
    ):
        # Set up dummy environment in moto:
        sagemaker = boto3.client("sagemaker")
        sagemaker.create_model(
            ModelName="dummy-model",
            PrimaryContainer={
                "Image": "123456789012.dkr.ecr.ap-southeast-3.amazonaws.com/dummy-image:dummy-label",
                "ModelDataUrl": "s3://dummy-bucket/model.tar.gz",
            },
            ExecutionRoleArn="arn:aws:iam::123456789012:role/DummyRole",
        )
        sagemaker.create_endpoint_config(
            EndpointConfigName="dummy-endpoint-config",
            ProductionVariants=[
                {
                    "VariantName": "Blue",
                    "ModelName": "dummy-model",
                    "InstanceType": "ml.g4dn.12xlarge",
                    "InitialInstanceCount": 1,
                    "VolumeSizeInGB": 9,
                },
                {
                    "VariantName": "Green",
                    "ModelName": "dummy-model",
                    "InstanceType": "ml.g4dn.8xlarge",
                    "InitialInstanceCount": 1,
                },
            ],
        )
        sagemaker.create_endpoint(
            EndpointName="dummy-endpoint",
            EndpointConfigName="dummy-endpoint-config",
        )

        # Check cost dimension creation:
        cost_dims = SageMakerRTEndpointStorage.from_endpoint("dummy-endpoint")
        assert isinstance(cost_dims, dict)
        assert "Blue_SageMakerRTEndpointStorage" in cost_dims
        assert len(cost_dims) == 1
        blue_dim = cost_dims["Blue_SageMakerRTEndpointStorage"]
        assert blue_dim.gbs_provisioned == 9
        assert blue_dim.price_per_gb_hour == lookup_price_mock.return_value
        assert blue_dim.region == sagemaker.meta.region_name

        lookup_price_mock.assert_has_calls([call(sagemaker.meta.region_name)])
