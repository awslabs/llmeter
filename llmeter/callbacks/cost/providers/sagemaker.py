# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for automating cost modelling on Amazon SageMaker endpoints"""

# Python Built-Ins:
from __future__ import annotations
from dataclasses import dataclass
import json
from math import ceil

# External Dependencies:
import boto3

# Local Dependencies:
from ....results import Result
from ..dimensions import RunCostDimensionBase
from ..model import CostModel


def _get_default_boto_region() -> str:
    """Look up the current default boto region"""
    return boto3.Session().region_name


def _fetch_single_product_ondemand_unit_price(
    service_code: str = "AmazonSageMaker",
    region: str | None = None,
    attribute_values: dict[str, str] | None = None,
    dim_unit: str | None = None,
    currency: str = "USD",
) -> tuple[float, dict]:
    """Fetch a specific USD price per unit from AWS Price List API, or else throw an error

    Use this method to query the AWS Price List API with filters that are expected to return:
    - Exactly one matching 'Product', which...
    - Offers exactly one 'OnDemand' pricing term, which...
    - Includes exactly one pricing dimension matching `dim_unit`, which...
    - Offers a `pricePerUnit` in the target `currency`

    Args:
        service_code: ServiceCode per the AWS Price List API
        region: If provided, this will be used as a filter when matching Products
        attribute_values: Mapping from product attribute/field name to expected value, for matching
            Products. The `region` is automatically added as a filter when provided.
        dim_unit: Case-insensitive `unit` string to find the correct pricing dimension in the
            `OnDemand` pricing term of the retrieved Product. If not set, will default if the
            product has exactly one dimension - or throw an error otherwise.
        currency: Currency identifier (typically 'USD') to fetch the final pricePerUnit.

    Returns:
        price_per_unit: The price per unit in the specified currency
        product: The full product object from the Price List API response, for reference

    Raises:
        ValueError: If zero or more than one Products were matched by the service code and filter
            criteria; or if the product did not have exactly one OnDemand pricing term; or if the
            relevant pricing dimension could not be identified; or if the selected `currency` was
            not offered in the pricing term.
    """
    # Pre-process arguments:
    if not attribute_values:
        attribute_values = {}
    if dim_unit:
        dim_unit = dim_unit.lower()

    # TERM_MATCH is the *only* supported match type, so we can offer our simplified external API:
    filters = [
        {"Field": k, "Type": "TERM_MATCH", "Value": v}
        for k, v in attribute_values.items()
    ]
    if region:
        filters.append({"Field": "regionCode", "Type": "TERM_MATCH", "Value": region})

    pricing = boto3.client("pricing", region_name="us-east-1")
    products = []
    for page in pricing.get_paginator("get_products").paginate(
        ServiceCode=service_code,
        Filters=filters,
    ):
        products += [json.loads(p) for p in page["PriceList"]]
    if len(products) != 1:
        raise ValueError(
            f"Expected 1 matching product from pricing API, got {len(products)}:\n{products}"
        )
    product = products[0]
    if "OnDemand" not in product["terms"]:
        raise ValueError(
            f"Expected pricing terms to include 'OnDemand' option. Got:\n{product['terms']}"
        )
    od_terms: dict = product["terms"]["OnDemand"]
    if len(od_terms.keys()) != 1:
        raise ValueError(
            f"Expected exactly 1 OnDemand term from pricing API. Got {len(od_terms)}:\n{od_terms}"
        )
    od_term = next(t for t in od_terms.values())
    hourly_dims: list[dict] = [
        v
        for v in od_term["priceDimensions"].values()
        if v.get("unit").lower() == dim_unit
    ]
    if len(hourly_dims) != 1:
        raise ValueError(
            "Expected exactly 1 price dimension with unit='%s'. Got %s:\n%s"
            % (dim_unit, len(hourly_dims), hourly_dims)
        )
    prices_by_curr = hourly_dims[0].get("pricePerUnit")
    if currency not in prices_by_curr:
        raise ValueError(
            f"Expected price per unit to include {currency}. Got:\n{prices_by_curr}"
        )
    return float(prices_by_curr[currency]), product


def cost_model_from_sagemaker_realtime_endpoint(
    endpoint_name: str, region: str | None = None
) -> CostModel:
    """Automatically infer an LLMeter CostModel from a deployed SageMaker real-time endpoint

    This method builds a basic cost estimating model for SageMaker real-time inference endpoints
    including compute and EBS storage costs, but excluding data transfer costs. Standard on-demand
    pricing is used, without accounting for private pricing, tiers, savings plans, or etc.

    NOTE: You'll need IAM permissions to `pricing:GetProducts`, `sagemaker:DescribeEndpoint`, and
    `sagemaker:DescribeEndpointConfig` to use this method.

    Args:
        endpoint_name: Name of the deployed SageMaker endpoint
        region: AWS region where the endpoint is running (default: current region)

    Returns:
        cost_model: A `CostModel` instance with inferred cost dimensions capturing instance compute
            and EBS storage. If multiple "variants" are deployed on the endpoint, separate
            dimensions will be created for each one with variant name as a prefix.
    """
    return CostModel(
        run_dims={
            **SageMakerRTEndpointCompute.from_endpoint(endpoint_name, region=region),
            **SageMakerRTEndpointStorage.from_endpoint(endpoint_name, region=region),
        }
    )


@dataclass
class SageMakerRTEndpointCompute(RunCostDimensionBase):
    """Run cost dimension to estimate Amazon SageMaker real-time endpoint compute charges

    NOTE: To auto-discover `price_per_hour` from instance type, you'll need IAM permissions to call
    the `pricing:GetProducts` API. For more information, see:
    https://docs.aws.amazon.com/service-authorization/latest/reference/list_awspricelist.html

    Calculated rates **do not currently include** EBS storage volume costs. See
    `SageMakerRTEndpointStorage` for estimating this.

    See `.fetch_sm_hosting_od_price()` for more details on how price is looked up when not
    explicitly provided. This lookup is provided on a best-effort basis, and may not accurately
    reflect all possible scenarios.

    Use `.from_endpoint()` to automatically discover compute cost dimensions from a deployed
    SageMaker real-time inference endpoint.

    Args:
        instance_type: Amazon SageMaker instance type e.g. 'ml.g5.4xlarge'
        instance_count: Number of instances running (default: 1)
        price_per_hour: Price per hour per instance (default: attempt to fetch from pricing API)
        region: AWS region where the endpoint is running (default: current region)
    """

    instance_type: str
    instance_count: float = 1
    price_per_hour: float | None = None
    region: str | None = None

    def __post_init__(self):
        if not self.region:
            self.region = _get_default_boto_region()
        if self.price_per_hour is None:
            self.price_per_hour = self.fetch_sm_hosting_od_price(
                self.instance_type, self.region
            )

    async def calculate(self, result: Result) -> float | None:
        if result.total_test_time is None:
            return None
        return (
            ceil(result.total_test_time)
            * self.instance_count
            * self.price_per_hour
            / 3600
        )

    @classmethod
    def from_endpoint(
        cls, endpoint_name: str, region: str | None = None
    ) -> dict[str, SageMakerRTEndpointCompute]:
        """Configure SageMakerRTEndpointCompute dimension(s) from an existing SageMaker endpoint

        NOTE: You'll need IAM permissions to `sagemaker:DescribeEndpoint` and
        `sagemaker:DescribeEndpointConfig` to use this method.

        This function returns a dictionary rather than a single `SageMakerRTEndpointCompute`,
        because different "variants" deployed behind an endpoint may be backed by clusters of
        different instance types, and therefore need separate dimensions.

        Instance counts will be retrieved at the point in time this method is called, so watch out
        if you have auto-scaling enabled on your endpoint.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            region: AWS region where the endpoint is running (default: current region)

        Returns:
            run_dims: A dictionary containing one or more dimensions, to pass to your
                `CostModel(run_dims=...)`. If the endpoint has only one "Variant", the returned
                dict will have a single key (cost dimension name) `SageMakerRTEndpointCompute`.
                Otherwise, keys will be generated as `{variant_name}SageMakerRTEndpointCompute`
        """
        sagemaker = boto3.client("sagemaker", region_name=region)
        endpoint_desc = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        ep_vars: dict[str, dict] = {
            e["VariantName"]: e for e in endpoint_desc["ProductionVariants"]
        }
        endpoint_config = sagemaker.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )
        cfg_vars: dict[str, dict] = {
            e["VariantName"]: e for e in endpoint_config["ProductionVariants"]
        }

        if len(ep_vars) == 1:
            variant = next(iter(ep_vars.values()))
            variant_config = cfg_vars[variant["VariantName"]]
            return {
                cls.__name__: cls(
                    instance_type=variant_config["InstanceType"],
                    instance_count=variant["CurrentInstanceCount"],
                    region=sagemaker.meta.region_name,
                )
            }
        else:
            return {
                f"{var_name}_{cls.__name__}": cls(
                    instance_type=cfg_vars[var_name]["InstanceType"],
                    instance_count=ep_vars[var_name]["CurrentInstanceCount"],
                    region=sagemaker.meta.region_name,
                )
                for var_name in ep_vars.keys()
            }

    @staticmethod
    def fetch_sm_hosting_od_price(instance_type: str, region: str) -> float:
        """Look up USD hourly rates for on-demand SM hosting instances from the AWS Price List API

        This function assumes:

        1. You're using standard "on-demand" pricing - no savings plans or private pricing
        2. No free tier or volume discounts are applicable to this usage
        3. Your pricing is provided in USD

        Args:
            instance_type: Amazon SageMaker instance type, e.g. 'ml.g5.4xlarge'
            region: AWS region where the endpoint is running, e.g. 'us-east-1'

        Returns:
            price_per_hour: The standard, on-demand hourly price for the given instance type in USD
        """
        unit_price, _ = _fetch_single_product_ondemand_unit_price(
            service_code="AmazonSageMaker",
            attribute_values={"component": "Hosting", "instanceName": instance_type},
            region=region,
            dim_unit="Hrs",
        )
        return unit_price


@dataclass
class SageMakerRTEndpointStorage(RunCostDimensionBase):
    """Run cost dimension to estimate EBS charges for Amazon SageMaker real-time endpoints

    NOTE: To auto-discover `price_per_gb_hour`, you'll need IAM permissions to call the
    `pricing:GetProducts` API. For more information, see:
    https://docs.aws.amazon.com/service-authorization/latest/reference/list_awspricelist.html

    See `.fetch_sm_hosting_ebs_price()` for more details on how price is looked up when not
    explicitly provided. This lookup is provided on a best-effort basis, and may not accurately
    reflect all possible scenarios.

    Args:
        gbs_provisioned: Total size of provisioned EBS volume(s) for the endpoint in Gigabytes
        price_per_gb_hour: Price per hour per GB (default: attempt to fetch from pricing API)
        region: AWS region where the endpoint is running (default: current region)
    """

    gbs_provisioned: float
    price_per_gb_hour: float | None = None
    region: str | None = None

    def __post_init__(self):
        if not self.region:
            self.region = _get_default_boto_region()
        if self.price_per_gb_hour is None:
            self.price_per_gb_hour = self.fetch_sm_hosting_ebs_price(self.region)

    async def calculate(self, result: Result) -> float | None:
        if result.total_test_time is None:
            return None
        return (
            ceil(result.total_test_time)
            * self.gbs_provisioned
            * self.price_per_gb_hour
            / 3600
        )

    @classmethod
    def from_endpoint(
        cls,
        endpoint_name: str,
        region: str | None = None,
        merge_variants: bool = False,
    ) -> dict[str, SageMakerRTEndpointStorage]:
        """Configure SageMakerRTEndpointStorage dimension(s) from an existing SageMaker endpoint

        NOTE: You'll need IAM permissions to `sagemaker:DescribeEndpoint` and
        `sagemaker:DescribeEndpointConfig` to use this method.

        This function returns a dictionary rather than a single `SageMakerRTEndpointStorage`,
        because different "variants" deployed behind an endpoint may be reported separately, if
        more than one are present.

        Instance counts will be retrieved at the point in time this method is called, so watch out
        if you have auto-scaling enabled on your endpoint.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            region: AWS region where the endpoint is running (default: current region)
            merge_variants: Set `True` to merge multiple "variants" into a single storage dimension
                (since the hourly rate is constant across instance types). By default, if multiple
                variants are configured these will be reported as separate cost dimensions.

        Returns:
            run_dims: A dictionary containing zero or more dimensions, to pass to your
                `CostModel(run_dims=...)`. If the endpoint has only one "Variant", the returned
                dict will have a single key (cost dimension name) `SageMakerRTEndpointCompute`.
                Otherwise, keys will be generated as `{variant_name}SageMakerRTEndpointCompute`.
                Any variants with 0 EBS storage configured will be omitted from the result.
        """
        sagemaker = boto3.client("sagemaker", region_name=region)
        region_final = sagemaker.meta.region_name
        endpoint_desc = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        ep_vars: dict[str, dict] = {
            e["VariantName"]: e for e in endpoint_desc["ProductionVariants"]
        }
        endpoint_config = sagemaker.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )
        cfg_vars: dict[str, dict] = {
            e["VariantName"]: e for e in endpoint_config["ProductionVariants"]
        }

        # variant.VolumeSizeInGB seems to be set automatically for EBS-enabled instance types, but
        # missing for non-EBS (storage-included) instance types.
        vars_with_ebs = [
            name for name, cfg in cfg_vars.items() if cfg.get("VolumeSizeInGB")
        ]

        if len(vars_with_ebs) == 0:
            return {}
        elif len(ep_vars) == 1:
            variant = next(iter(ep_vars.values()))
            variant_config = cfg_vars[variant["VariantName"]]
            return {
                cls.__name__: cls(
                    gbs_provisioned=variant_config["VolumeSizeInGB"]
                    * variant["CurrentInstanceCount"],
                    region=region_final,
                )
            }
        elif merge_variants:
            total_gbs = sum(
                [
                    cfg_vars[var_name].get("VolumeSizeInGB", 0)
                    * ep_vars[var_name]["CurrentInstanceCount"]
                    for var_name in ep_vars.keys()
                ]
            )
            return {cls.__name__: cls(gbs_provisioned=total_gbs, region=region_final)}
        else:
            return {
                f"{var_name}_{cls.__name__}": cls(
                    gbs_provisioned=cfg_vars[var_name]["VolumeSizeInGB"]
                    * ep_vars[var_name]["CurrentInstanceCount"],
                    region=region_final,
                )
                for var_name in ep_vars.keys()
                if var_name in vars_with_ebs
            }

    @staticmethod
    def fetch_sm_hosting_ebs_price(region: str) -> float:
        """Look up hourly USD rate for SageMaker hosting EBS storage from the AWS Price List API

        The API actually lists this rate as monthly, so we take an assumption of 30days * 24hrs to
        convert.

        Args:
            region: AWS region where the endpoint is running, e.g. 'us-east-1'

        Returns:
            price_per_gb_hour: The standard, on-demand price per GB-hour in USD
        """
        unit_price, _ = _fetch_single_product_ondemand_unit_price(
            service_code="AmazonSageMaker",
            attribute_values={"volumeType": "General Purpose-Hosting"},
            region=region,
            dim_unit="GB-Mo",
        )

        return unit_price / (30 * 24)  # Convert GB-mo to GB-hr
