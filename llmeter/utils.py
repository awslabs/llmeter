# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from itertools import filterfalse
from math import isnan
from statistics import StatisticsError, mean, median, quantiles
from typing import Sequence


class DeferredError:
    """Stores an exception and raises it at a later time if this object is accessed in any way.

    Useful to allow soft-dependencies on imports, so that the ImportError can be raised again
    later if code actually relies on the missing library.
    Lifted from https://github.com/aws/sagemaker-python-sdk/blob/e626647692d136155d72cf5b100043af41ab6c43/src/sagemaker/utils.py#L788

    Example::

        try:
            import obscurelib
        except ImportError as e:
            logger.warning("Failed to import obscurelib. Obscure features will not work.")
            obscurelib = DeferredError(e)
    """

    def __init__(self, exception):
        self.exc = exception

    def __getattr__(self, name):
        """Called by Python interpreter before using any method or property on the object.

        So this will short-circuit essentially any access to this object.

        Args:
            name:
        """
        raise self.exc


def summary_stats_from_list(
    data: Sequence[int | float],
    percentiles: Sequence[int] = (50, 90, 99),
) -> dict[str, int | float]:
    """Calculate summary statistics for a sequence of numbers in pure Python

    Args:
        data: Sequence of numbers
        percentiles: Integer percentiles from 1-99
    Returns:
        stats: Dictionary of descriptive statistics including "average" (the arithmetic mean of the
            input `data`), and the requested `percentiles` in format "p50", "p90", "p99", etc.
    """
    clean_data = list(filterfalse(isnan, data))
    try:
        result = dict(average=mean(clean_data))
        # Rather than always using n=100-based quantiles, we'll adapt between a few specific bases
        # depending on the requested percentiles - and calculate these splits only as needed:
        qbases = {k: None for k in [4, 10, 100]}
        for p in percentiles:
            if len(clean_data) == 1:
                result[f"p{p}"] = clean_data[0]
            elif p == 50:
                result[f"p{p}"] = median(clean_data)
            else:
                try:
                    qbase = next(k for k in qbases if p % (100 / k) == 0)
                except StopIteration as e:
                    raise ValueError(
                        f"Invalid percentile {p} must be a whole number between 1-99"
                    ) from e
                if qbases[qbase] is None:
                    qbases[qbase] = quantiles(clean_data, n=qbase)
                result[f"p{p}"] = qbases[qbase][int(p * qbase / 100) - 1]
        return result

    except StatisticsError:
        return {}
