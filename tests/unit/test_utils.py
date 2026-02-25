# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from llmeter.utils import DeferredError, summary_stats_from_list


def test_deferred_error_on_import():
    # Simulate the import attempt
    try:
        import obscurelib  # type: ignore
    except ImportError as e:
        obscurelib = DeferredError(e)

    # Verify that obscurelib is now a DeferredError instance
    assert isinstance(obscurelib, DeferredError)

    # Attempt to use the 'obscurelib'
    with pytest.raises(ImportError) as exc_info:
        obscurelib.some_function()

    # Verify that the original ImportError is raised
    assert "No module named 'obscurelib'" in str(exc_info.value)

    # Try to access an attribute
    with pytest.raises(ImportError) as exc_info:
        _ = obscurelib.some_attribute

    # Verify that the original ImportError is raised
    assert "No module named 'obscurelib'" in str(exc_info.value)


def test_summary_stats_from_list():
    EPSILON = 1e-15  # Numerical precision for testing

    # Empty list
    assert summary_stats_from_list([]) == {}
    assert summary_stats_from_list([float("nan"), float("nan")]) == {}

    assert summary_stats_from_list([1, 2, 3], percentiles=[]) == {"average": 2}
    assert summary_stats_from_list([0], percentiles=[]) == {"average": 0}
    assert summary_stats_from_list([42]) == {
        "p50": 42,
        "p90": 42,
        "p99": 42,
        "average": 42,
    }

    assert summary_stats_from_list([3, 1, 2], percentiles=[50])["p50"] == 2
    assert summary_stats_from_list([4, 1, 3, 2], percentiles=[50])["p50"] == 2.5
    assert (
        abs(
            summary_stats_from_list([0.4, 0.3, 0.2, 0.1], percentiles=[75])["p75"]
            - 0.375
        )
        < EPSILON
    )

    assert summary_stats_from_list([1.03127], percentiles=[50])["p50"] == 1.03127

    # Why, you might ask? Great question - take it up with:
    # https://docs.python.org/3/library/statistics.html#statistics.quantiles
    assert summary_stats_from_list([0, 1], percentiles=[83])["p83"] == 1.49

    assert summary_stats_from_list([1, 2, 3, 4, 5]) == {
        "p50": 3,
        "p90": 5.4,
        "p99": 5.94,
        "average": 3,
    }
    assert summary_stats_from_list([-5, -3, -1, 0, 2, 4]) == {
        "p50": -0.5,
        "p90": 4.6,
        "p99": 5.86,
        "average": -0.5,
    }
    assert summary_stats_from_list(list(range(1, 100))) == {
        "average": 50,
        "p50": 50,
        "p90": 90,
        "p99": 99,
    }

    # NaN values should be ignored:
    assert summary_stats_from_list([1, 2, float("nan"), 4, 5]) == {
        "p50": 3,
        "p90": 5.5,
        "p99": 5.95,
        "average": 3,
    }
