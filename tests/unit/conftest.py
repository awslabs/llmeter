# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for unit tests."""

from pathlib import Path

import pytest

RESULT_SNAPSHOTS_DIR = Path(__file__).parent / "fixtures" / "result_snapshots"


@pytest.fixture
def snapshots_dir() -> Path:
    """Base directory holding the static synthetic result snapshots."""
    return RESULT_SNAPSHOTS_DIR
