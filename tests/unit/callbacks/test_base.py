# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from llmeter.callbacks.base import Callback


class TestBase:
    def test__load_from_file_not_implemented(self):
        """
        Test that _load_from_file raises NotImplementedError.
        """
        with pytest.raises(NotImplementedError):
            Callback._load_from_file("valid_path.json")

    def test_load_from_file_not_implemented(self):
        """
        Test that load_from_file raises a NotImplementedError as it's not yet implemented.
        """
        with pytest.raises(NotImplementedError):
            Callback.load_from_file("valid_path.txt")

    def test_load_from_file_raises_not_implemented_error(self):
        """
        Test that Callback.load_from_file raises a NotImplementedError.

        This test verifies that calling the static method load_from_file
        on the Callback class raises a NotImplementedError, as the method
        is not yet implemented.
        """
        with pytest.raises(NotImplementedError) as excinfo:
            Callback.load_from_file("dummy_path")

        assert (
            str(excinfo.value)
            == "TODO: Callback.load_from_file is not yet implemented!"
        )

    def test_load_from_file_raises_not_implemented_error_2(self):
        """
        Test that _load_from_file raises NotImplementedError when called.
        This method is not yet implemented in the base Callback class.
        """
        with pytest.raises(NotImplementedError) as excinfo:
            Callback._load_from_file("dummy_path")

        assert (
            str(excinfo.value)
            == "TODO: Callback._load_from_file is not yet implemented!"
        )
