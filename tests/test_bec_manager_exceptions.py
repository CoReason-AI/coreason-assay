# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from unittest.mock import patch

import pytest
from coreason_assay.bec_manager import BECManager


def test_validate_test_case_data_exception_handling() -> None:
    """
    Test that _validate_test_case_data re-raises unexpected exceptions
    after logging them.
    """
    # We mock TestCase.model_validate to raise a generic Exception
    with patch("coreason_assay.bec_manager.TestCase.model_validate") as mock_validate:
        mock_validate.side_effect = Exception("Unexpected error")

        with pytest.raises(Exception, match="Unexpected error"):
            BECManager._validate_test_case_data({}, "source", 1)


def test_load_from_jsonl_exception_handling() -> None:
    """Test exception handling in load_from_jsonl for unexpected errors."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.open", side_effect=Exception("Disk error")):
            with pytest.raises(Exception, match="Disk error"):
                BECManager.load_from_jsonl("dummy.jsonl")


def test_load_from_csv_exception_handling() -> None:
    """Test exception handling in load_from_csv for unexpected errors."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.open", side_effect=Exception("Disk error")):
            with pytest.raises(Exception, match="Disk error"):
                BECManager.load_from_csv("dummy.csv")
