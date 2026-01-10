# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import csv
import json
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from coreason_assay.bec_manager import BECManager


class TestBECManagerCSV:
    def test_load_from_csv_valid_simple(self, tmp_path: Any) -> None:
        """Test loading a simple CSV with basic fields."""
        corpus_id = str(uuid4())
        case_id = str(uuid4())
        csv_file = tmp_path / "simple.csv"

        headers = ["id", "corpus_id", "prompt", "expected_text"]
        row = [case_id, corpus_id, "Hello World", "Hi there"]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row)

        cases = BECManager.load_from_csv(csv_file)

        assert len(cases) == 1
        assert cases[0].id == UUID(case_id)
        assert cases[0].corpus_id == UUID(corpus_id)  # Should match value equality
        assert str(cases[0].corpus_id) == corpus_id
        assert cases[0].inputs.prompt == "Hello World"
        assert cases[0].expectations.text == "Hi there"
        # Check defaults
        assert cases[0].inputs.files == []
        assert cases[0].expectations.reasoning == []

    def test_load_from_csv_complex_json_fields(self, tmp_path: Any) -> None:
        """Test loading CSV with JSON encoded fields."""
        corpus_id = str(uuid4())
        csv_file = tmp_path / "complex.csv"

        context = {"role": "admin", "id": 123}
        reasoning = ["Step 1", "Step 2"]
        tool_outputs = {"api_call": {"status": 200}}

        headers = ["corpus_id", "prompt", "context", "expected_reasoning", "tool_outputs"]
        row = [corpus_id, "Complex Prompt", json.dumps(context), json.dumps(reasoning), json.dumps(tool_outputs)]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row)

        cases = BECManager.load_from_csv(csv_file)

        assert len(cases) == 1
        assert cases[0].inputs.context == context
        assert cases[0].expectations.reasoning == reasoning
        assert cases[0].inputs.tool_outputs == tool_outputs

    def test_load_from_csv_invalid_json(self, tmp_path: Any) -> None:
        """Test that invalid JSON in a field raises ValueError."""
        csv_file = tmp_path / "invalid_json.csv"

        headers = ["corpus_id", "prompt", "context"]
        row = [str(uuid4()), "Prompt", "{invalid_json:"]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row)

        with pytest.raises(ValueError, match="Invalid JSON in field 'context'"):
            BECManager.load_from_csv(csv_file)

    def test_load_from_csv_missing_required_field(self, tmp_path: Any) -> None:
        """Test that missing required Pydantic fields (corpus_id) raises ValidationError."""
        csv_file = tmp_path / "missing_req.csv"

        headers = ["prompt", "expected_text"]
        row = ["Prompt", "Text"]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row)

        with pytest.raises(ValidationError):
            BECManager.load_from_csv(csv_file)

    def test_load_from_csv_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            BECManager.load_from_csv("non_existent.csv")

    def test_load_from_csv_unicode(self, tmp_path: Any) -> None:
        """Test Unicode characters in CSV."""
        corpus_id = str(uuid4())
        csv_file = tmp_path / "unicode.csv"

        headers = ["corpus_id", "prompt", "expected_text"]
        row = [corpus_id, "你好", "Hello 🌍"]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row)

        cases = BECManager.load_from_csv(csv_file)
        assert cases[0].inputs.prompt == "你好"
        assert cases[0].expectations.text == "Hello 🌍"

    def test_load_from_csv_unexpected_error(self, tmp_path: Any) -> None:
        """Test unexpected exception handling."""
        csv_file = tmp_path / "test.csv"
        csv_file.touch()

        # Mock open or csv.DictReader to raise a generic Exception
        with patch("csv.DictReader", side_effect=Exception("Unexpected boom")):
            with pytest.raises(Exception, match="Unexpected boom"):
                BECManager.load_from_csv(csv_file)
