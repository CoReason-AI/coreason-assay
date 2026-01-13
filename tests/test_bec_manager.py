# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import json
from typing import Any, Dict
from unittest.mock import patch
from uuid import uuid4

import pytest
from coreason_assay.bec_manager import BECManager
from coreason_assay.models import TestCase
from pydantic import ValidationError


class TestBECManager:
    @pytest.fixture
    def valid_test_case_dict(self) -> Dict[str, Any]:
        return {
            "id": str(uuid4()),
            "corpus_id": str(uuid4()),
            "inputs": {
                "prompt": "Test Prompt",
                "files": [],
                "context": {},
                "tool_outputs": {},
            },
            "expectations": {
                "text": "Expected Output",
                "schema_id": None,
                "structure": None,
                "reasoning": [],
                "forbidden_content": [],
                "tool_mocks": {},
            },
        }

    def test_load_from_jsonl_valid(self, tmp_path: Any, valid_test_case_dict: Dict[str, Any]) -> None:
        # Create a valid JSONL file
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps(valid_test_case_dict) + "\n")
            # Write a second case slightly modified
            valid_test_case_dict["inputs"]["prompt"] = "Second Prompt"
            f.write(json.dumps(valid_test_case_dict) + "\n")

        cases = BECManager.load_from_jsonl(file_path)

        assert len(cases) == 2
        assert isinstance(cases[0], TestCase)
        assert cases[0].inputs.prompt == "Test Prompt"
        assert cases[1].inputs.prompt == "Second Prompt"

    def test_load_from_jsonl_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            BECManager.load_from_jsonl("non_existent_file.jsonl")

    def test_load_from_jsonl_invalid_json(self, tmp_path: Any, valid_test_case_dict: Dict[str, Any]) -> None:
        file_path = tmp_path / "invalid.jsonl"
        with open(file_path, "w") as f:
            # Valid first line
            f.write(json.dumps(valid_test_case_dict) + "\n")
            # Invalid JSON second line
            f.write("INVALID JSON HERE\n")

        with pytest.raises(ValueError, match="Invalid JSON at line 2"):
            BECManager.load_from_jsonl(file_path)

    def test_load_from_jsonl_validation_error(self, tmp_path: Any) -> None:
        file_path = tmp_path / "schema_invalid.jsonl"
        with open(file_path, "w") as f:
            # Missing required field 'corpus_id'
            invalid_data = {
                "inputs": {"prompt": "Hi"},
                "expectations": {},
            }
            f.write(json.dumps(invalid_data) + "\n")

        with pytest.raises(ValidationError):
            BECManager.load_from_jsonl(file_path)

    def test_load_from_jsonl_empty_lines(self, tmp_path: Any, valid_test_case_dict: Dict[str, Any]) -> None:
        file_path = tmp_path / "empty_lines.jsonl"
        with open(file_path, "w") as f:
            f.write("\n")
            f.write(json.dumps(valid_test_case_dict) + "\n")
            f.write("\n")

        cases = BECManager.load_from_jsonl(file_path)
        assert len(cases) == 1
        assert cases[0].inputs.prompt == "Test Prompt"

    def test_load_from_jsonl_unexpected_error(self, tmp_path: Any) -> None:
        """Test unexpected exception handling (coverage)."""
        file_path = tmp_path / "test.jsonl"
        file_path.touch()

        # Mock open to raise a generic Exception
        with patch("pathlib.Path.open", side_effect=Exception("Unexpected boom")):
            with pytest.raises(Exception, match="Unexpected boom"):
                BECManager.load_from_jsonl(file_path)

    def test_load_from_jsonl_complex_nested_unicode(self, tmp_path: Any, valid_test_case_dict: Dict[str, Any]) -> None:
        """
        Test that deeply nested structures and Unicode characters are handled correctly.
        """
        complex_context = {
            "user": {
                "profile": {"name": "Jules 🧙‍♂️", "roles": ["admin", "wizard"]},
                "history": [{"id": 1, "action": "login"}, {"id": 2, "action": "cast_spell"}],
            },
            "meta": {"source": "S3", "version": 1.0},
        }

        valid_test_case_dict["inputs"]["context"] = complex_context
        valid_test_case_dict["inputs"]["prompt"] = "Translate: 你好, World! 🌍"

        file_path = tmp_path / "complex.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_test_case_dict, ensure_ascii=False) + "\n")

        cases = BECManager.load_from_jsonl(file_path)
        assert len(cases) == 1
        loaded_case = cases[0]

        # Verify Unicode
        assert loaded_case.inputs.prompt == "Translate: 你好, World! 🌍"
        assert loaded_case.inputs.context["user"]["profile"]["name"] == "Jules 🧙‍♂️"

        # Verify Nesting
        assert loaded_case.inputs.context["user"]["history"][1]["action"] == "cast_spell"

    def test_load_from_jsonl_extra_fields(self, tmp_path: Any, valid_test_case_dict: Dict[str, Any]) -> None:
        """
        Test that extra fields in the JSON are ignored (resilience).
        """
        valid_test_case_dict["inputs"]["extra_field"] = "Should be ignored"
        valid_test_case_dict["unknown_root_field"] = 12345

        file_path = tmp_path / "extra_fields.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps(valid_test_case_dict) + "\n")

        cases = BECManager.load_from_jsonl(file_path)
        assert len(cases) == 1
        # Pydantic (by default) ignores extra fields, so this should not raise ValidationError
        # and the extra fields should not be present in the model if we inspected `model_dump()`.

        dumped = cases[0].model_dump()
        assert "unknown_root_field" not in dumped
        # Note: TestCaseInput might store extra fields if configured to Allow, but default is Ignore.
        # Let's verify inputs
        inputs_dump = cases[0].inputs.model_dump()
        assert "extra_field" not in inputs_dump

    def test_load_from_jsonl_explicit_nulls(self, tmp_path: Any, valid_test_case_dict: Dict[str, Any]) -> None:
        """
        Test that explicit 'null' values in JSON are correctly handled for Optional fields.
        """
        # Set optional fields to null
        valid_test_case_dict["expectations"]["text"] = None
        valid_test_case_dict["expectations"]["schema_id"] = None

        file_path = tmp_path / "nulls.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps(valid_test_case_dict) + "\n")

        cases = BECManager.load_from_jsonl(file_path)
        assert len(cases) == 1
        expectations = cases[0].expectations
        assert expectations.text is None
        assert expectations.schema_id is None
