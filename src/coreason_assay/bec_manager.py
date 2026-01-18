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
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from coreason_assay.models import TestCase, TestCaseExpectation, TestCaseInput
from coreason_assay.utils.logger import logger


class BECManager:
    """
    Manager for the Benchmark Evaluation Corpus (BEC).
    Handles ingestion of Test Cases from various formats.
    """

    @staticmethod
    def _parse_json_field(value: Optional[str], field_name: str) -> Any:
        """
        Helper to parse a JSON string field, returning default if empty.
        Supports both empty strings and 'null' as None/Empty.
        """
        if not value or value.strip() == "":
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in field '{field_name}': {e}") from e

    @staticmethod
    def _validate_test_case_data(data: Dict[str, Any], source: str, index: int) -> TestCase:
        """
        Validates and creates a TestCase object from a dictionary.
        Common logic for both JSONL and CSV loaders.
        """
        try:
            return TestCase.model_validate(data)
        except ValidationError as e:
            logger.error(f"Validation error at item {index} in {source}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error creating TestCase at item {index} in {source}: {e}")
            raise e

    @classmethod
    def load_from_jsonl(cls, file_path: Union[str, Path]) -> List[TestCase]:
        """
        Loads Test Cases from a JSONL file.
        Each line in the file must be a valid JSON object matching the TestCase schema.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

        test_cases: List[TestCase] = []

        try:
            with path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        test_case = cls._validate_test_case_data(data, str(path), line_num)
                        test_cases.append(test_case)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON at line {line_num} in {path}: {e}")
                        raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

        except Exception as e:
            if not isinstance(e, (FileNotFoundError, ValueError, ValidationError)):
                logger.exception(f"Unexpected error reading {path}")
            raise e

        logger.info(f"Successfully loaded {len(test_cases)} test cases from {path}")
        return test_cases

    @classmethod
    def load_from_csv(cls, file_path: Union[str, Path]) -> List[TestCase]:
        """
        Loads Test Cases from a CSV file.
        The CSV must have columns mapping to TestCase fields.
        Complex nested fields (lists, dicts) must be JSON-encoded strings.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

        test_cases: List[TestCase] = []

        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, start=1):
                    # 1. Parse Inputs
                    inputs = TestCaseInput(
                        prompt=row.get("prompt", ""),
                        files=cls._parse_json_field(row.get("files"), "files") or [],
                        context=cls._parse_json_field(row.get("context"), "context") or {},
                        tool_outputs=cls._parse_json_field(row.get("tool_outputs"), "tool_outputs") or {},
                    )

                    # 2. Parse Expectations
                    expectations = TestCaseExpectation(
                        text=row.get("expected_text") or None,
                        schema_id=row.get("expected_schema_id") or None,
                        structure=cls._parse_json_field(row.get("expected_structure"), "expected_structure"),
                        reasoning=cls._parse_json_field(row.get("expected_reasoning"), "expected_reasoning") or [],
                        forbidden_content=cls._parse_json_field(row.get("forbidden_content"), "forbidden_content")
                        or [],
                        tool_mocks=cls._parse_json_field(row.get("tool_mocks"), "tool_mocks") or {},
                        tone=None,
                    )

                    # 3. Construct Data Dict for Validation
                    test_case_data: Dict[str, Any] = {
                        "inputs": inputs,
                        "expectations": expectations,
                    }

                    if row.get("id"):
                        test_case_data["id"] = row["id"]
                    if row.get("corpus_id"):
                        test_case_data["corpus_id"] = row["corpus_id"]

                    # 4. Validate
                    try:
                        test_case = cls._validate_test_case_data(test_case_data, str(path), row_num)
                        test_cases.append(test_case)
                    except ValueError as e:
                        # Catch JSON parsing errors from _parse_json_field bubbles
                        logger.error(f"Value error at row {row_num} in {path}: {e}")
                        raise e

        except Exception as e:
            if not isinstance(e, (FileNotFoundError, ValueError, ValidationError)):
                logger.exception(f"Unexpected error reading {path}")
            raise e

        logger.info(f"Successfully loaded {len(test_cases)} test cases from {path}")
        return test_cases

    @classmethod
    def load_from_zip(cls, zip_path: Union[str, Path], target_dir: Union[str, Path]) -> List[TestCase]:
        """
        Loads Test Cases from a ZIP archive.
        The ZIP must contain exactly one manifest file (.csv or .jsonl).
        Assets referenced in the manifest (inputs.files) are checked for existence
        within the extracted directory.
        """
        z_path = Path(zip_path)
        t_dir = Path(target_dir)

        if not z_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {z_path}")

        # 1. Extract ZIP
        try:
            with zipfile.ZipFile(z_path, "r") as zf:
                zf.extractall(t_dir)
        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid ZIP file: {e}") from e

        # 2. Find Manifest
        # Candidates: .csv or .jsonl, excluding macOS artifacts
        candidates = [c for c in list(t_dir.rglob("*.csv")) + list(t_dir.rglob("*.jsonl")) if "__MACOSX" not in c.parts]

        if not candidates:
            raise ValueError("No manifest file (.csv or .jsonl) found in ZIP archive.")

        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous ZIP content: Multiple potential manifest files found: {[c.name for c in candidates]}"
            )

        manifest_path = candidates[0]
        logger.info(f"Found manifest file: {manifest_path}")

        # 3. Load Cases
        if manifest_path.suffix.lower() == ".csv":
            cases = cls.load_from_csv(manifest_path)
        else:
            cases = cls.load_from_jsonl(manifest_path)

        # 4. Resolve and Validate File Paths
        cls._resolve_file_paths(cases, manifest_path.parent, t_dir)

        return cases

    @staticmethod
    def _resolve_file_paths(cases: List[TestCase], manifest_dir: Path, extraction_root: Path) -> None:
        """
        Resolves relative file paths in test cases against the manifest directory.
        Enforces security checks to prevent path traversal outside extraction_root.
        Modifies cases in-place.
        """
        extraction_root_resolved = extraction_root.resolve()

        for case in cases:
            resolved_files = []
            for file_ref in case.inputs.files:
                # Skip URLs
                if "://" in file_ref:
                    resolved_files.append(file_ref)
                    continue

                # Normalize and resolve
                # Replace backslashes for Windows paths compatibility
                clean_ref = Path(file_ref.replace("\\", "/"))
                abs_path = (manifest_dir / clean_ref).resolve()

                # Security Check
                if not abs_path.is_relative_to(extraction_root_resolved):
                    logger.warning(f"File path {file_ref} resolves to {abs_path} outside extraction dir. Rejecting.")
                    raise ValueError(
                        f"Security Error: File path '{file_ref}' attempts to access outside the extraction directory."
                    )

                if not abs_path.exists():
                    raise FileNotFoundError(f"Referenced asset not found in ZIP: {file_ref} (looked at {abs_path})")

                resolved_files.append(str(abs_path))

            case.inputs.files = resolved_files
