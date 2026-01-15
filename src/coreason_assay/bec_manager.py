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
        """Helper to parse a JSON string field, returning default if empty."""
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in field '{field_name}': {e}") from e

    @staticmethod
    def load_from_jsonl(file_path: Union[str, Path]) -> List[TestCase]:
        """
        Loads Test Cases from a JSONL file.
        Each line in the file must be a valid JSON object matching the TestCase schema.

        Args:
            file_path: Path to the .jsonl file.

        Returns:
            List[TestCase]: A list of validated TestCase objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If a line contains invalid JSON.
            ValidationError: If a JSON object does not match the TestCase schema.
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
                        test_case = TestCase.model_validate(data)
                        test_cases.append(test_case)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON at line {line_num} in {path}: {e}")
                        raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e
                    except ValidationError as e:
                        logger.error(f"Validation error at line {line_num} in {path}: {e}")
                        raise e

        except Exception as e:
            # Re-raise known exceptions, log unknown ones if necessary
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

        Expected columns (examples):
        - id, corpus_id (UUIDs)
        - prompt (str)
        - files (JSON list of strings)
        - context (JSON dict)
        - tool_outputs (JSON dict)
        - expected_text (str)
        - expected_schema_id (str)
        - expected_structure (JSON dict)
        - expected_reasoning (JSON list of strings)
        - forbidden_content (JSON list of strings)
        - tool_mocks (JSON dict)

        Args:
            file_path: Path to the .csv file.

        Returns:
            List[TestCase]: A list of validated TestCase objects.
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
                    try:
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

                        # 3. Construct TestCase
                        # If ID or corpus_id are missing, let Pydantic handle validation
                        # (it might generate defaults or raise error)
                        # NOTE: TestCase.id has a default factory, but corpus_id is required.
                        test_case_data: Dict[str, Any] = {
                            "inputs": inputs,
                            "expectations": expectations,
                        }

                        if row.get("id"):
                            test_case_data["id"] = row["id"]
                        if row.get("corpus_id"):
                            test_case_data["corpus_id"] = row["corpus_id"]

                        test_case = TestCase.model_validate(test_case_data)
                        test_cases.append(test_case)

                    except ValidationError as e:
                        logger.error(f"Validation error at row {row_num} in {path}: {e}")
                        raise e
                    except ValueError as e:
                        # Catch JSON parsing errors
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

        Args:
            zip_path: Path to the .zip file.
            target_dir: Directory where the zip contents will be extracted.

        Returns:
            List[TestCase]: A list of validated TestCase objects with absolute file paths.

        Raises:
            FileNotFoundError: If zip_path does not exist or referenced assets are missing.
            ValueError: If manifest is missing, ambiguous, or invalid.
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
        # We look for files ending in .csv or .jsonl in the root of the extracted dir
        # We generally expect the manifest to be at the top level.
        # We search recursively but fail if multiple candidates are found to avoid ambiguity.
        candidates = list(t_dir.rglob("*.csv")) + list(t_dir.rglob("*.jsonl"))

        # Filter out __MACOSX artifacts if any
        candidates = [c for c in candidates if "__MACOSX" not in c.parts]

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
        # The manifest file is at `manifest_path`. Relative paths in the manifest
        # should generally be relative to the manifest's directory.
        manifest_dir = manifest_path.parent

        for case in cases:
            resolved_files = []
            for file_ref in case.inputs.files:
                # If it's a URL (s3://, http://), leave it alone
                if "://" in file_ref:
                    resolved_files.append(file_ref)
                    continue

                # Assume local relative path
                # Normalize separators for cross-platform compatibility (Windows manifest on Linux)
                file_ref_normalized = file_ref.replace("\\", "/")
                clean_ref = Path(file_ref_normalized)

                # Construct absolute path
                abs_path = (manifest_dir / clean_ref).resolve()

                # Security check: prevent path traversal out of target_dir
                try:
                    abs_path.relative_to(t_dir.resolve())
                except ValueError:
                    logger.warning(
                        f"File path {file_ref} resolves to {abs_path} which is outside extraction dir {t_dir}. Rejecting."  # noqa: E501
                    )
                    raise ValueError(
                        f"Security Error: File path '{file_ref}' attempts to access outside the extraction directory."
                    ) from None

                if not abs_path.exists():
                    raise FileNotFoundError(f"Referenced asset not found in ZIP: {file_ref} (looked at {abs_path})")

                resolved_files.append(str(abs_path))

            # Update the case with resolved paths
            case.inputs.files = resolved_files

        return cases
