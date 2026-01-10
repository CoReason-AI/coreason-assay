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
from pathlib import Path
from typing import List, Union

from pydantic import ValidationError

from coreason_assay.models import TestCase
from coreason_assay.utils.logger import logger


class BECManager:
    """
    Manager for the Benchmark Evaluation Corpus (BEC).
    Handles ingestion of Test Cases from various formats.
    """

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
