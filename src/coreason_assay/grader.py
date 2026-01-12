# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from jsonschema import SchemaError, ValidationError, validate

from coreason_assay.models import Score, TestCaseInput, TestResult


class BaseGrader(ABC):
    """
    Abstract base class for all Graders.
    A Grader evaluates a TestResult and produces a Score.
    """

    @abstractmethod
    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
    ) -> Score:
        """
        Evaluate the result and return a Score.

        Args:
            result: The result of the test run.
            inputs: Optional TestCaseInput containing the original request (for context-aware grading).
            expectations: Optional dictionary of specific expectations for this grader
                          (e.g., specific schema ID, custom threshold).
                          If not provided, the grader may use defaults or data from result.
        """
        pass  # pragma: no cover


class LatencyGrader(BaseGrader):
    """
    Grades the execution latency against a threshold.
    """

    def __init__(self, threshold_ms: float = 5000.0):
        self.default_threshold_ms = threshold_ms

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
    ) -> Score:
        latency = result.metrics.get("latency_ms")
        if latency is None:
            return Score(
                name="Latency",
                value=0,
                passed=False,
                reasoning="Latency metric missing from execution result.",
            )

        threshold = self.default_threshold_ms
        if expectations and "latency_threshold_ms" in expectations:
            threshold = expectations["latency_threshold_ms"]

        passed = latency <= threshold
        return Score(
            name="Latency",
            value=latency,
            max_value=threshold,
            passed=passed,
            reasoning=f"Latency {latency:.2f}ms is {'within' if passed else 'exceeds'} threshold of {threshold}ms.",
        )


class JsonSchemaGrader(BaseGrader):
    """
    Grades whether the output matches the expected JSON schema.
    """

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
    ) -> Score:
        structured_output = result.actual_output.structured_output

        if structured_output is None:
            return Score(
                name="JsonSchema",
                value=0,
                passed=False,
                reasoning="No structured output produced.",
            )

        # Retrieve the expected schema from expectations
        # We look for 'structure' as the schema definition
        expected_schema = expectations.get("structure") if expectations else None

        if not expected_schema:
            # If no schema is provided, we default to passing if structured_output exists
            # This matches the previous behavior where we just checked presence if no expectations
            return Score(
                name="JsonSchema",
                value=1,
                passed=True,
                reasoning="Structured output present (no schema provided for validation).",
            )

        try:
            validate(instance=structured_output, schema=expected_schema)
        except ValidationError as e:
            return Score(
                name="JsonSchema",
                value=0,
                passed=False,
                reasoning=f"Validation failed: {e.message}",
            )
        except SchemaError as e:
            return Score(
                name="JsonSchema",
                value=0,
                passed=False,
                reasoning=f"Invalid JSON Schema provided in expectations: {e.message}",
            )

        return Score(
            name="JsonSchema",
            value=1,
            passed=True,
            reasoning="Structured output matches the expected JSON schema.",
        )
