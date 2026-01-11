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

from coreason_assay.models import Score, TestResult


class BaseGrader(ABC):
    """
    Abstract base class for all Graders.
    A Grader evaluates a TestResult and produces a Score.
    """

    @abstractmethod
    def grade(self, result: TestResult, expectations: Optional[Dict[str, Any]] = None) -> Score:
        """
        Evaluate the result and return a Score.

        Args:
            result: The result of the test run.
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

    def grade(self, result: TestResult, expectations: Optional[Dict[str, Any]] = None) -> Score:
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
    Grades whether the output matches the expected structure.
    For this iteration, checks if structured_output is present and valid.
    """

    def grade(self, result: TestResult, expectations: Optional[Dict[str, Any]] = None) -> Score:
        structured_output = result.actual_output.structured_output

        # If we expect structure (from global expectations or TestCase expectations)
        # Note: In a real integration, we'd pass the TestCase's expectations here.
        # For now, we assume if this grader is called, structure is desired.

        if structured_output is None:
            return Score(
                name="JsonSchema",
                value=0,
                passed=False,
                reasoning="No structured output produced.",
            )

        # In a full implementation, we would validate against a schema.
        # Here we check if the expected keys (if provided) exist.
        expected_structure = expectations.get("structure") if expectations else None

        if expected_structure:
            if not isinstance(structured_output, dict):
                return Score(
                    name="JsonSchema",
                    value=0,
                    passed=False,
                    reasoning=f"Expected a dictionary for structured output, got {type(structured_output).__name__}.",
                )

            # Simple key presence check for this iteration
            missing_keys = [k for k in expected_structure.keys() if k not in structured_output]
            if missing_keys:
                return Score(
                    name="JsonSchema",
                    value=0,
                    passed=False,
                    reasoning=f"Missing keys in structured output: {missing_keys}",
                )

        return Score(
            name="JsonSchema",
            value=1,
            passed=True,
            reasoning="Structured output present and matches expectations.",
        )
