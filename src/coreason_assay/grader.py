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
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from jsonschema import SchemaError, ValidationError, validate

from coreason_assay.interfaces import LLMClient
from coreason_assay.models import Score, TestCaseInput, TestResult
from coreason_assay.utils.logger import logger


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


class ReasoningGrader(BaseGrader):
    """
    Grades whether the agent followed the expected reasoning steps.
    Uses an LLMClient to evaluate the execution trace.
    """

    PROMPT_TEMPLATE = """You are an expert evaluator of AI reasoning chains.
Your task is to verify if the actual execution trace of an AI agent contains specific required reasoning steps.

Required Reasoning Steps:
{required_steps}

Actual Execution Trace:
{trace}

(Fallback) Actual Output Text:
{text}

Instructions:
1. Analyze the trace (and text if trace is insufficient) to find evidence of each required step.
2. Return a JSON object with the following structure:
{{
  "steps_analysis": [
    {{"step": "step description", "found": true/false, "evidence": "quote or explanation"}},
    ...
  ],
  "score": <float between 0.0 and 1.0 representing percentage of steps found>
}}

Return ONLY the JSON.
"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
    ) -> Score:
        # Check for reasoning expectations
        required_steps = expectations.get("reasoning") if expectations else None

        if not required_steps:
            return Score(
                name="ReasoningAlignment",
                value=1.0,
                passed=True,
                reasoning="No reasoning expectations provided.",
            )

        trace = result.actual_output.trace or ""
        text = result.actual_output.text or ""

        # Format steps list
        formatted_steps = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(required_steps)])

        prompt = self.PROMPT_TEMPLATE.format(
            required_steps=formatted_steps,
            trace=trace,
            text=text,
        )

        try:
            response_text = self.llm_client.complete(prompt)
            # Try to parse JSON from the response
            # Sometimes LLMs wrap JSON in ```json ... ```
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            analysis = json.loads(cleaned_response)

            score_val = float(analysis.get("score", 0.0))
            steps_analysis = analysis.get("steps_analysis", [])

            # Construct detailed reasoning from analysis
            details = []
            for item in steps_analysis:
                status = "✅" if item.get("found") else "❌"
                step = item.get("step")
                evidence = item.get("evidence", "No evidence")
                details.append(f"{status} {step}: {evidence}")

            reasoning_text = "\n".join(details)
            if not reasoning_text:
                reasoning_text = "LLM provided no detailed analysis."

            # Pass if score is 1.0 (or very close)
            # Or should we allow partial pass? Usually QC is strict.
            # But PRD says "Score: 50% (Correct Answer, Invalid Process)".
            # If the requirement is "Did it follow steps?", then if it missed steps, it failed "Reasoning Alignment"?
            # However, the Score object has a `passed` boolean.
            # If score < 1.0, passed = False seems appropriate for "Alignment".
            passed = score_val >= 0.99  # Allow some float error, but basically 100%

            return Score(
                name="ReasoningAlignment",
                value=score_val,
                max_value=1.0,
                passed=passed,
                reasoning=reasoning_text,
            )

        except Exception as e:
            logger.error(f"Error in ReasoningGrader: {e}")
            return Score(
                name="ReasoningAlignment",
                value=0.0,
                passed=False,
                reasoning=f"Grading failed due to internal error: {str(e)}",
            )
