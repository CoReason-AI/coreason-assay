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

from coreason_manifest.definitions.agent import AgentDefinition
from jsonschema import SchemaError, ValidationError, validate

from coreason_assay.interfaces import LLMClient
from coreason_assay.models import Score, TestCaseInput, TestResult
from coreason_assay.prompts import FAITHFULNESS_GRADER_PROMPT, REASONING_GRADER_PROMPT, TONE_GRADER_PROMPT
from coreason_assay.utils.logger import logger
from coreason_assay.utils.parsing import parse_json_from_llm_response


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
        agent: Optional[AgentDefinition] = None,
    ) -> Score:
        """
        Evaluate the result and return a Score.

        Args:
            result: The result of the test run.
            inputs: Optional TestCaseInput containing the original request (for context-aware grading).
            expectations: Optional dictionary of specific expectations for this grader
                          (e.g., specific schema ID, custom threshold).
                          If not provided, the grader may use defaults or data from result.
            agent: Optional AgentDefinition of the agent being tested.
        """
        pass  # pragma: no cover


class LLMGrader(BaseGrader):
    """
    Base class for graders that utilize an LLMClient for evaluation.
    Provides utility methods for prompt execution and JSON parsing.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def _get_llm_analysis(self, prompt: str) -> Dict[str, Any]:
        """
        Executes the prompt via the LLM client and parses the JSON response.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            Dict[str, Any]: The parsed JSON analysis.

        Raises:
            Exception: If LLM call fails or JSON parsing error occurs.
        """
        response_text = self.llm_client.complete(prompt)
        return parse_json_from_llm_response(response_text)


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
        agent: Optional[AgentDefinition] = None,
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
        agent: Optional[AgentDefinition] = None,
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


class ReasoningGrader(LLMGrader):
    """
    Grades whether the agent followed the expected reasoning steps.
    Uses an LLMClient to evaluate the execution trace.
    """

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
        agent: Optional[AgentDefinition] = None,
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

        trace_obj = result.actual_output.trace
        # Convert structured trace to string for LLM consumption
        trace_str = trace_obj.model_dump_json(indent=2) if trace_obj else ""
        text = result.actual_output.text or ""

        # Format steps list
        formatted_steps = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(required_steps)])

        # Use Template substitution
        prompt = REASONING_GRADER_PROMPT.safe_substitute(REQUIRED_STEPS=formatted_steps, TRACE=trace_str, TEXT=text)

        try:
            analysis = self._get_llm_analysis(prompt)

            score_val_raw = analysis.get("score", 0.0)
            try:
                score_val = float(score_val_raw)
            except (ValueError, TypeError):
                # Handle cases where score might be "100%" or non-numeric garbage
                # If it's a string ending in %, strip it
                if isinstance(score_val_raw, str) and score_val_raw.endswith("%"):
                    try:
                        score_val = float(score_val_raw.rstrip("%")) / 100.0
                    except ValueError:
                        score_val = 0.0
                else:
                    score_val = 0.0

            steps_analysis = analysis.get("steps_analysis", [])

            # Construct detailed reasoning from analysis
            details = []
            for item in steps_analysis:
                # Handle string "true"/"false" if LLM returns strings
                found_raw = item.get("found")
                if isinstance(found_raw, str):
                    is_found = found_raw.lower() == "true"
                else:
                    is_found = bool(found_raw)

                status = "✅" if is_found else "❌"
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


class ForbiddenContentGrader(BaseGrader):
    """
    Grades whether the agent's output contains any forbidden content.
    """

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
        agent: Optional[AgentDefinition] = None,
    ) -> Score:
        forbidden_list = expectations.get("forbidden_content") if expectations else None

        if not forbidden_list:
            return Score(
                name="ForbiddenContent",
                value=1.0,
                passed=True,
                reasoning="No forbidden content specified in expectations.",
            )

        text = result.actual_output.text
        if not text:
            # If there is no text output, we cannot check for forbidden content.
            # Technically, if there's no output, there's no forbidden content.
            return Score(
                name="ForbiddenContent",
                value=1.0,
                passed=True,
                reasoning="No text output to check for forbidden content.",
            )

        found_terms = []
        text_lower = text.lower()

        for term in forbidden_list:
            if not term:
                continue
            if term.lower() in text_lower:
                found_terms.append(term)

        if found_terms:
            return Score(
                name="ForbiddenContent",
                value=0.0,
                passed=False,
                reasoning=f"Found forbidden content: {', '.join([f'{t!r}' for t in found_terms])}",
            )

        return Score(
            name="ForbiddenContent",
            value=1.0,
            passed=True,
            reasoning="None of the forbidden terms were found in the output.",
        )


class FaithfulnessGrader(LLMGrader):
    """
    Grades whether the agent's answer is faithful to the provided context.
    Uses an LLMClient to detect hallucinations or contradictions.
    """

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
        agent: Optional[AgentDefinition] = None,
    ) -> Score:
        # Extract context
        context_str = ""
        if inputs:
            # We look for context in inputs.context
            # We convert the whole dict to a string representation
            if inputs.context:
                context_str = json.dumps(inputs.context, indent=2)

        if not context_str:
            return Score(
                name="Faithfulness",
                value=0.0,
                passed=False,
                reasoning="No context provided in inputs to verify against.",
            )

        answer_text = result.actual_output.text or ""
        if not answer_text:
            return Score(
                name="Faithfulness",
                value=0.0,
                passed=False,
                reasoning="No answer text produced by agent.",
            )

        # Use Template substitution
        prompt = FAITHFULNESS_GRADER_PROMPT.safe_substitute(CONTEXT=context_str, ANSWER=answer_text)

        try:
            analysis = self._get_llm_analysis(prompt)

            is_faithful = analysis.get("faithful", False)
            if isinstance(is_faithful, str):
                is_faithful = is_faithful.lower() == "true"

            score_val = float(analysis.get("score", 1.0 if is_faithful else 0.0))
            reasoning = analysis.get("reasoning", "No reasoning provided.")

            return Score(
                name="Faithfulness",
                value=score_val,
                max_value=1.0,
                passed=is_faithful,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"Error in FaithfulnessGrader: {e}")
            return Score(
                name="Faithfulness",
                value=0.0,
                passed=False,
                reasoning=f"Grading failed due to internal error: {str(e)}",
            )


class ToneGrader(LLMGrader):
    """
    Grades whether the agent's tone matches expectations.
    Uses an LLMClient to evaluate the text.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client)
        self.default_tone = "Professional and Empathetic"

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
        agent: Optional[AgentDefinition] = None,
    ) -> Score:
        # Determine expected tone
        expected_tone = self.default_tone
        if expectations:
            tone_override = expectations.get("tone")
            if isinstance(tone_override, str) and tone_override.strip():
                expected_tone = tone_override.strip()

        text = result.actual_output.text
        if not text:
            return Score(
                name="Tone",
                value=0.0,
                passed=False,
                reasoning="No text output to check for tone.",
            )

        # Construct prompt
        prompt = TONE_GRADER_PROMPT.safe_substitute(TONE=expected_tone, RESPONSE=text)

        try:
            analysis = self._get_llm_analysis(prompt)

            matches_tone = analysis.get("matches_tone", False)
            if isinstance(matches_tone, str):
                matches_tone = matches_tone.lower() == "true"

            score_val = float(analysis.get("score", 1.0 if matches_tone else 0.0))
            reasoning = analysis.get("reasoning", "No reasoning provided.")

            return Score(
                name="Tone",
                value=score_val,
                max_value=1.0,
                passed=matches_tone,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"Error in ToneGrader: {e}")
            return Score(
                name="Tone",
                value=0.0,
                passed=False,
                reasoning=f"Grading failed due to internal error: {str(e)}",
            )
