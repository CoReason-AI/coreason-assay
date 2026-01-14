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

    # We use explicit markers for replacement instead of format() to be safe with JSON braces
    PROMPT_TEMPLATE = """You are an expert evaluator of AI reasoning chains.
Your task is to verify if the actual execution trace of an AI agent contains specific required reasoning steps.

Required Reasoning Steps:
__REQUIRED_STEPS__

Actual Execution Trace:
__TRACE__

(Fallback) Actual Output Text:
__TEXT__

Instructions:
1. Analyze the trace (and text if trace is insufficient) to find evidence of each required step.
2. Return a JSON object with the following structure:
{
  "steps_analysis": [
    {"step": "step description", "found": true/false, "evidence": "quote or explanation"},
    ...
  ],
  "score": <float between 0.0 and 1.0 representing percentage of steps found>
}

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

        # Use replace() instead of format() to avoid issues with curly braces in trace/text (e.g. JSON)
        prompt = (
            self.PROMPT_TEMPLATE.replace("__REQUIRED_STEPS__", formatted_steps)
            .replace("__TRACE__", trace)
            .replace("__TEXT__", text)
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
        if text is None:
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


class FaithfulnessGrader(BaseGrader):
    """
    Grades whether the agent's answer is faithful to the provided context.
    Uses an LLMClient to detect hallucinations or contradictions.
    """

    PROMPT_TEMPLATE = """You are an expert fact-checker for AI assistants.
Your task is to verify if the AI's generated answer is supported by the provided Context.
Does the Answer hallucinate information not present in the Context, or contradict it?

Context:
__CONTEXT__

AI Answer:
__ANSWER__

Instructions:
1. Analyze the Answer against the Context.
2. Return a JSON object with the following structure:
{
  "faithful": true/false,
  "reasoning": "Explanation of why it is faithful or not. Cite specific contradictions if any.",
  "score": 1.0 (if faithful) or 0.0 (if not)
}

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

        # Use chained replace carefully.
        # Although unlikely, if context_str contains "__ANSWER__", the second replace would inject the answer
        # into the context section. To prevent this, we could use unique keys or do it in two steps.
        # But replacing sequentially is standard.
        # To be safe against collision, we replace in a specific order if we know the structure.
        # But context is arbitrary.
        # Better: use string formatting or replace simultaneously.
        # Since replace() processes the string, we can do:
        prompt = self.PROMPT_TEMPLATE.replace("__CONTEXT__", context_str)
        prompt = prompt.replace("__ANSWER__", answer_text)

        try:
            response_text = self.llm_client.complete(prompt)
            # Clean JSON
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            analysis = json.loads(cleaned_response)

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

class ToneGrader(BaseGrader):
    """
    Grades whether the agent's tone matches expectations.
    Uses an LLMClient to evaluate the text.
    """

    PROMPT_TEMPLATE = """You are an expert tone analyzer for AI assistants.
Your task is to verify if the AI's response matches the expected tone.

Expected Tone:
__TONE__

AI Response:
__RESPONSE__

Instructions:
1. Analyze the Response to see if it aligns with the Expected Tone.
2. Return a JSON object with the following structure:
{
  "matches_tone": true/false,
  "reasoning": "Explanation of why it matches or fails. Cite specific words or phrases.",
  "score": 1.0 (if matches) or 0.0 (if not)
}

Return ONLY the JSON.
"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.default_tone = "Professional and Empathetic"

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
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
        prompt = self.PROMPT_TEMPLATE.replace("__TONE__", expected_tone).replace("__RESPONSE__", text)

        try:
            response_text = self.llm_client.complete(prompt)
            # Clean JSON
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            analysis = json.loads(cleaned_response)

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
