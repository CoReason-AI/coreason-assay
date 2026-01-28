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
from typing import Any, Dict, Optional
from uuid import uuid4

import pytest

from coreason_assay.grader import ReasoningGrader
from coreason_assay.interfaces import LLMClient
from coreason_assay.models import TestResult, TestResultOutput


class MockLLMClient(LLMClient):
    def __init__(self, response_map: Optional[Dict[str, str]] = None, default_response: Optional[str] = None):
        self.response_map = response_map or {}
        self.default_response = default_response
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for key, response in self.response_map.items():
            if key in prompt:
                return response
        if self.default_response:
            return self.default_response
        return "{}"


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def reasoning_grader(mock_llm_client: MockLLMClient) -> ReasoningGrader:
    return ReasoningGrader(llm_client=mock_llm_client)


@pytest.fixture
def basic_result() -> TestResult:
    return TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="The patient has diabetes.",
            trace="Step 1: Check glucose. Step 2: Compare to limit.",
            structured_output=None,
        ),
        metrics={},
        scores=[],
        passed=False,
    )


def test_no_reasoning_expectations(reasoning_grader: ReasoningGrader, basic_result: TestResult) -> None:
    expectations: Dict[str, Any] = {}  # No "reasoning" key
    score = reasoning_grader.grade(basic_result, expectations=expectations)
    assert score.passed is True
    assert score.value == 1.0
    assert score.reasoning == "No reasoning expectations provided."


def test_perfect_match(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, basic_result: TestResult
) -> None:
    expectations = {"reasoning": ["Check glucose", "Compare to limit"]}

    # Configure mock to return 100% score
    mock_response = json.dumps(
        {
            "steps_analysis": [
                {"step": "Check glucose", "found": True, "evidence": "Step 1: Check glucose"},
                {"step": "Compare to limit", "found": True, "evidence": "Step 2: Compare to limit"},
            ],
            "score": 1.0,
        }
    )
    mock_llm_client.default_response = mock_response

    score = reasoning_grader.grade(basic_result, expectations=expectations)

    assert score.passed is True
    assert score.value == 1.0
    assert score.reasoning is not None
    assert "✅ Check glucose" in score.reasoning
    assert "✅ Compare to limit" in score.reasoning

    # Verify prompt contains trace
    assert len(mock_llm_client.calls) == 1
    prompt = mock_llm_client.calls[0]
    assert "Check glucose" in prompt
    assert "Step 1: Check glucose" in prompt


def test_partial_match(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, basic_result: TestResult
) -> None:
    expectations = {"reasoning": ["Check glucose", "Prescribe medication"]}

    mock_response = json.dumps(
        {
            "steps_analysis": [
                {"step": "Check glucose", "found": True, "evidence": "Step 1: Check glucose"},
                {"step": "Prescribe medication", "found": False, "evidence": "Not found"},
            ],
            "score": 0.5,
        }
    )
    mock_llm_client.default_response = mock_response

    score = reasoning_grader.grade(basic_result, expectations=expectations)

    assert score.passed is False
    assert score.value == 0.5
    assert score.reasoning is not None
    assert "✅ Check glucose" in score.reasoning
    assert "❌ Prescribe medication" in score.reasoning


def test_no_match(mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, basic_result: TestResult) -> None:
    expectations = {"reasoning": ["Do something irrelevant"]}

    mock_response = json.dumps(
        {
            "steps_analysis": [{"step": "Do something irrelevant", "found": False, "evidence": "Not found"}],
            "score": 0.0,
        }
    )
    mock_llm_client.default_response = mock_response

    score = reasoning_grader.grade(basic_result, expectations=expectations)

    assert score.passed is False
    assert score.value == 0.0


def test_malformed_llm_response(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, basic_result: TestResult
) -> None:
    expectations = {"reasoning": ["Step 1"]}

    mock_llm_client.default_response = "Not JSON"

    score = reasoning_grader.grade(basic_result, expectations=expectations)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None
    assert "Grading failed" in score.reasoning


def test_json_markdown_stripping(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, basic_result: TestResult
) -> None:
    expectations = {"reasoning": ["Step 1"]}

    mock_response = (
        "```json\n" + json.dumps({"steps_analysis": [{"step": "Step 1", "found": True}], "score": 1.0}) + "\n```"
    )
    mock_llm_client.default_response = mock_response

    score = reasoning_grader.grade(basic_result, expectations=expectations)

    assert score.passed is True
    assert score.value == 1.0


def test_fallback_to_text(mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader) -> None:
    # Result with empty trace but valid text
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="I checked glucose levels.",
            trace=None,
            structured_output=None,
        ),
        metrics={},
        scores=[],
        passed=False,
    )

    expectations = {"reasoning": ["Check glucose"]}

    mock_response = json.dumps(
        {
            "steps_analysis": [{"step": "Check glucose", "found": True}],
            "score": 1.0,
        }
    )
    mock_llm_client.default_response = mock_response

    score = reasoning_grader.grade(result, expectations=expectations)

    assert score.passed is True
    assert score.value == 1.0

    # Check that text was included in the prompt
    prompt = mock_llm_client.calls[0]
    assert "I checked glucose levels" in prompt


def test_empty_analysis_details(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, basic_result: TestResult
) -> None:
    expectations = {"reasoning": ["Step 1"]}
    # LLM returns valid JSON but empty steps_analysis list
    mock_response = json.dumps(
        {
            "steps_analysis": [],
            "score": 0.0,
        }
    )
    mock_llm_client.default_response = mock_response

    score = reasoning_grader.grade(basic_result, expectations=expectations)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning == "LLM provided no detailed analysis."
