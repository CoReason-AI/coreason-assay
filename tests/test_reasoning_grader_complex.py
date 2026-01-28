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
from typing import Dict, Optional
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
def complex_result() -> TestResult:
    return TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="Result.",
            trace="Complex Trace",
            structured_output=None,
        ),
        metrics={},
        scores=[],
        passed=False,
    )


def test_trace_with_json_and_braces(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, complex_result: TestResult
) -> None:
    # This trace would cause .format() to crash if not escaped
    trace_with_json = '{"key": "value", "nested": { "foo": "bar" }}'
    complex_result.actual_output.trace = trace_with_json

    expectations = {"reasoning": ["Step 1"]}

    mock_llm_client.default_response = json.dumps({"steps_analysis": [{"step": "Step 1", "found": True}], "score": 1.0})

    # Should not raise error
    score = reasoning_grader.grade(complex_result, expectations=expectations)

    assert score.passed is True
    assert score.value == 1.0

    # Verify prompt contains the JSON correctly
    prompt = mock_llm_client.calls[0]
    assert trace_with_json in prompt


def test_fuzzy_score_parsing(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, complex_result: TestResult
) -> None:
    expectations = {"reasoning": ["Step 1"]}

    # Test string score "0.5"
    mock_llm_client.default_response = json.dumps(
        {
            "steps_analysis": [{"step": "Step 1", "found": "true"}],  # Also test boolean as string
            "score": "0.5",
        }
    )

    score = reasoning_grader.grade(complex_result, expectations=expectations)

    assert score.value == 0.5
    assert score.reasoning is not None
    assert "✅ Step 1" in score.reasoning  # "true" string should be True


def test_percentage_score_parsing(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, complex_result: TestResult
) -> None:
    expectations = {"reasoning": ["Step 1"]}

    mock_llm_client.default_response = json.dumps({"steps_analysis": [], "score": "100%"})

    score = reasoning_grader.grade(complex_result, expectations=expectations)
    assert score.value == 1.0


def test_invalid_score_parsing(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, complex_result: TestResult
) -> None:
    expectations = {"reasoning": ["Step 1"]}

    # Test invalid string percentage
    mock_llm_client.default_response = json.dumps({"steps_analysis": [], "score": "bad%"})
    score = reasoning_grader.grade(complex_result, expectations=expectations)
    assert score.value == 0.0

    # Test completely invalid score
    mock_llm_client.default_response = json.dumps({"steps_analysis": [], "score": "invalid"})
    score = reasoning_grader.grade(complex_result, expectations=expectations)
    assert score.value == 0.0


def test_duplicate_expectations(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, complex_result: TestResult
) -> None:
    expectations = {"reasoning": ["Check A", "Check A"]}

    mock_llm_client.default_response = json.dumps(
        {"steps_analysis": [{"step": "Check A", "found": True}, {"step": "Check A", "found": True}], "score": 1.0}
    )

    score = reasoning_grader.grade(complex_result, expectations=expectations)
    assert score.value == 1.0

    # Verify prompt lists both
    prompt = mock_llm_client.calls[0]
    assert "1. Check A" in prompt
    assert "2. Check A" in prompt


def test_massive_trace_input(
    mock_llm_client: MockLLMClient, reasoning_grader: ReasoningGrader, complex_result: TestResult
) -> None:
    # 1MB trace
    massive_trace = "Log line..." * 100000
    complex_result.actual_output.trace = massive_trace
    expectations = {"reasoning": ["Step 1"]}

    mock_llm_client.default_response = json.dumps({"score": 1.0})

    score = reasoning_grader.grade(complex_result, expectations=expectations)
    assert score.passed is True
