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
from typing import Optional
from uuid import uuid4

import pytest

from coreason_assay.grader import FaithfulnessGrader
from coreason_assay.interfaces import LLMClient
from coreason_assay.models import TestCaseInput, TestResult, TestResultOutput


class MockLLMClient(LLMClient):
    def __init__(self, default_response: Optional[str] = None):
        self.default_response = default_response
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.default_response:
            return self.default_response
        return "{}"


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def faithfulness_grader(mock_llm_client: MockLLMClient) -> FaithfulnessGrader:
    return FaithfulnessGrader(llm_client=mock_llm_client)


@pytest.fixture
def basic_result() -> TestResult:
    return TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="The sky is blue.",
            trace=None,
            structured_output=None,
        ),
        metrics={},
        scores=[],
        passed=False,
    )


@pytest.fixture
def basic_inputs() -> TestCaseInput:
    return TestCaseInput(
        prompt="What color is the sky?",
        context={"ground_truth": "The sky is blue due to Rayleigh scattering."},
    )


def test_faithful_match(
    mock_llm_client: MockLLMClient,
    faithfulness_grader: FaithfulnessGrader,
    basic_result: TestResult,
    basic_inputs: TestCaseInput,
) -> None:
    # Setup mock for pass
    mock_response = json.dumps(
        {
            "faithful": True,
            "reasoning": "The answer matches the ground truth context.",
            "score": 1.0,
        }
    )
    mock_llm_client.default_response = mock_response

    score = faithfulness_grader.grade(basic_result, inputs=basic_inputs)

    assert score.name == "Faithfulness"
    assert score.passed is True
    assert score.value == 1.0
    assert score.reasoning is not None
    assert "matches the ground truth" in score.reasoning

    # Verify prompt
    assert len(mock_llm_client.calls) == 1
    prompt = mock_llm_client.calls[0]
    assert "The sky is blue" in prompt  # Answer
    assert "Rayleigh scattering" in prompt  # Context


def test_unfaithful_match(
    mock_llm_client: MockLLMClient,
    faithfulness_grader: FaithfulnessGrader,
    basic_result: TestResult,
    basic_inputs: TestCaseInput,
) -> None:
    # Setup mock for fail
    mock_response = json.dumps(
        {
            "faithful": False,
            "reasoning": "The answer contradicts the context.",
            "score": 0.0,
        }
    )
    mock_llm_client.default_response = mock_response

    score = faithfulness_grader.grade(basic_result, inputs=basic_inputs)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None
    assert "contradicts" in score.reasoning


def test_missing_context(faithfulness_grader: FaithfulnessGrader, basic_result: TestResult) -> None:
    # Inputs without context
    inputs = TestCaseInput(prompt="foo", context={})

    score = faithfulness_grader.grade(basic_result, inputs=inputs)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None
    assert "No context provided" in score.reasoning


def test_missing_answer(faithfulness_grader: FaithfulnessGrader, basic_inputs: TestCaseInput) -> None:
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
        metrics={},
        scores=[],
        passed=False,
    )

    score = faithfulness_grader.grade(result, inputs=basic_inputs)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None
    assert "No answer text" in score.reasoning


def test_malformed_llm_response(
    mock_llm_client: MockLLMClient,
    faithfulness_grader: FaithfulnessGrader,
    basic_result: TestResult,
    basic_inputs: TestCaseInput,
) -> None:
    mock_llm_client.default_response = "Not JSON"

    score = faithfulness_grader.grade(basic_result, inputs=basic_inputs)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None
    assert "Grading failed" in score.reasoning


def test_json_markdown_stripping(
    mock_llm_client: MockLLMClient,
    faithfulness_grader: FaithfulnessGrader,
    basic_result: TestResult,
    basic_inputs: TestCaseInput,
) -> None:
    mock_response = "```json\n" + json.dumps({"faithful": True, "score": 1.0}) + "\n```"
    mock_llm_client.default_response = mock_response

    score = faithfulness_grader.grade(basic_result, inputs=basic_inputs)

    assert score.passed is True
    assert score.value == 1.0


def test_string_boolean_parsing(
    mock_llm_client: MockLLMClient,
    faithfulness_grader: FaithfulnessGrader,
    basic_result: TestResult,
    basic_inputs: TestCaseInput,
) -> None:
    # LLM returns "true" string instead of boolean
    mock_response = json.dumps({"faithful": "true", "score": 1.0})
    mock_llm_client.default_response = mock_response

    score = faithfulness_grader.grade(basic_result, inputs=basic_inputs)

    assert score.passed is True
