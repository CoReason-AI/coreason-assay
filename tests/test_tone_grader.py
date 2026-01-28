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
from typing import Any, Dict
from unittest.mock import Mock
from uuid import UUID

import pytest
from coreason_assay.grader import ToneGrader
from coreason_assay.interfaces import LLMClient
from coreason_assay.models import TestResult, TestResultOutput


class MockLLMClient(LLMClient):
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = ""

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response_text


@pytest.fixture
def mock_result() -> TestResult:
    return TestResult(
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        case_id=UUID("00000000-0000-0000-0000-000000000002"),
        actual_output=TestResultOutput(
            text="I understand your pain and I am here to help.", trace=None, structured_output=None
        ),
        metrics={},
        scores=[],
        passed=False,
    )


def test_tone_grader_default_success(mock_result: TestResult) -> None:
    llm_response = json.dumps(
        {"matches_tone": True, "reasoning": "The response is very empathetic and professional.", "score": 1.0}
    )
    client = MockLLMClient(llm_response)
    grader = ToneGrader(client)

    score = grader.grade(mock_result, inputs=None, expectations=None)

    assert score.passed is True
    assert score.value == 1.0
    assert score.name == "Tone"
    assert "Professional and Empathetic" in client.last_prompt


def test_tone_grader_override_success(mock_result: TestResult) -> None:
    llm_response = json.dumps(
        {"matches_tone": True, "reasoning": "The response matches the urgent tone.", "score": 1.0}
    )
    client = MockLLMClient(llm_response)
    grader = ToneGrader(client)

    expectations: Dict[str, Any] = {"tone": "Urgent and Directive"}
    score = grader.grade(mock_result, inputs=None, expectations=expectations)

    assert score.passed is True
    assert "Urgent and Directive" in client.last_prompt


def test_tone_grader_fail(mock_result: TestResult) -> None:
    llm_response = json.dumps({"matches_tone": False, "reasoning": "The response is rude.", "score": 0.0})
    client = MockLLMClient(llm_response)
    grader = ToneGrader(client)

    score = grader.grade(mock_result, inputs=None, expectations=None)

    assert score.passed is False
    assert score.value == 0.0


def test_tone_grader_no_text(mock_result: TestResult) -> None:
    mock_result.actual_output.text = None
    client = MockLLMClient("")
    grader = ToneGrader(client)

    score = grader.grade(mock_result)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None
    assert "No text output" in score.reasoning


def test_tone_grader_llm_error(mock_result: TestResult) -> None:
    client = Mock()
    client.complete.side_effect = Exception("API Error")
    grader = ToneGrader(client)

    score = grader.grade(mock_result)

    assert score.passed is False
    assert score.reasoning is not None
    assert "Grading failed" in score.reasoning


def test_tone_grader_malformed_json(mock_result: TestResult) -> None:
    client = MockLLMClient("NOT JSON")
    grader = ToneGrader(client)

    score = grader.grade(mock_result)

    assert score.passed is False
    assert score.reasoning is not None
    assert "Grading failed" in score.reasoning
