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
from uuid import uuid4

import pytest

from coreason_assay.grader import ToneGrader
from coreason_assay.interfaces import LLMClient
from coreason_assay.models import TestResult, TestResultOutput


class MockLLMClient(LLMClient):
    def __init__(self, response_dict: Dict[str, Any]):
        self.response_text = json.dumps(response_dict)
        self.last_prompt = ""

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response_text


@pytest.fixture
def mock_result() -> TestResult:
    return TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(text="Placeholder", trace=None, structured_output=None),
        metrics={},
        scores=[],
        passed=False,
    )


def test_tone_grader_empty_text_explicit(mock_result: TestResult) -> None:
    """Ensure empty string output is handled gracefully (fails before LLM call)."""
    mock_result.actual_output.text = ""
    client = MockLLMClient({})
    grader = ToneGrader(client)

    score = grader.grade(mock_result)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None
    assert "No text output" in score.reasoning
    assert client.last_prompt == ""  # LLM should not be called


def test_tone_grader_whitespace_tone_expectation(mock_result: TestResult) -> None:
    """Ensure whitespace-only tone expectation falls back to default."""
    mock_result.actual_output.text = "Some text"
    client = MockLLMClient({"matches_tone": True, "score": 1.0})
    grader = ToneGrader(client)

    expectations: Dict[str, Any] = {"tone": "   "}
    score = grader.grade(mock_result, expectations=expectations)

    assert score.passed is True
    assert "Professional and Empathetic" in client.last_prompt


def test_tone_grader_unicode_emoji(mock_result: TestResult) -> None:
    """Ensure Unicode/Emoji in tone and text are handled correctly."""
    text_with_emoji = "Hello! 👋 I am happy to help! 😃"
    mock_result.actual_output.text = text_with_emoji

    # Client returns success
    client = MockLLMClient({"matches_tone": True, "score": 1.0})
    grader = ToneGrader(client)

    expectations: Dict[str, Any] = {"tone": "Friendly 🌈"}
    score = grader.grade(mock_result, expectations=expectations)

    assert score.passed is True
    # Verify proper string replacement in prompt
    assert text_with_emoji in client.last_prompt
    assert "Friendly 🌈" in client.last_prompt


def test_tone_grader_code_json_in_output(mock_result: TestResult) -> None:
    """Ensure valid JSON or Code in agent output doesn't break prompt construction."""
    json_output = '{"key": "value", "list": [1, 2, 3]}'
    mock_result.actual_output.text = json_output

    client = MockLLMClient({"matches_tone": True, "score": 1.0})
    grader = ToneGrader(client)

    score = grader.grade(mock_result)

    assert score.passed is True
    assert json_output in client.last_prompt


def test_tone_grader_large_input(mock_result: TestResult) -> None:
    """Ensure large text input is passed to LLM (assuming LLM handles it)."""
    large_text = "word " * 10000
    mock_result.actual_output.text = large_text

    client = MockLLMClient({"matches_tone": True, "score": 1.0})
    grader = ToneGrader(client)

    score = grader.grade(mock_result)

    assert score.passed is True
    # We just check if it ran without error and contained the text
    assert len(client.last_prompt) > len(large_text)
