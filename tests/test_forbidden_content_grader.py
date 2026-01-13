# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from uuid import uuid4

import pytest
from coreason_assay.grader import ForbiddenContentGrader
from coreason_assay.models import TestResult, TestResultOutput


@pytest.fixture
def mock_result_with_text() -> TestResult:
    return TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="The patient has a headache and should take aspirin.", trace="log", structured_output=None
        ),
        metrics={"latency_ms": 100},
        passed=False,
    )


def test_forbidden_content_none_specified(mock_result_with_text: TestResult) -> None:
    grader = ForbiddenContentGrader()
    # No expectations provided
    score = grader.grade(mock_result_with_text)
    assert score.passed is True
    assert score.value == 1.0
    assert score.reasoning is not None and "No forbidden content" in score.reasoning

    # Empty list in expectations
    score = grader.grade(mock_result_with_text, expectations={"forbidden_content": []})
    assert score.passed is True
    assert score.value == 1.0


def test_forbidden_content_not_found(mock_result_with_text: TestResult) -> None:
    grader = ForbiddenContentGrader()
    expectations = {"forbidden_content": ["cancer", "tumor"]}
    score = grader.grade(mock_result_with_text, expectations=expectations)

    assert score.passed is True
    assert score.value == 1.0
    assert score.reasoning is not None and "None of the forbidden terms were found" in score.reasoning


def test_forbidden_content_found(mock_result_with_text: TestResult) -> None:
    grader = ForbiddenContentGrader()
    # "headache" is in the text
    expectations = {"forbidden_content": ["headache"]}
    score = grader.grade(mock_result_with_text, expectations=expectations)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None and "Found forbidden content: 'headache'" in score.reasoning


def test_forbidden_content_case_insensitive(mock_result_with_text: TestResult) -> None:
    grader = ForbiddenContentGrader()
    # "ASPIRIN" should match "aspirin"
    expectations = {"forbidden_content": ["ASPIRIN"]}
    score = grader.grade(mock_result_with_text, expectations=expectations)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None and "Found forbidden content: 'ASPIRIN'" in score.reasoning


def test_forbidden_content_partial_match(mock_result_with_text: TestResult) -> None:
    grader = ForbiddenContentGrader()
    # "head" should match "headache"
    expectations = {"forbidden_content": ["head"]}
    score = grader.grade(mock_result_with_text, expectations=expectations)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None and "Found forbidden content: 'head'" in score.reasoning


def test_forbidden_content_multiple_matches(mock_result_with_text: TestResult) -> None:
    grader = ForbiddenContentGrader()
    expectations = {"forbidden_content": ["patient", "aspirin", "cancer"]}
    score = grader.grade(mock_result_with_text, expectations=expectations)

    assert score.passed is False
    assert score.value == 0.0
    # Should list found items
    assert score.reasoning is not None
    assert "patient" in score.reasoning
    assert "aspirin" in score.reasoning
    assert "cancer" not in score.reasoning


def test_forbidden_content_no_text_output() -> None:
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(text=None, trace="log", structured_output=None),
        metrics={},
        passed=False,
    )
    grader = ForbiddenContentGrader()
    expectations = {"forbidden_content": ["fail"]}
    score = grader.grade(result, expectations=expectations)

    # If no text, no forbidden content can be found -> Pass
    assert score.passed is True
    assert score.value == 1.0
    assert score.reasoning is not None and "No text output to check" in score.reasoning
