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
def complex_result() -> TestResult:
    return TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="The classification of the category is C++. Also, check the résumé.",
            trace="log",
            structured_output=None,
        ),
        metrics={"latency_ms": 100},
        passed=False,
    )


def test_forbidden_content_empty_string_ignored(complex_result: TestResult) -> None:
    """
    Edge case: If the user accidentally puts an empty string in the forbidden list,
    it should effectively match everything (since "" is in every string).
    However, practically, this is usually a configuration error.
    The grader should probably ignore empty strings to be safe, or fail loudly.
    Let's assume we want to ignore them to prevent false positives on everything.
    """
    grader = ForbiddenContentGrader()
    expectations = {"forbidden_content": ["", "invalid"]}
    score = grader.grade(complex_result, expectations=expectations)

    # "invalid" is not in text. "" is in text.
    # If logic is simple `if term in text`, "" will match.
    # We should assert that the implementation is robust enough to ignore "" or handle it.
    # Current implementation might fail this test if it matches "".
    assert score.passed is True
    assert score.value == 1.0


def test_forbidden_content_substring_collision(complex_result: TestResult) -> None:
    """
    Test that we don't flag words that contain the forbidden term as a substring
    unless that is the desired behavior.
    The current implementation is simple substring matching, so "cat" will match "category".
    This test documents that behavior.
    If we wanted whole-word matching, this test would expect True.
    For now, we expect False because "cat" is in "category".
    """
    grader = ForbiddenContentGrader()
    expectations = {"forbidden_content": ["cat"]}
    score = grader.grade(complex_result, expectations=expectations)

    assert score.passed is False
    assert score.value == 0.0
    assert score.reasoning is not None and "'cat'" in score.reasoning


def test_forbidden_content_unicode_normalization(complex_result: TestResult) -> None:
    """
    Test matching of unicode characters.
    Text has "résumé".
    forbidden: "resume" -> Should match if we did rigorous normalization, but basic string matching won't.
    forbidden: "résumé" -> Should match.
    """
    grader = ForbiddenContentGrader()

    # Exact match with accents
    expectations_1 = {"forbidden_content": ["résumé"]}
    score_1 = grader.grade(complex_result, expectations=expectations_1)
    assert score_1.passed is False

    # Mismatch due to accents (documents that we don't do unidecode normalization currently)
    expectations_2 = {"forbidden_content": ["resume"]}
    score_2 = grader.grade(complex_result, expectations=expectations_2)
    assert score_2.passed is True


def test_forbidden_content_special_characters(complex_result: TestResult) -> None:
    """
    Test matching of special characters like "C++".
    """
    grader = ForbiddenContentGrader()
    expectations = {"forbidden_content": ["C++"]}
    score = grader.grade(complex_result, expectations=expectations)

    assert score.passed is False
    assert score.reasoning is not None and "C++" in score.reasoning


def test_forbidden_content_whitespace_handling() -> None:
    """
    Test that whitespace is respected.
    """
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(text="System error code 500.", trace="log", structured_output=None),
        metrics={},
        passed=False,
    )
    grader = ForbiddenContentGrader()

    # " error " should match
    expectations_1 = {"forbidden_content": [" error "]}
    score_1 = grader.grade(result, expectations=expectations_1)
    assert score_1.passed is False

    # "error " should match
    expectations_2 = {"forbidden_content": ["error "]}
    score_2 = grader.grade(result, expectations=expectations_2)
    assert score_2.passed is False

    # " code 500" should match
    expectations_3 = {"forbidden_content": [" code 500"]}
    score_3 = grader.grade(result, expectations=expectations_3)
    assert score_3.passed is False
