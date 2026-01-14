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

from coreason_assay.grader import JsonSchemaGrader, LatencyGrader
from coreason_assay.models import Score, TestResult, TestResultOutput


@pytest.fixture
def mock_result() -> TestResult:
    return TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(text="foo", trace="log", structured_output={"key": "value"}),
        metrics={"latency_ms": 1000.0},
        passed=False,
    )


def test_latency_grader_pass(mock_result: TestResult) -> None:
    # Threshold is 5000ms by default, latency is 1000ms
    grader = LatencyGrader()
    score = grader.grade(mock_result)

    assert isinstance(score, Score)
    assert score.name == "Latency"
    assert score.value == 1000.0
    assert score.passed is True
    assert score.reasoning is not None and "within" in score.reasoning


def test_latency_grader_fail(mock_result: TestResult) -> None:
    # Set threshold to 500ms, latency is 1000ms
    grader = LatencyGrader(threshold_ms=500.0)
    score = grader.grade(mock_result)

    assert score.passed is False
    assert score.max_value == 500.0
    assert score.reasoning is not None and "exceeds" in score.reasoning


def test_latency_grader_expectations_override(mock_result: TestResult) -> None:
    # Default is 5000ms, override to 500ms in expectations
    grader = LatencyGrader(threshold_ms=5000.0)
    score = grader.grade(mock_result, expectations={"latency_threshold_ms": 500.0})

    assert score.passed is False
    assert score.max_value == 500.0
    assert score.reasoning is not None and "exceeds" in score.reasoning


def test_latency_grader_missing_metric() -> None:
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(text="foo", trace="log", structured_output=None),
        metrics={},
        passed=False,
    )
    grader = LatencyGrader()
    score = grader.grade(result)

    assert score.passed is False
    assert score.value == 0
    assert score.reasoning is not None and "missing" in score.reasoning


def test_json_schema_grader_pass_no_schema(mock_result: TestResult) -> None:
    grader = JsonSchemaGrader()
    # No specific structure expectation, but output exists
    score = grader.grade(mock_result)
    assert score.passed is True
    assert score.reasoning is not None and "no schema provided" in score.reasoning


def test_json_schema_grader_structure_match(mock_result: TestResult) -> None:
    grader = JsonSchemaGrader()
    # Simple schema: required key "key" which must be a string
    schema = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}
    expectations = {"structure": schema}
    score = grader.grade(mock_result, expectations=expectations)
    assert score.passed is True
    assert score.reasoning is not None and "matches the expected" in score.reasoning


def test_json_schema_grader_structure_mismatch_type(mock_result: TestResult) -> None:
    grader = JsonSchemaGrader()
    # Expect "key" to be integer, but it is "value" (string)
    schema = {"type": "object", "properties": {"key": {"type": "integer"}}, "required": ["key"]}
    expectations = {"structure": schema}
    score = grader.grade(mock_result, expectations=expectations)
    assert score.passed is False
    assert score.reasoning is not None and "Validation failed" in score.reasoning


def test_json_schema_grader_structure_mismatch_missing_key(mock_result: TestResult) -> None:
    grader = JsonSchemaGrader()
    # Expect "other_key"
    schema = {"type": "object", "properties": {"other_key": {"type": "string"}}, "required": ["other_key"]}
    expectations = {"structure": schema}
    score = grader.grade(mock_result, expectations=expectations)
    assert score.passed is False
    assert score.reasoning is not None and "Validation failed" in score.reasoning
    assert score.reasoning is not None and "'other_key' is a required property" in score.reasoning


def test_json_schema_grader_nested_structure() -> None:
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="foo", trace="log", structured_output={"user": {"name": "Alice", "age": 30}}
        ),
        metrics={},
        passed=False,
    )
    grader = JsonSchemaGrader()
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            }
        },
        "required": ["user"],
    }
    expectations = {"structure": schema}
    score = grader.grade(result, expectations=expectations)
    assert score.passed is True


def test_json_schema_grader_invalid_schema(mock_result: TestResult) -> None:
    grader = JsonSchemaGrader()
    # Invalid schema (type is 'foo' which is invalid)
    schema = {
        "type": "foo",
    }
    expectations = {"structure": schema}
    score = grader.grade(mock_result, expectations=expectations)
    assert score.passed is False
    assert score.reasoning is not None and "Invalid JSON Schema" in score.reasoning


def test_json_schema_grader_no_output() -> None:
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(text="foo", trace="log", structured_output=None),
        metrics={},
        passed=False,
    )
    grader = JsonSchemaGrader()
    score = grader.grade(result)
    assert score.passed is False
    assert score.reasoning is not None and "No structured output" in score.reasoning
