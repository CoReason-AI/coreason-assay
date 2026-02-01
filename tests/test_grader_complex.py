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
from coreason_assay.models import TestResult, TestResultOutput


@pytest.fixture
def complex_mock_result() -> TestResult:
    return TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(error=None, text="foo", trace=None, structured_output={"key": "value", "extra": "data"}),
        metrics={"latency_ms": 1000.0},
        passed=False,
    )


def test_latency_grader_boundary_condition(complex_mock_result: TestResult) -> None:
    # Latency is 1000.0
    grader = LatencyGrader(threshold_ms=1000.0)
    score = grader.grade(complex_mock_result)

    assert score.passed is True
    assert score.value == 1000.0
    assert score.reasoning is not None and "within" in score.reasoning


def test_latency_grader_zero_threshold(complex_mock_result: TestResult) -> None:
    # Latency is 1000.0, Threshold is 0.0
    grader = LatencyGrader(threshold_ms=0.0)
    score = grader.grade(complex_mock_result)

    assert score.passed is False
    assert score.max_value == 0.0
    assert score.reasoning is not None and "exceeds" in score.reasoning


def test_json_schema_grader_empty_structures(complex_mock_result: TestResult) -> None:
    # Output is {"key": "value", ...}
    # Expectation is {} (no specific keys required) - Valid JSON Schema (empty schema accepts anything)
    grader = JsonSchemaGrader()
    score = grader.grade(complex_mock_result, expectations={"structure": {}})

    assert score.passed is True


def test_json_schema_grader_extra_keys(complex_mock_result: TestResult) -> None:
    # Output is {"key": "value", "extra": "data"}
    # Expectation is {"type": "object", "required": ["key"]} (JSON Schema)
    grader = JsonSchemaGrader()
    schema = {"type": "object", "required": ["key"]}
    score = grader.grade(complex_mock_result, expectations={"structure": schema})

    assert score.passed is True
    # "extra" key is allowed by default in JSON schema


def test_json_schema_grader_non_dict_output() -> None:
    # Output is a list
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="foo",
            trace=None,
            # Output is a list
            structured_output=["item1", "item2"],
        ),
        metrics={},
        passed=False,
    )

    grader = JsonSchemaGrader()
    # Expectation is an object
    schema = {"type": "object"}
    score = grader.grade(result, expectations={"structure": schema})

    assert score.passed is False
    assert score.reasoning is not None and "Validation failed" in score.reasoning


def test_json_schema_grader_strict_properties(complex_mock_result: TestResult) -> None:
    # Output is {"key": "value", "extra": "data"}
    # Expectation: No extra properties allowed
    grader = JsonSchemaGrader()
    schema = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    }
    score = grader.grade(complex_mock_result, expectations={"structure": schema})

    assert score.passed is False
    assert score.reasoning is not None and "Validation failed" in score.reasoning
    # The error message should ideally mention 'extra' property is unexpected


def test_json_schema_grader_array_validation() -> None:
    # Output is a list of objects
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="list",
            trace=None,
            structured_output=[
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
            ],
        ),
        metrics={},
        passed=False,
    )

    grader = JsonSchemaGrader()
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
            "required": ["id", "name"],
        },
    }
    score = grader.grade(result, expectations={"structure": schema})

    assert score.passed is True


def test_json_schema_grader_pattern_validation() -> None:
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(error=None, text="foo", trace=None, structured_output={"email": "invalid-email"}),
        metrics={},
        passed=False,
    )

    grader = JsonSchemaGrader()
    # Simple regex for testing pattern validation
    schema = {
        "type": "object",
        "properties": {"email": {"type": "string", "pattern": "^\\S+@\\S+\\.\\S+$"}},
    }
    score = grader.grade(result, expectations={"structure": schema})

    assert score.passed is False
    assert score.reasoning is not None and "Validation failed" in score.reasoning


def test_json_schema_grader_deep_nested_error() -> None:
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="foo",
            trace=None,
            structured_output={
                "level1": {
                    "level2": {
                        "level3": "wrong_type"  # Should be integer
                    }
                }
            },
        ),
        metrics={},
        passed=False,
    )

    grader = JsonSchemaGrader()
    schema = {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {"level3": {"type": "integer"}},
                    }
                },
            }
        },
    }
    score = grader.grade(result, expectations={"structure": schema})

    assert score.passed is False
    assert score.reasoning is not None and "Validation failed" in score.reasoning
    # jsonschema usually gives a clear message.
    # We might not get the full path in e.message unless we traverse e.path,
    # but e.message usually says " 'wrong_type' is not of type 'integer' "


def test_json_schema_grader_nullable_fields() -> None:
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(error=None, text="foo", trace=None, structured_output={"optional_field": None}),
        metrics={},
        passed=False,
    )

    grader = JsonSchemaGrader()
    schema = {
        "type": "object",
        "properties": {"optional_field": {"type": ["string", "null"]}},
        "required": ["optional_field"],
    }
    score = grader.grade(result, expectations={"structure": schema})

    assert score.passed is True
