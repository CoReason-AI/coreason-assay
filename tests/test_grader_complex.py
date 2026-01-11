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
        actual_output=TestResultOutput(text="foo", trace="log", structured_output={"key": "value", "extra": "data"}),
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
            trace="log",
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
