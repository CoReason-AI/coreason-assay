# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Optional
from uuid import uuid4

from coreason_assay.models import (
    AggregateMetric,
    Score,
    TestResult,
    TestResultOutput,
    TestRun,
    TestRunStatus,
)
from coreason_assay.reporting import generate_report_card


def test_nan_value_pass_rate_accuracy() -> None:
    """
    Verifies that 'Pass Rate' is calculated correctly even when score values are NaN.
    The denominator should be the total number of scores, not just valid numeric ones.
    """
    run_id = uuid4()
    test_run = TestRun(
        id=run_id,
        corpus_version="v1",
        agent_draft_version="d1",
        status=TestRunStatus.DONE,
    )

    # Case 1: Valid Value, Passed
    result_1 = TestResult(
        run_id=run_id,
        case_id=uuid4(),
        passed=True,
        actual_output=TestResultOutput(
            text="ok",
            trace="trace",
            structured_output={},
        ),
        metrics={},
        scores=[Score(name="TestMetric", value=1.0, passed=True, reasoning="good")],
    )

    # Case 2: NaN Value, Passed (e.g., Latency missing but forced pass override?)
    # Or just some grader returning NaN but marking passed.
    result_2 = TestResult(
        run_id=run_id,
        case_id=uuid4(),
        passed=True,
        actual_output=TestResultOutput(
            text="ok",
            trace="trace",
            structured_output={},
        ),
        metrics={},
        scores=[Score(name="TestMetric", value=float("nan"), passed=True, reasoning="weird but passed")],
    )

    # Case 3: Valid Value, Failed
    result_3 = TestResult(
        run_id=run_id,
        case_id=uuid4(),
        passed=False,
        actual_output=TestResultOutput(
            text="fail",
            trace="trace",
            structured_output={},
        ),
        metrics={},
        scores=[Score(name="TestMetric", value=0.0, passed=False, reasoning="bad")],
    )

    results = [result_1, result_2, result_3]
    report = generate_report_card(test_run, results)

    def get_metric(name: str) -> Optional[AggregateMetric]:
        for m in report.aggregates:
            if m.name == name:
                return m
        return None

    # Expected:
    # Total Count = 3
    # Passed Count = 2 (Case 1 + Case 2)
    # Pass Rate = 2 / 3 = 0.666...
    pass_rate = get_metric("TestMetric Pass Rate")
    assert pass_rate is not None
    assert abs(pass_rate.value - (2 / 3)) < 1e-6
    assert pass_rate.total_samples == 3

    # Average Score Calculation
    # Valid Values: [1.0, 0.0] (NaN is excluded)
    # Average = 0.5
    # Total Samples for Average = 2
    avg_score = get_metric("Average TestMetric Score")
    assert avg_score is not None
    assert avg_score.value == 0.5
    assert avg_score.total_samples == 2


def test_mixed_empty_and_valid_scores() -> None:
    """
    Verifies aggregation when some results are missing the score entirely.
    """
    run_id = uuid4()
    test_run = TestRun(
        id=run_id,
        corpus_version="v1",
        agent_draft_version="d1",
        status=TestRunStatus.DONE,
    )

    # Case 1: Has Score
    result_1 = TestResult(
        run_id=run_id,
        case_id=uuid4(),
        passed=True,
        actual_output=TestResultOutput(
            text="ok",
            trace="trace",
            structured_output={},
        ),
        scores=[Score(name="MyScore", value=10, passed=True, reasoning="ok")],
    )

    # Case 2: Missing Score
    result_2 = TestResult(
        run_id=run_id,
        case_id=uuid4(),
        passed=True,
        actual_output=TestResultOutput(
            text="ok",
            trace="trace",
            structured_output={},
        ),
        scores=[],  # No scores here
    )

    results = [result_1, result_2]
    report = generate_report_card(test_run, results)

    def get_metric(name: str) -> Optional[AggregateMetric]:
        for m in report.aggregates:
            if m.name == name:
                return m
        return None

    # MyScore stats:
    # Total occurrences = 1 (only result_1)
    # Passed = 1
    # Pass Rate = 1.0
    pass_rate = get_metric("MyScore Pass Rate")
    assert pass_rate is not None
    assert pass_rate.value == 1.0
    assert pass_rate.total_samples == 1
