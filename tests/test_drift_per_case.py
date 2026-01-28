# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Tuple
from uuid import UUID, uuid4

import pytest
from coreason_assay.drift import generate_drift_report
from coreason_assay.models import Score, TestResult, TestResultOutput, TestRun


@pytest.fixture
def run_metadata() -> Tuple[TestRun, TestRun]:
    """Generates a pair of runs with same corpus version."""
    v = "v1.0"
    r1 = TestRun(corpus_version=v, agent_draft_version="draft-1")
    r2 = TestRun(corpus_version=v, agent_draft_version="draft-2")
    return r1, r2


@pytest.fixture
def run_metadata_mismatch() -> Tuple[TestRun, TestRun]:
    """Generates a pair of runs with different corpus versions."""
    r1 = TestRun(corpus_version="v1.0", agent_draft_version="draft-1")
    r2 = TestRun(corpus_version="v1.1", agent_draft_version="draft-2")
    return r1, r2


def make_result(case_id: UUID, passed: bool, latency: float = 100.0, score_val: float = 1.0) -> TestResult:
    run_id = uuid4()
    return TestResult(
        run_id=run_id,
        case_id=case_id,
        passed=passed,
        actual_output=TestResultOutput(text="test", trace=None, structured_output=None),
        metrics={"latency_ms": latency},
        scores=[Score(name="TestScore", value=score_val, passed=passed, reasoning="Test reason")],
    )


def test_drift_corpus_version_mismatch(run_metadata_mismatch: Tuple[TestRun, TestRun]) -> None:
    r1, r2 = run_metadata_mismatch
    with pytest.raises(ValueError, match="Cannot compare runs with different corpus versions"):
        generate_drift_report(r2, [], r1, [])


def test_drift_per_case_regression(run_metadata: Tuple[TestRun, TestRun]) -> None:
    prev_run, curr_run = run_metadata
    case_id = uuid4()

    # Case passed in previous run
    res_prev = make_result(case_id, passed=True)
    # Case failed in current run
    res_curr = make_result(case_id, passed=False, score_val=0.0)

    report = generate_drift_report(curr_run, [res_curr], prev_run, [res_prev])

    assert len(report.case_drifts) == 1
    drift = report.case_drifts[0]
    assert drift.case_id == case_id
    assert drift.is_regression is True
    assert "Passed -> Failed" in drift.change_description


def test_drift_per_case_improvement(run_metadata: Tuple[TestRun, TestRun]) -> None:
    prev_run, curr_run = run_metadata
    case_id = uuid4()

    # Case failed in previous run
    res_prev = make_result(case_id, passed=False, score_val=0.0)
    # Case passed in current run
    res_curr = make_result(case_id, passed=True, score_val=1.0)

    report = generate_drift_report(curr_run, [res_curr], prev_run, [res_prev])

    assert len(report.case_drifts) == 1
    drift = report.case_drifts[0]
    assert drift.case_id == case_id
    assert drift.is_regression is False
    assert "Failed -> Passed" in drift.change_description


def test_drift_no_change(run_metadata: Tuple[TestRun, TestRun]) -> None:
    prev_run, curr_run = run_metadata
    case_id = uuid4()

    # Both passed
    res_prev = make_result(case_id, passed=True)
    res_curr = make_result(case_id, passed=True)

    report = generate_drift_report(curr_run, [res_curr], prev_run, [res_prev])

    assert len(report.case_drifts) == 0


def test_drift_missing_case(run_metadata: Tuple[TestRun, TestRun]) -> None:
    prev_run, curr_run = run_metadata
    case_id_1 = uuid4()
    case_id_2 = uuid4()

    # Previous has case 1, Current has case 2 (Mismatch scenario despite version match)
    res_prev = make_result(case_id_1, passed=True)
    res_curr = make_result(case_id_2, passed=True)

    report = generate_drift_report(curr_run, [res_curr], prev_run, [res_prev])

    # Should have no matches, so no comparisons
    assert len(report.case_drifts) == 0


def test_drift_aggregates_preserved(run_metadata: Tuple[TestRun, TestRun]) -> None:
    prev_run, curr_run = run_metadata
    case_id = uuid4()

    # 100ms vs 200ms latency (Regression)
    res_prev = make_result(case_id, passed=True, latency=100.0)
    res_curr = make_result(case_id, passed=True, latency=200.0)

    report = generate_drift_report(curr_run, [res_curr], prev_run, [res_prev])

    # Check Aggregates
    latency_metric = next(m for m in report.metrics if m.name == "Average Execution Latency")
    assert latency_metric.previous_value == 100.0
    assert latency_metric.current_value == 200.0
    assert latency_metric.is_regression is True  # Latency increased
