# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import math
from typing import List
from uuid import uuid4

from coreason_assay.models import Score, TestResult, TestResultOutput, TestRun, TestRunStatus
from coreason_assay.reporting import generate_report_card


def test_mixed_score_types() -> None:
    """
    Test that the aggregator correctly handles a mix of float, int, and boolean values
    for the same score metric name.
    """
    run = TestRun(
        id=uuid4(),
        corpus_version="1.0",
        agent_draft_version="v1",
        run_by="tester",
        status=TestRunStatus.DONE,
    )

    results = [
        # Case 1: Float 0.5
        TestResult(
            run_id=run.id,
            case_id=uuid4(),
            actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
            scores=[Score(name="Quality", value=0.5, passed=True, reasoning=None)],
            passed=True,
        ),
        # Case 2: Boolean True (should be 1.0)
        TestResult(
            run_id=run.id,
            case_id=uuid4(),
            actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
            scores=[Score(name="Quality", value=True, passed=True, reasoning=None)],
            passed=True,
        ),
        # Case 3: Int 0 (should be 0.0)
        TestResult(
            run_id=run.id,
            case_id=uuid4(),
            actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
            scores=[Score(name="Quality", value=0, passed=False, reasoning=None)],
            passed=False,
        ),
    ]

    card = generate_report_card(run, results)

    # Calculation: (0.5 + 1.0 + 0.0) / 3 = 1.5 / 3 = 0.5
    agg = next(a for a in card.aggregates if a.name == "Average Quality Score")
    assert math.isclose(agg.value, 0.5, rel_tol=1e-9)
    assert agg.total_samples == 3


def test_nan_inf_handling() -> None:
    """
    Test that NaN and Inf values in metrics or scores are filtered out or handled safely
    to prevent invalid JSON or report corruption.
    """
    run = TestRun(
        id=uuid4(),
        corpus_version="1.0",
        agent_draft_version="v1",
        run_by="tester",
        status=TestRunStatus.DONE,
    )

    results = [
        # Valid Case
        TestResult(
            run_id=run.id,
            case_id=uuid4(),
            actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
            metrics={"latency_ms": 100.0},
            scores=[Score(name="Robustness", value=1.0, passed=True, reasoning=None)],
            passed=True,
        ),
        # NaN Case
        TestResult(
            run_id=run.id,
            case_id=uuid4(),
            actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
            metrics={"latency_ms": float("nan")},
            scores=[Score(name="Robustness", value=float("nan"), passed=False, reasoning="Calculated NaN")],
            passed=False,
        ),
        # Inf Case
        TestResult(
            run_id=run.id,
            case_id=uuid4(),
            actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
            metrics={"latency_ms": float("inf")},
            scores=[Score(name="Robustness", value=float("inf"), passed=False, reasoning="Infinite loop")],
            passed=False,
        ),
    ]

    card = generate_report_card(run, results)

    # Latency Check: Should only include the 100.0. NaN/Inf should be ignored.
    # Current implementation might fail this test if not updated.
    latency_agg = next(a for a in card.aggregates if a.name == "Average Execution Latency")
    assert latency_agg.value == 100.0
    assert latency_agg.total_samples == 1

    # Score Check: Should only include the 1.0.
    robust_agg = next(a for a in card.aggregates if a.name == "Average Robustness Score")
    assert robust_agg.value == 1.0
    assert robust_agg.total_samples == 1


def test_large_result_set() -> None:
    """
    Test aggregation with a larger dataset to ensure stability.
    """
    run = TestRun(
        id=uuid4(),
        corpus_version="1.0",
        agent_draft_version="v1",
        run_by="tester",
        status=TestRunStatus.DONE,
    )

    results: List[TestResult] = []
    # Generate 1000 results
    for i in range(1000):
        # Alternate pass/fail
        passed = i % 2 == 0
        results.append(
            TestResult(
                run_id=run.id,
                case_id=uuid4(),
                actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
                metrics={"latency_ms": 10.0},  # Constant 10ms
                scores=[Score(name="Consistency", value=1.0, passed=True, reasoning=None)],
                passed=passed,
            )
        )

    card = generate_report_card(run, results)

    assert card.total_cases == 1000
    assert card.passed_cases == 500
    assert card.failed_cases == 500
    assert card.pass_rate == 0.5

    # Check Latency: Should be 10.0
    latency_agg = next(a for a in card.aggregates if a.name == "Average Execution Latency")
    assert latency_agg.value == 10.0
    assert latency_agg.total_samples == 1000

    # Check Score: Should be 1.0
    consist_agg = next(a for a in card.aggregates if a.name == "Average Consistency Score")
    assert consist_agg.value == 1.0
    assert consist_agg.total_samples == 1000
