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

from coreason_assay.models import Score, TestResult, TestResultOutput, TestRun, TestRunStatus
from coreason_assay.reporting import generate_report_card


def test_generate_report_card_basic() -> None:
    """
    Test generating a report card with a mix of passed and failed cases.
    """
    run = TestRun(
        id=uuid4(),
        corpus_version="1.0",
        agent_draft_version="v1",
        run_by="tester",
        status=TestRunStatus.DONE,
    )

    # Case 1: Passed, Latency 100ms
    r1 = TestResult(
        run_id=run.id,
        case_id=uuid4(),
        actual_output=TestResultOutput(text="foo", trace=None, structured_output=None),
        metrics={"latency_ms": 100.0},
        scores=[
            Score(name="Latency", value=100.0, passed=True, reasoning=None),
            Score(name="Faithfulness", value=1.0, passed=True, reasoning=None),
        ],
        passed=True,
    )

    # Case 2: Failed, Latency 200ms
    r2 = TestResult(
        run_id=run.id,
        case_id=uuid4(),
        actual_output=TestResultOutput(text="bar", trace=None, structured_output=None),
        metrics={"latency_ms": 200.0},
        scores=[
            Score(name="Latency", value=200.0, passed=True, reasoning=None),
            Score(name="Faithfulness", value=0.0, passed=False, reasoning=None),  # Failed here
        ],
        passed=False,
    )

    card = generate_report_card(run, [r1, r2])

    assert card.run_id == run.id
    assert card.total_cases == 2
    assert card.passed_cases == 1
    assert card.failed_cases == 1
    assert card.pass_rate == 0.5

    # Check Aggregates
    # 1. Raw Execution Latency (metric)
    # (100 + 200) / 2 = 150
    latency_agg = next(a for a in card.aggregates if a.name == "Average Execution Latency")
    assert latency_agg.value == 150.0
    assert latency_agg.unit == "ms"
    assert latency_agg.total_samples == 2

    # 2. Latency Score (from Grader)
    # (100 + 200) / 2 = 150
    latency_score_agg = next(a for a in card.aggregates if a.name == "Average Latency Score")
    assert latency_score_agg.value == 150.0
    assert latency_score_agg.unit == "score"

    # 3. Faithfulness Score
    # (1.0 + 0.0) / 2 = 0.5
    faith_agg = next(a for a in card.aggregates if a.name == "Average Faithfulness Score")
    assert faith_agg.value == 0.5
    assert faith_agg.total_samples == 2


def test_generate_report_card_empty() -> None:
    """
    Test generating a report card with zero results.
    """
    run = TestRun(
        id=uuid4(),
        corpus_version="1.0",
        agent_draft_version="v1",
        run_by="tester",
        status=TestRunStatus.DONE,
    )

    card = generate_report_card(run, [])

    assert card.total_cases == 0
    assert card.pass_rate == 0.0
    assert len(card.aggregates) == 0


def test_generate_report_card_boolean_scores() -> None:
    """
    Test aggregation of boolean scores (should be converted to 1.0/0.0).
    """
    run = TestRun(
        id=uuid4(),
        corpus_version="1.0",
        agent_draft_version="v1",
        run_by="tester",
    )

    r1 = TestResult(
        run_id=run.id,
        case_id=uuid4(),
        actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
        scores=[Score(name="BoolMetric", value=True, passed=True, reasoning=None)],
        passed=True,
    )
    r2 = TestResult(
        run_id=run.id,
        case_id=uuid4(),
        actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
        scores=[Score(name="BoolMetric", value=False, passed=False, reasoning=None)],
        passed=False,
    )

    card = generate_report_card(run, [r1, r2])

    agg = next(a for a in card.aggregates if a.name == "Average BoolMetric Score")
    assert agg.value == 0.5  # (1 + 0) / 2
    assert agg.total_samples == 2


def test_generate_report_card_missing_latency() -> None:
    """
    Test aggregation when some results miss latency metrics.
    """
    run = TestRun(
        id=uuid4(),
        corpus_version="1.0",
        agent_draft_version="v1",
        run_by="tester",
    )

    r1 = TestResult(
        run_id=run.id,
        case_id=uuid4(),
        actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
        metrics={"latency_ms": 100.0},
        passed=True,
    )
    r2 = TestResult(
        run_id=run.id,
        case_id=uuid4(),
        actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
        metrics={},  # No latency
        passed=True,
    )

    card = generate_report_card(run, [r1, r2])

    latency_agg = next(a for a in card.aggregates if a.name == "Average Execution Latency")
    assert latency_agg.value == 100.0
    assert latency_agg.total_samples == 1
