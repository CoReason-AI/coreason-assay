# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Generator, List, Tuple
from uuid import UUID, uuid4

import pytest

from coreason_assay.drift import generate_drift_report
from coreason_assay.models import AggregateMetric, DriftReport, ReportCard, Score, TestResult, TestResultOutput, TestRun


@pytest.fixture
def run_id_1() -> Generator[UUID, None, None]:
    yield uuid4()


@pytest.fixture
def run_id_2() -> Generator[UUID, None, None]:
    yield uuid4()


def _mock_data_from_card(card: ReportCard) -> Tuple[TestRun, List[TestResult]]:
    """Helper to backfill TestRun and TestResults from a ReportCard for testing."""
    run = TestRun(
        id=card.run_id,
        corpus_version="v1.0",
        agent_draft_version="draft",
        run_by="tester",
    )
    # We can't easily reconstruct individual results from aggregates perfectly,
    # but for these tests we only care about the aggregates calculation which
    # happens inside generate_drift_report via generate_report_card.
    # However, generate_drift_report calls generate_report_card(run, results).
    # So we need to provide results that WOULD produce this report card.
    # Actually, simpler: The tests here check logic of _compare_aggregates mostly.
    # But now generate_drift_report RE-CALCULATES the report card from results.
    # So passing the manually constructed ReportCard is useless if we don't pass results that match it.

    # We must construct results that match the ReportCard's stats.
    results = []

    # Generate passed/failed cases to match counts
    for _ in range(card.passed_cases):
        results.append(
            TestResult(
                run_id=card.run_id,
                case_id=uuid4(),
                passed=True,
                actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
                scores=[],
            )
        )
    for _ in range(card.failed_cases):
        results.append(
            TestResult(
                run_id=card.run_id,
                case_id=uuid4(),
                passed=False,
                actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
                scores=[],
            )
        )

    # Now inject metrics/scores to match aggregates
    # This is tricky because one result can have multiple scores.
    # We'll just distribute them naively.
    for agg in card.aggregates:
        # If it's a "Latency" metric (from metrics dict)
        if "Latency" in agg.name and agg.unit == "ms":
            # Set latency on the first N results
            for i in range(agg.total_samples):
                if i < len(results):
                    results[i].metrics["latency_ms"] = agg.value

        # If it's a Score
        elif "Score" in agg.name or agg.unit == "score":
            # Create a score object
            for i in range(agg.total_samples):
                if i < len(results):
                    # For score aggregation, we need the score object
                    # name must match "Average {Name} Score" -> Name
                    score_name = agg.name.replace("Average ", "").replace(" Score", "")
                    results[i].scores.append(
                        Score(name=score_name, value=agg.value, passed=agg.value >= 1.0, reasoning="")  # Guess
                    )
        # Handle "Mystery Metric"
        elif agg.name == "Mystery Metric":
            for i in range(agg.total_samples):
                if i < len(results):
                    results[i].scores.append(Score(name="Mystery Metric", value=agg.value, passed=True, reasoning=""))
        # Handle "New Metric"
        elif agg.name == "New Metric":
            for i in range(agg.total_samples):
                if i < len(results):
                    results[i].scores.append(Score(name="New Metric", value=agg.value, passed=True, reasoning=""))

    return run, results


def test_drift_report_basic(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test basic drift calculation with clean data.
    """
    prev_card = ReportCard(
        run_id=run_id_1,
        total_cases=100,
        passed_cases=90,
        failed_cases=10,
        pass_rate=0.9,
        aggregates=[
            AggregateMetric(name="Average Execution Latency", value=1000.0, unit="ms", total_samples=100),
            AggregateMetric(name="Average Faithfulness Score", value=0.95, unit="score", total_samples=100),
        ],
    )
    run_prev, results_prev = _mock_data_from_card(prev_card)

    curr_card = ReportCard(
        run_id=run_id_2,
        total_cases=100,
        passed_cases=80,  # Dropped
        failed_cases=20,
        pass_rate=0.8,
        aggregates=[
            AggregateMetric(name="Average Execution Latency", value=1200.0, unit="ms", total_samples=100),  # Slower
            AggregateMetric(name="Average Faithfulness Score", value=0.95, unit="score", total_samples=100),  # Same
        ],
    )
    run_curr, results_curr = _mock_data_from_card(curr_card)

    report = generate_drift_report(run_curr, results_curr, run_prev, results_prev)

    assert isinstance(report, DriftReport)
    assert report.current_run_id == run_id_2
    assert report.previous_run_id == run_id_1

    # Note: metrics might include "Faithfulness Pass Rate" now because generate_report_card generates both avg
    # and pass rate. The original test expected 3 (Pass Rate + 2 aggregates).
    # generate_report_card produces "Average X Score" and "X Pass Rate".
    # So we might have more.
    # We check existence and values.

    # Check Pass Rate (Regression)
    pr = next(m for m in report.metrics if m.name == "Pass Rate")
    assert pr.current_value == 0.8
    assert pr.previous_value == 0.9
    assert pr.delta == pytest.approx(0.1)
    assert pr.is_regression is True

    # Check Latency (Regression: 1000 -> 1200 is bad)
    lat = next(m for m in report.metrics if m.name == "Average Execution Latency")
    assert lat.current_value == 1200.0
    assert lat.previous_value == 1000.0
    assert lat.delta == pytest.approx(200.0)
    assert lat.is_regression is True

    # Check Faithfulness (No Change)
    faith = next(m for m in report.metrics if m.name == "Average Faithfulness Score")
    assert faith.current_value == 0.95
    assert faith.previous_value == 0.95
    assert faith.delta == 0.0
    assert faith.is_regression is False


def test_drift_improvement(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test that improvements are NOT flagged as regressions.
    """
    prev_card = ReportCard(
        run_id=run_id_1,
        total_cases=10,
        passed_cases=5,
        failed_cases=5,
        pass_rate=0.5,
        aggregates=[
            AggregateMetric(name="Average Execution Latency", value=2000.0, unit="ms", total_samples=10),
        ],
    )
    run_prev, results_prev = _mock_data_from_card(prev_card)

    curr_card = ReportCard(
        run_id=run_id_2,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            AggregateMetric(name="Average Execution Latency", value=1500.0, unit="ms", total_samples=10),
        ],
    )
    run_curr, results_curr = _mock_data_from_card(curr_card)

    report = generate_drift_report(run_curr, results_curr, run_prev, results_prev)

    # Pass Rate Improved
    pr = next(m for m in report.metrics if m.name == "Pass Rate")
    assert pr.is_regression is False
    assert pr.current_value == 1.0

    # Latency Improved (Lower)
    lat = next(m for m in report.metrics if m.name == "Average Execution Latency")
    assert lat.current_value == 1500.0
    assert lat.is_regression is False


def test_drift_missing_aggregates(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test handling of metrics present in current but not previous.
    """
    prev_card = ReportCard(
        run_id=run_id_1, total_cases=10, passed_cases=10, failed_cases=0, pass_rate=1.0, aggregates=[]
    )
    run_prev, results_prev = _mock_data_from_card(prev_card)

    curr_card = ReportCard(
        run_id=run_id_2,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[AggregateMetric(name="New Metric", value=10.0, unit=None, total_samples=10)],
    )
    run_curr, results_curr = _mock_data_from_card(curr_card)

    report = generate_drift_report(run_curr, results_curr, run_prev, results_prev)

    # "New Metric" should NOT appear because we can't compare it
    # Note: "Average New Metric Score" might be generated by logic
    metric_names = [m.name for m in report.metrics]
    assert "New Metric" not in metric_names
    assert "Average New Metric Score" not in metric_names  # Name generated by logic
    assert "Pass Rate" in metric_names


def test_drift_zero_division(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test percentage calculation when previous value is zero.
    """
    prev_card = ReportCard(
        run_id=run_id_1,
        total_cases=10,
        passed_cases=0,
        failed_cases=10,
        pass_rate=0.0,  # 0.0 start
        aggregates=[],
    )
    run_prev, results_prev = _mock_data_from_card(prev_card)

    curr_card = ReportCard(
        run_id=run_id_2, total_cases=10, passed_cases=5, failed_cases=5, pass_rate=0.5, aggregates=[]
    )
    run_curr, results_curr = _mock_data_from_card(curr_card)

    report = generate_drift_report(run_curr, results_curr, run_prev, results_prev)
    pr = next(m for m in report.metrics if m.name == "Pass Rate")

    assert pr.previous_value == 0.0
    assert pr.pct_change == 1.0  # Logic caps at 1.0
