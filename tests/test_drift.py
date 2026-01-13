# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Generator
from uuid import UUID, uuid4

import pytest

from coreason_assay.drift import generate_drift_report
from coreason_assay.models import AggregateMetric, DriftReport, ReportCard


@pytest.fixture
def run_id_1() -> Generator[UUID, None, None]:
    yield uuid4()


@pytest.fixture
def run_id_2() -> Generator[UUID, None, None]:
    yield uuid4()


def test_drift_report_basic(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test basic drift calculation with clean data.
    """
    prev = ReportCard(
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

    curr = ReportCard(
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

    report = generate_drift_report(curr, prev)

    assert isinstance(report, DriftReport)
    assert report.current_run_id == run_id_2
    assert report.previous_run_id == run_id_1
    assert len(report.metrics) == 3  # Pass Rate + 2 aggregates

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
    prev = ReportCard(
        run_id=run_id_1,
        total_cases=10,
        passed_cases=5,
        failed_cases=5,
        pass_rate=0.5,
        aggregates=[
            AggregateMetric(name="Average Execution Latency", value=2000.0, unit="ms", total_samples=10),
        ],
    )

    curr = ReportCard(
        run_id=run_id_2,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            AggregateMetric(name="Average Execution Latency", value=1500.0, unit="ms", total_samples=10),
        ],
    )

    report = generate_drift_report(curr, prev)

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
    prev = ReportCard(run_id=run_id_1, total_cases=10, passed_cases=10, failed_cases=0, pass_rate=1.0, aggregates=[])

    curr = ReportCard(
        run_id=run_id_2,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[AggregateMetric(name="New Metric", value=10.0, unit=None, total_samples=10)],
    )

    report = generate_drift_report(curr, prev)

    # "New Metric" should NOT appear because we can't compare it
    metric_names = [m.name for m in report.metrics]
    assert "New Metric" not in metric_names
    assert "Pass Rate" in metric_names


def test_drift_zero_division(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test percentage calculation when previous value is zero.
    """
    prev = ReportCard(
        run_id=run_id_1,
        total_cases=10,
        passed_cases=0,
        failed_cases=10,
        pass_rate=0.0,  # 0.0 start
        aggregates=[],
    )
    curr = ReportCard(run_id=run_id_2, total_cases=10, passed_cases=5, failed_cases=5, pass_rate=0.5, aggregates=[])

    report = generate_drift_report(curr, prev)
    pr = next(m for m in report.metrics if m.name == "Pass Rate")

    assert pr.previous_value == 0.0
    assert pr.pct_change == 1.0  # Logic caps at 1.0
