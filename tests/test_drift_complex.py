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
from coreason_assay.models import AggregateMetric, ReportCard


@pytest.fixture
def run_id_1() -> Generator[UUID, None, None]:
    yield uuid4()


@pytest.fixture
def run_id_2() -> Generator[UUID, None, None]:
    yield uuid4()


def test_drift_directionality_by_unit(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test that directionality is determined by unit, not name.
    """
    prev = ReportCard(
        run_id=run_id_1,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            # "Latency" but unit="score" -> Higher is Better
            AggregateMetric(name="Latency Compliance Score", value=1.0, unit="score", total_samples=10),
            # "Speed" but unit="ms" -> Lower is Better
            AggregateMetric(name="System Speed", value=100.0, unit="ms", total_samples=10),
        ],
    )

    curr = ReportCard(
        run_id=run_id_2,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            # Lower score -> Regression (since Higher is Better)
            AggregateMetric(name="Latency Compliance Score", value=0.5, unit="score", total_samples=10),
            # Higher ms -> Regression (since Lower is Better)
            AggregateMetric(name="System Speed", value=200.0, unit="ms", total_samples=10),
        ],
    )

    report = generate_drift_report(curr, prev)

    # Check Latency Compliance Score
    lcs = next(m for m in report.metrics if m.name == "Latency Compliance Score")
    # Dropped from 1.0 to 0.5. Since unit="score" (Higher is Better), this IS a regression.
    assert lcs.is_regression is True
    assert lcs.delta == pytest.approx(0.5)

    # Check System Speed
    ss = next(m for m in report.metrics if m.name == "System Speed")
    # Increased from 100 to 200. Since unit="ms" (Lower is Better), this IS a regression.
    assert ss.is_regression is True
    assert ss.delta == 200.0 - 100.0


def test_drift_metric_disappearance(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test that metrics present in previous but missing in current are ignored.
    """
    prev = ReportCard(
        run_id=run_id_1,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            AggregateMetric(name="Old Metric", value=1.0, unit="score", total_samples=10),
        ],
    )

    curr = ReportCard(
        run_id=run_id_2,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            # "Old Metric" is missing
        ],
    )

    report = generate_drift_report(curr, prev)

    # Only Pass Rate should be present
    metric_names = [m.name for m in report.metrics]
    assert "Old Metric" not in metric_names
    assert "Pass Rate" in metric_names


def test_drift_epsilon(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test that tiny changes (floating point noise) are ignored.
    """
    prev = ReportCard(
        run_id=run_id_1,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            AggregateMetric(name="Noise Metric", value=1.0, unit="score", total_samples=10),
        ],
    )

    # Change by 1e-10 (less than 1e-9 epsilon)
    tiny_delta = 1e-10
    curr = ReportCard(
        run_id=run_id_2,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            AggregateMetric(name="Noise Metric", value=1.0 - tiny_delta, unit="score", total_samples=10),
        ],
    )

    report = generate_drift_report(curr, prev)
    m = next(m for m in report.metrics if m.name == "Noise Metric")

    # Should NOT be a regression because delta is smaller than epsilon
    assert m.is_regression is False
    assert m.delta == pytest.approx(tiny_delta)


def test_drift_unknown_unit_defaults(run_id_1: UUID, run_id_2: UUID) -> None:
    """
    Test that unknown units default to Higher is Better.
    """
    prev = ReportCard(
        run_id=run_id_1,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            AggregateMetric(name="Mystery Metric", value=10.0, unit="unknown", total_samples=10),
        ],
    )

    # Decrease value
    curr = ReportCard(
        run_id=run_id_2,
        total_cases=10,
        passed_cases=10,
        failed_cases=0,
        pass_rate=1.0,
        aggregates=[
            AggregateMetric(name="Mystery Metric", value=5.0, unit="unknown", total_samples=10),
        ],
    )

    report = generate_drift_report(curr, prev)
    m = next(m for m in report.metrics if m.name == "Mystery Metric")

    # Default assumption: Higher is Better. So drop 10->5 is Regression.
    assert m.is_regression is True
