# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Dict, List, Optional

from coreason_assay.models import (
    CaseDrift,
    DriftMetric,
    DriftReport,
    ReportCard,
    TestResult,
    TestRun,
)
from coreason_assay.reporting import generate_report_card


def generate_drift_report(
    current_run: TestRun,
    current_results: List[TestResult],
    previous_run: TestRun,
    previous_results: List[TestResult],
) -> DriftReport:
    """
    Compares two Test Runs and generates a DriftReport highlighting regressions.
    Validates that both runs used the same corpus version.

    Args:
        current_run: The current TestRun metadata.
        current_results: List of results from the current run.
        previous_run: The previous TestRun metadata.
        previous_results: List of results from the previous run.

    Returns:
        DriftReport: Contains delta metrics and regression flags.

    Raises:
        ValueError: If corpus versions do not match.
    """
    # 1. Validation
    if current_run.corpus_version != previous_run.corpus_version:
        raise ValueError(
            f"Cannot compare runs with different corpus versions: "
            f"{current_run.corpus_version} vs {previous_run.corpus_version}"
        )

    # 2. Generate Report Cards for Aggregate Comparison
    current_card = generate_report_card(current_run, current_results)
    previous_card = generate_report_card(previous_run, previous_results)

    drift_metrics = _compare_aggregates(current_card, previous_card)

    # 3. Per-Case Comparison
    case_drifts = _compare_cases(current_results, previous_results)

    return DriftReport(
        current_run_id=current_run.id,
        previous_run_id=previous_run.id,
        metrics=drift_metrics,
        case_drifts=case_drifts,
    )


def _compare_aggregates(current: ReportCard, previous: ReportCard) -> List[DriftMetric]:
    """Internal helper to compare high-level stats and aggregates."""
    drift_metrics = []

    # Pass Rate
    drift_metrics.append(
        _calculate_drift(
            name="Pass Rate",
            current_val=current.pass_rate,
            prev_val=previous.pass_rate,
            higher_is_better=True,
            unit="ratio",
        )
    )

    # Aggregates
    prev_aggs: Dict[str, float] = {m.name: m.value for m in previous.aggregates}

    for curr_agg in current.aggregates:
        if curr_agg.name in prev_aggs:
            prev_val = prev_aggs[curr_agg.name]

            unit_lower = (curr_agg.unit or "").lower()
            lower_is_better = unit_lower in ["ms", "s", "seconds"]
            higher_is_better = not lower_is_better

            drift_metrics.append(
                _calculate_drift(
                    name=curr_agg.name,
                    current_val=curr_agg.value,
                    prev_val=prev_val,
                    higher_is_better=higher_is_better,
                    unit=curr_agg.unit,
                )
            )

    return drift_metrics


def _compare_cases(current_results: List[TestResult], previous_results: List[TestResult]) -> List[CaseDrift]:
    """Internal helper to identify per-case regressions."""
    drifts: List[CaseDrift] = []

    # Map previous results by case_id for O(1) lookup
    prev_map = {str(r.case_id): r for r in previous_results}

    for curr in current_results:
        case_id_str = str(curr.case_id)
        if case_id_str not in prev_map:
            # New case in current run? Or mismatch?
            # Since we validated corpus_version, we assume identical cases.
            # If missing, maybe the corpus changed despite version?
            # Or maybe just missing data. We skip for now.
            continue

        prev = prev_map[case_id_str]

        # Check for Status Regression (Pass -> Fail)
        if prev.passed and not curr.passed:
            drifts.append(
                CaseDrift(
                    case_id=curr.case_id,
                    change_description="Status Regression: Passed -> Failed",
                    is_regression=True,
                )
            )
        # Check for Status Improvement (Fail -> Pass)
        elif not prev.passed and curr.passed:
            drifts.append(
                CaseDrift(
                    case_id=curr.case_id,
                    change_description="Status Improvement: Failed -> Passed",
                    is_regression=False,
                )
            )

        # Future: We could add latency regression here
        # e.g., if latency increased by > 50%

    return drifts


def _calculate_drift(
    name: str,
    current_val: float,
    prev_val: float,
    higher_is_better: bool,
    unit: Optional[str] = None,
) -> DriftMetric:
    """Helper to calculate delta and regression status."""
    delta = current_val - prev_val
    epsilon = 1e-9

    if prev_val == 0:
        pct_change = 1.0 if current_val != 0 else 0.0
    else:
        pct_change = delta / abs(prev_val)

    is_regression = False
    if higher_is_better:
        if delta < -epsilon:
            is_regression = True
    else:
        if delta > epsilon:
            is_regression = True

    return DriftMetric(
        name=name,
        unit=unit,
        current_value=current_val,
        previous_value=prev_val,
        delta=abs(delta),
        pct_change=pct_change,
        is_regression=is_regression,
    )
