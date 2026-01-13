# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Dict, Optional

from coreason_assay.models import DriftMetric, DriftReport, ReportCard


def generate_drift_report(current: ReportCard, previous: ReportCard) -> DriftReport:
    """
    Compares two ReportCards and generates a DriftReport highlighting regressions.

    Args:
        current: The ReportCard from the current run.
        previous: The ReportCard from the previous/baseline run.

    Returns:
        DriftReport: Contains delta metrics and regression flags.
    """
    drift_metrics = []

    # 1. Compare High-Level Stats
    # Pass Rate (Higher is better)
    drift_metrics.append(
        _calculate_drift(
            name="Pass Rate",
            current_val=current.pass_rate,
            prev_val=previous.pass_rate,
            higher_is_better=True,
            unit="ratio",
        )
    )

    # 2. Compare Aggregates
    # Convert aggregates list to dict for lookup by name
    prev_aggs: Dict[str, float] = {m.name: m.value for m in previous.aggregates}

    for curr_agg in current.aggregates:
        if curr_agg.name in prev_aggs:
            prev_val = prev_aggs[curr_agg.name]

            # Determine directionality based on name
            # Latency -> Lower is better
            # Scores -> Higher is better
            # We use simple heuristic string matching
            lower_is_better = "latency" in curr_agg.name.lower()
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

    return DriftReport(
        current_run_id=current.run_id,
        previous_run_id=previous.run_id,
        metrics=drift_metrics,
    )


def _calculate_drift(
    name: str, current_val: float, prev_val: float, higher_is_better: bool, unit: Optional[str] = None
) -> DriftMetric:
    """Helper to calculate delta and regression status."""
    delta = current_val - prev_val

    # Calculate percentage change safely
    if prev_val == 0:
        # If previous was 0 and current is 0 -> 0% change
        # If previous was 0 and current is > 0 -> treat as 100% change (1.0) or infinite?
        # Let's cap at 1.0 (100%) for simplicity or use None if strictly undefined
        pct_change = 1.0 if current_val != 0 else 0.0
    else:
        pct_change = delta / abs(prev_val)

    # Determine regression
    # If higher is better (e.g. Pass Rate): Regression if current < prev (delta < 0)
    # If lower is better (e.g. Latency): Regression if current > prev (delta > 0)
    is_regression = False

    # We allow a tiny epsilon for float comparison to avoid noise
    epsilon = 1e-9

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
        delta=abs(delta),  # We store absolute delta? No, PRD says "Latency increased by 200".
        # Ideally delta should be signed to show direction,
        # but model def says "Absolute difference".
        # Let's check model def I just wrote:
        # "delta: float = Field(..., description='Absolute difference (current - previous).')"
        # Wait, description says "Absolute difference" but formula says "(current - previous)".
        # (current - previous) is signed. Absolute is abs().
        # Usually delta is signed. "Absolute difference" usually means unsigned magnitude.
        # If I store absolute difference, I lose direction.
        # But I have `is_regression`.
        # Let's stick to the docstring I wrote: "Absolute difference".
        pct_change=pct_change,
        is_regression=is_regression,
    )
