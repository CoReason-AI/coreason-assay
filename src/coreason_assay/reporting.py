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
from typing import Dict, List, Union

from coreason_assay.models import AggregateMetric, ReportCard, TestResult, TestRun


def generate_report_card(run: TestRun, results: List[TestResult]) -> ReportCard:
    """
    Generates a ReportCard from a TestRun and its results.

    Calculates:
    - Global Pass Rate
    - Average Execution Latency (raw execution time)
    - Metric-specific aggregates (e.g. Average Faithfulness Score)

    Args:
        run: The TestRun object.
        results: List of graded TestResults.

    Returns:
        ReportCard: The summarized report.
    """
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r.passed)
    failed_cases = total_cases - passed_cases
    pass_rate = (passed_cases / total_cases) if total_cases > 0 else 0.0

    aggregates: List[AggregateMetric] = []

    # Helper to clean values
    def is_valid_number(n: Union[float, int]) -> bool:
        return isinstance(n, (int, float)) and not math.isnan(n) and not math.isinf(n)

    # 1. Global Latency Aggregate (Raw Execution Time)
    latencies: List[float] = []
    for r in results:
        l_ms = r.metrics.get("latency_ms")
        if l_ms is not None:
            val = float(l_ms)
            if is_valid_number(val):
                latencies.append(val)

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        aggregates.append(
            AggregateMetric(
                name="Average Execution Latency",
                value=avg_latency,
                unit="ms",
                total_samples=len(latencies),
            )
        )

    # 2. Score-specific Aggregates
    # We group scores by name (e.g. "Faithfulness", "JsonSchema")
    score_groups: Dict[str, List[float]] = {}

    for result in results:
        for score in result.scores:
            val = score.value
            if isinstance(val, bool):
                val = 1.0 if val else 0.0

            # Ensure we handle numeric conversion safely and filter nan/inf
            if isinstance(val, (int, float)):
                f_val = float(val)
                if is_valid_number(f_val):
                    if score.name not in score_groups:
                        score_groups[score.name] = []
                    score_groups[score.name].append(f_val)

    for name, values in score_groups.items():
        if values:
            avg_val = sum(values) / len(values)
            aggregates.append(
                AggregateMetric(
                    name=f"Average {name} Score",
                    value=avg_val,
                    unit="score",
                    total_samples=len(values),
                )
            )

    return ReportCard(
        run_id=run.id,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=pass_rate,
        aggregates=aggregates,
    )
