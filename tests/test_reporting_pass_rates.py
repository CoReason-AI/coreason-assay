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


def test_reporting_pass_rates_and_averages() -> None:
    """
    Verifies that the Report Card generates both 'Average {Name} Score'
    and '{Name} Pass Rate' for every score dimension.
    """
    run_id = uuid4()
    test_run = TestRun(
        id=run_id,
        corpus_version="v1",
        agent_draft_version="d1",
        status=TestRunStatus.DONE,
    )

    # Case 1: All Good
    # Latency: 100ms (Pass)
    # Reasoning: 1.0 (Pass)
    # JsonSchema: 1.0 (Pass)
    result_1 = TestResult(
        run_id=run_id,
        case_id=uuid4(),
        passed=True,
        actual_output=TestResultOutput(
            text="ok",
            trace=None,
            structured_output={},
        ),
        metrics={"latency_ms": 100},
        scores=[
            Score(name="Latency", value=100.0, passed=True, reasoning="good"),
            Score(name="ReasoningAlignment", value=1.0, passed=True, reasoning="good"),
            Score(name="JsonSchema", value=1.0, passed=True, reasoning="good"),
        ],
    )

    # Case 2: Mixed Failure
    # Latency: 6000ms (Fail)
    # Reasoning: 0.5 (Fail)
    # JsonSchema: 1.0 (Pass)
    result_2 = TestResult(
        run_id=run_id,
        case_id=uuid4(),
        passed=False,
        actual_output=TestResultOutput(
            text="slow and wrong",
            trace=None,
            structured_output={},
        ),
        metrics={"latency_ms": 6000},
        scores=[
            Score(name="Latency", value=6000.0, passed=False, reasoning="too slow"),
            Score(name="ReasoningAlignment", value=0.5, passed=False, reasoning="missing steps"),
            Score(name="JsonSchema", value=1.0, passed=True, reasoning="valid json"),
        ],
    )

    results = [result_1, result_2]

    report = generate_report_card(test_run, results)

    # Helper to find metric by name
    def get_metric(name: str) -> Optional[AggregateMetric]:
        for m in report.aggregates:
            if m.name == name:
                return m
        return None

    # 1. Latency
    # Avg Value: (100 + 6000) / 2 = 3050.0
    # Pass Rate: 1 / 2 = 0.5
    avg_latency = get_metric("Average Latency Score")
    pass_latency = get_metric("Latency Pass Rate")

    assert avg_latency is not None, "Missing 'Average Latency Score'"
    assert avg_latency.value == 3050.0

    assert pass_latency is not None, "Missing 'Latency Pass Rate'"
    assert pass_latency.value == 0.5

    # 2. ReasoningAlignment
    # Avg Value: (1.0 + 0.5) / 2 = 0.75
    # Pass Rate: 1 / 2 = 0.5
    avg_reasoning = get_metric("Average ReasoningAlignment Score")
    pass_reasoning = get_metric("ReasoningAlignment Pass Rate")

    assert avg_reasoning is not None
    assert avg_reasoning.value == 0.75

    assert pass_reasoning is not None
    assert pass_reasoning.value == 0.5

    # 3. JsonSchema
    # Avg Value: (1.0 + 1.0) / 2 = 1.0
    # Pass Rate: 2 / 2 = 1.0
    avg_schema = get_metric("Average JsonSchema Score")
    pass_schema = get_metric("JsonSchema Pass Rate")

    assert avg_schema is not None
    assert avg_schema.value == 1.0

    assert pass_schema is not None
    assert pass_schema.value == 1.0
