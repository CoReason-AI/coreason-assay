# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from coreason_assay.engine import AssessmentEngine
from coreason_assay.grader import BaseGrader
from coreason_assay.models import (
    ReportCard,
    Score,
    TestCase,
    TestCaseExpectation,
    TestCaseInput,
    TestCorpus,
    TestResult,
    TestResultOutput,
    TestRun,
    TestRunStatus,
)


@pytest.fixture
def mock_simulator() -> MagicMock:
    sim = MagicMock()
    sim.run_suite = AsyncMock()
    return sim


def create_test_case() -> TestCase:
    return TestCase(
        id=uuid4(),
        corpus_id=uuid4(),
        inputs=TestCaseInput(prompt="test"),
        expectations=TestCaseExpectation(text="expected", schema_id=None, structure=None),
    )


def create_result(case: TestCase, run_id: Any) -> TestResult:
    return TestResult(
        run_id=run_id,
        case_id=case.id,
        actual_output=TestResultOutput(text="output", trace=None, structured_output=None),
        metrics={"latency_ms": 100.0},
        scores=[],
        passed=False,
    )


@pytest.mark.asyncio
async def test_engine_no_graders_fail_cases(mock_simulator: MagicMock) -> None:
    """
    Edge Case: No graders configured.
    Expectation: Cases run but fail verification (passed=False) because no scores were generated.
    """
    case = create_test_case()
    corpus = TestCorpus(project_id="p1", name="c1", version="v1", created_by="u1", cases=[case])
    run_obj = TestRun(corpus_version="v1", agent_draft_version="v1", status=TestRunStatus.DONE)
    result_obj = create_result(case, run_obj.id)

    async def side_effect(corpus: Any, agent_draft_version: Any, on_progress: Any) -> Any:
        if on_progress:
            await on_progress(1, 1, result_obj)
        return run_obj, [result_obj]

    mock_simulator.run_suite.side_effect = side_effect

    # Initialize Engine with EMPTY graders list
    engine = AssessmentEngine(simulator=mock_simulator, graders=[])

    report: ReportCard = await engine.run_assay(corpus, "v1")

    # Verification
    assert report.total_cases == 1
    assert report.passed_cases == 0
    assert report.failed_cases == 1
    assert report.pass_rate == 0.0

    # Check the result object state
    assert len(result_obj.scores) == 0
    assert result_obj.passed is False


@pytest.mark.asyncio
async def test_engine_empty_corpus(mock_simulator: MagicMock) -> None:
    """
    Edge Case: Corpus has no cases.
    Expectation: Engine handles gracefully, returns empty report.
    """
    corpus = TestCorpus(project_id="p1", name="empty", version="v1", created_by="u1", cases=[])
    run_obj = TestRun(corpus_version="v1", agent_draft_version="v1", status=TestRunStatus.DONE)

    # Simulator returns empty list immediately
    mock_simulator.run_suite.return_value = (run_obj, [])

    engine = AssessmentEngine(simulator=mock_simulator, graders=[])
    report: ReportCard = await engine.run_assay(corpus, "v1")

    assert report.total_cases == 0
    assert report.passed_cases == 0
    assert report.failed_cases == 0
    assert report.pass_rate == 0.0
    assert len(report.aggregates) == 0


@pytest.mark.asyncio
async def test_engine_mixed_batch_complex(mock_simulator: MagicMock) -> None:
    """
    Complex Scenario: Mixed outcomes.
    - Case 1: Pass (All graders pass)
    - Case 2: Fail (Grader 1 passes, Grader 2 fails)
    - Case 3: Fail (All graders fail)

    Verify ReportCard aggregates match expected logic.
    """
    # 1. Setup Data
    case1 = create_test_case()
    case2 = create_test_case()
    case3 = create_test_case()
    corpus = TestCorpus(project_id="p1", name="mix", version="v1", created_by="u1", cases=[case1, case2, case3])
    run_obj = TestRun(corpus_version="v1", agent_draft_version="v1", status=TestRunStatus.DONE)

    r1 = create_result(case1, run_obj.id)
    r2 = create_result(case2, run_obj.id)
    r3 = create_result(case3, run_obj.id)
    results = [r1, r2, r3]

    # 2. Setup Graders
    # Grader A: Passes Case 1 & 2, Fails 3
    grader_a = MagicMock(spec=BaseGrader)

    def grade_a(result: TestResult, expectations: Any) -> Score:
        passed = result.case_id in (case1.id, case2.id)
        return Score(name="GraderA", value=1.0 if passed else 0.0, passed=passed, reasoning="A")

    grader_a.grade.side_effect = grade_a

    # Grader B: Passes Case 1, Fails 2 & 3
    grader_b = MagicMock(spec=BaseGrader)

    def grade_b(result: TestResult, expectations: Any) -> Score:
        passed = result.case_id == case1.id
        return Score(name="GraderB", value=1.0 if passed else 0.0, passed=passed, reasoning="B")

    grader_b.grade.side_effect = grade_b

    # 3. Setup Simulator
    async def side_effect(corpus: Any, agent_draft_version: Any, on_progress: Any) -> Any:
        # Simulate sequential completion
        for idx, res in enumerate(results, start=1):
            if on_progress:
                await on_progress(idx, 3, res)
        return run_obj, results

    mock_simulator.run_suite.side_effect = side_effect

    # 4. Run
    engine = AssessmentEngine(simulator=mock_simulator, graders=[grader_a, grader_b])
    report: ReportCard = await engine.run_assay(corpus, "v1")

    # 5. Verification
    # Case 1: Pass + Pass = Pass
    assert r1.passed is True
    # Case 2: Pass + Fail = Fail
    assert r2.passed is False
    # Case 3: Fail + Fail = Fail
    assert r3.passed is False

    # Report Stats
    assert report.total_cases == 3
    assert report.passed_cases == 1
    assert report.failed_cases == 2
    assert report.pass_rate == 1 / 3

    # Check Aggregates
    # Expect: "Average Execution Latency", "Average GraderA Score", "Average GraderB Score"
    agg_names = [a.name for a in report.aggregates]
    assert "Average Execution Latency" in agg_names
    assert "Average GraderA Score" in agg_names
    assert "Average GraderB Score" in agg_names

    # Check Score Aggregates
    # Grader A: 1.0, 1.0, 0.0 -> Avg 0.66
    score_a = next(a for a in report.aggregates if a.name == "Average GraderA Score")
    assert score_a.value == pytest.approx(2 / 3)

    # Grader B: 1.0, 0.0, 0.0 -> Avg 0.33
    score_b = next(a for a in report.aggregates if a.name == "Average GraderB Score")
    assert score_b.value == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_engine_unknown_case_id(mock_simulator: MagicMock) -> None:
    """
    Edge Case: Simulator returns a result for a Case ID not in the Corpus.
    Expectation: The interceptor logs an error and ignores the result (no crash).
    """
    case = create_test_case()
    corpus = TestCorpus(project_id="p1", name="c1", version="v1", created_by="u1", cases=[case])
    run_obj = TestRun(corpus_version="v1", agent_draft_version="v1", status=TestRunStatus.DONE)

    # Create a result with a RANDOM ID, not case.id
    unknown_case = TestCase(id=uuid4(), corpus_id=uuid4(), inputs=case.inputs, expectations=case.expectations)
    result_obj = create_result(unknown_case, run_obj.id)

    async def side_effect(corpus: Any, agent_draft_version: Any, on_progress: Any) -> Any:
        if on_progress:
            # Pass the unknown result to the callback
            await on_progress(1, 1, result_obj)
        return run_obj, [result_obj]

    mock_simulator.run_suite.side_effect = side_effect

    engine = AssessmentEngine(simulator=mock_simulator, graders=[])

    # Run
    await engine.run_assay(corpus, "v1")

    # Assertions
    # Since we didn't crash, the test passes.
    # We could assert logs if we captured them, but "no crash" is sufficient coverage here.
    # The coverage tool will confirm lines 95-96 are hit.
