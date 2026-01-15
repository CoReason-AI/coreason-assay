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


@pytest.fixture
def mock_grader() -> MagicMock:
    grader = MagicMock(spec=BaseGrader)
    # Default to passing
    grader.grade.return_value = Score(
        name="TestScore",
        value=1.0,
        passed=True,
        reasoning="Passed",
    )
    return grader


@pytest.fixture
def simple_corpus() -> TestCorpus:
    case_id = uuid4()
    return TestCorpus(
        project_id="p1",
        name="c1",
        version="v1",
        created_by="u1",
        cases=[
            TestCase(
                id=case_id,
                corpus_id=uuid4(),
                inputs=TestCaseInput(prompt="hi"),
                expectations=TestCaseExpectation(tone=None, text="hello", schema_id=None, structure=None),
            )
        ],
    )


@pytest.mark.asyncio
async def test_run_assay_basic_flow(
    mock_simulator: MagicMock, mock_grader: MagicMock, simple_corpus: TestCorpus
) -> None:
    # Setup Data
    case = simple_corpus.cases[0]
    run_obj = TestRun(corpus_version="v1", agent_draft_version="v1", status=TestRunStatus.DONE)
    result_obj = TestResult(
        run_id=run_obj.id,
        case_id=case.id,
        actual_output=TestResultOutput(text="hello", trace=None, structured_output=None),
        metrics={"latency_ms": 100},
        scores=[],
        passed=False,
    )

    # Configure run_suite to call the callback and return results
    async def side_effect(corpus: TestCorpus, agent_draft_version: str, on_progress: Any) -> Any:
        if on_progress:
            await on_progress(1, 1, result_obj)
        return run_obj, [result_obj]

    mock_simulator.run_suite.side_effect = side_effect

    # Initialize Engine
    engine = AssessmentEngine(simulator=mock_simulator, graders=[mock_grader])

    # Run
    report: ReportCard = await engine.run_assay(simple_corpus, "v1")

    # Verify Report
    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.pass_rate == 1.0

    # Verify Grading happened
    assert len(result_obj.scores) == 1
    assert result_obj.scores[0].name == "TestScore"
    assert result_obj.passed is True

    # Verify Grader called with correct expectations
    mock_grader.grade.assert_called_once()
    call_args = mock_grader.grade.call_args
    # call_args.args is (result,)
    # call_args.kwargs is {'inputs': ..., 'expectations': ...}
    assert call_args.args[0] == result_obj
    assert call_args.kwargs["expectations"]["text"] == "hello"


@pytest.mark.asyncio
async def test_run_assay_failure(mock_simulator: MagicMock, mock_grader: MagicMock, simple_corpus: TestCorpus) -> None:
    # Setup Data
    case = simple_corpus.cases[0]
    run_obj = TestRun(corpus_version="v1", agent_draft_version="v1", status=TestRunStatus.DONE)
    result_obj = TestResult(
        run_id=run_obj.id,
        case_id=case.id,
        actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
        scores=[],
        passed=False,
    )

    # Mock Grader Failure
    mock_grader.grade.return_value = Score(
        name="TestScore",
        value=0.0,
        passed=False,
        reasoning="Failed",
    )

    async def side_effect(corpus: TestCorpus, agent_draft_version: str, on_progress: Any) -> Any:
        if on_progress:
            await on_progress(1, 1, result_obj)
        return run_obj, [result_obj]

    mock_simulator.run_suite.side_effect = side_effect

    engine = AssessmentEngine(simulator=mock_simulator, graders=[mock_grader])
    report: ReportCard = await engine.run_assay(simple_corpus, "v1")

    assert report.passed_cases == 0
    assert report.failed_cases == 1
    assert report.pass_rate == 0.0
    assert result_obj.passed is False


@pytest.mark.asyncio
async def test_on_progress_passthrough(
    mock_simulator: MagicMock, mock_grader: MagicMock, simple_corpus: TestCorpus
) -> None:
    # Setup
    case = simple_corpus.cases[0]
    run_obj = TestRun(corpus_version="v1", agent_draft_version="v1", status=TestRunStatus.DONE)
    result_obj = TestResult(
        run_id=run_obj.id,
        case_id=case.id,
        actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
        scores=[],
        passed=False,
    )

    async def side_effect(corpus: TestCorpus, agent_draft_version: str, on_progress: Any) -> Any:
        if on_progress:
            await on_progress(1, 1, result_obj)
        return run_obj, [result_obj]

    mock_simulator.run_suite.side_effect = side_effect

    engine = AssessmentEngine(simulator=mock_simulator, graders=[mock_grader])
    user_callback = AsyncMock()

    await engine.run_assay(simple_corpus, "v1", on_progress=user_callback)

    user_callback.assert_called_once()
    args = user_callback.call_args[0]
    # args are (completed, total, result)
    result_arg = args[2]
    # Ensure the result passed to callback is ALREADY graded
    assert len(result_arg.scores) == 1
    assert result_arg.scores[0].name == "TestScore"


@pytest.mark.asyncio
async def test_multiple_graders(mock_simulator: MagicMock, simple_corpus: TestCorpus) -> None:
    # Setup two graders
    g1 = MagicMock(spec=BaseGrader)
    g1.grade.return_value = Score(name="G1", value=1.0, passed=True, reasoning="Pass")
    g2 = MagicMock(spec=BaseGrader)
    g2.grade.return_value = Score(name="G2", value=0.0, passed=False, reasoning="Fail")

    case = simple_corpus.cases[0]
    run_obj = TestRun(corpus_version="v1", agent_draft_version="v1", status=TestRunStatus.DONE)
    result_obj = TestResult(
        run_id=run_obj.id,
        case_id=case.id,
        actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
        scores=[],
        passed=False,
    )

    async def side_effect(corpus: TestCorpus, agent_draft_version: str, on_progress: Any) -> Any:
        if on_progress:
            await on_progress(1, 1, result_obj)
        return run_obj, [result_obj]

    mock_simulator.run_suite.side_effect = side_effect

    engine = AssessmentEngine(simulator=mock_simulator, graders=[g1, g2])
    report: ReportCard = await engine.run_assay(simple_corpus, "v1")

    assert len(result_obj.scores) == 2
    # Result should be failed because G2 failed
    assert result_obj.passed is False
    assert report.passed_cases == 0


@pytest.mark.asyncio
async def test_grader_exception(mock_simulator: MagicMock, simple_corpus: TestCorpus) -> None:
    # Grader raises exception
    g1 = MagicMock(spec=BaseGrader)
    g1.grade.side_effect = Exception("Boom")

    case = simple_corpus.cases[0]
    run_obj = TestRun(corpus_version="v1", agent_draft_version="v1", status=TestRunStatus.DONE)
    result_obj = TestResult(
        run_id=run_obj.id,
        case_id=case.id,
        actual_output=TestResultOutput(text=None, trace=None, structured_output=None),
        scores=[],
        passed=False,
    )

    async def side_effect(corpus: TestCorpus, agent_draft_version: str, on_progress: Any) -> Any:
        if on_progress:
            await on_progress(1, 1, result_obj)
        return run_obj, [result_obj]

    mock_simulator.run_suite.side_effect = side_effect

    engine = AssessmentEngine(simulator=mock_simulator, graders=[g1])
    report: ReportCard = await engine.run_assay(simple_corpus, "v1")

    # Should not crash, but result has no scores and fails
    assert len(result_obj.scores) == 0
    assert result_obj.passed is False
    assert report.failed_cases == 1
