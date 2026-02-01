# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from coreason_assay.engine import AssessmentEngine
from coreason_assay.grader import BaseGrader
from coreason_assay.models import (
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


class ContextSensitiveGrader(BaseGrader):
    """
    A test grader that requires access to the original inputs
    to determine the score.
    """

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
        agent: Any = None,
    ) -> Score:
        if inputs is None:
            return Score(name="ContextCheck", value=0.0, passed=False, reasoning="Inputs missing")

        # Check if the prompt contains a secret keyword "SECRET"
        passed = "SECRET" in inputs.prompt
        return Score(
            name="ContextCheck",
            value=1.0 if passed else 0.0,
            passed=passed,
            reasoning=f"Prompt was: {inputs.prompt}",
        )


class MutatingGrader(BaseGrader):
    """
    A test grader that modifies the inputs (side-effect).
    """

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
        agent: Any = None,
    ) -> Score:
        if inputs:
            # Modify the context dict
            inputs.context["mutated"] = True
        return Score(name="Mutator", value=1.0, passed=True, reasoning="Mutated inputs")


class ReadingGrader(BaseGrader):
    """
    A test grader that checks if inputs were mutated.
    """

    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
        agent: Any = None,
    ) -> Score:
        if inputs and inputs.context.get("mutated"):
            return Score(name="Reader", value=0.0, passed=False, reasoning="Inputs were mutated!")
        return Score(name="Reader", value=1.0, passed=True, reasoning="Inputs clean")


@pytest.fixture
def mock_simulator() -> MagicMock:
    sim = MagicMock()
    sim.run_suite = AsyncMock()
    return sim


@pytest.mark.asyncio
async def test_context_propagation(mock_simulator: MagicMock) -> None:
    """
    Verify that AssessmentEngine correctly passes 'inputs' to the grader.
    """
    # Create case with secret keyword
    case = TestCase(
        id=uuid4(),
        corpus_id=uuid4(),
        inputs=TestCaseInput(prompt="This is a SECRET message"),
        expectations=TestCaseExpectation(tone=None, text=None, schema_id=None, structure=None),
    )
    corpus = TestCorpus(project_id="p", name="c", version="v", created_by="u", cases=[case])
    run_obj = TestRun(corpus_version="v", agent_draft_version="v", status=TestRunStatus.DONE)
    result_obj = TestResult(
        run_id=run_obj.id,
        case_id=case.id,
        actual_output=TestResultOutput(error=None, text="out", trace=None, structured_output=None),
        scores=[],
        passed=False,
    )

    # Simulator returns result
    async def side_effect(corpus: Any, agent_draft_version: Any, on_progress: Any, agent: Any = None) -> Any:
        if on_progress:
            await on_progress(1, 1, result_obj)
        return run_obj, [result_obj]

    mock_simulator.run_suite.side_effect = side_effect

    grader = ContextSensitiveGrader()
    engine = AssessmentEngine(simulator=mock_simulator, graders=[grader])

    await engine.run_assay(corpus, "v")

    assert len(result_obj.scores) == 1
    score = result_obj.scores[0]
    assert score.passed is True
    assert score.name == "ContextCheck"
    assert score.reasoning is not None and "Prompt was: This is a SECRET message" in score.reasoning


@pytest.mark.asyncio
async def test_input_mutation_side_effect(mock_simulator: MagicMock) -> None:
    """
    Verify behavior when a grader modifies the inputs object.
    Current behavior: Inputs ARE mutable and shared (passed by reference).
    This test documents that side-effects persist between graders.
    """
    case = TestCase(
        id=uuid4(),
        corpus_id=uuid4(),
        inputs=TestCaseInput(prompt="test", context={"original": True}),
        expectations=TestCaseExpectation(tone=None, text=None, schema_id=None, structure=None),
    )
    corpus = TestCorpus(project_id="p", name="c", version="v", created_by="u", cases=[case])
    run_obj = TestRun(corpus_version="v", agent_draft_version="v", status=TestRunStatus.DONE)
    result_obj = TestResult(
        run_id=run_obj.id,
        case_id=case.id,
        actual_output=TestResultOutput(error=None, text="out", trace=None, structured_output=None),
        scores=[],
        passed=False,
    )

    mock_simulator.run_suite.side_effect = lambda c, a, p, ag=None: (run_obj, [result_obj])

    # Note: We need to trigger the callback manually or simulate it
    async def side_effect(corpus: Any, agent_draft_version: Any, on_progress: Any, agent: Any = None) -> Any:
        if on_progress:
            await on_progress(1, 1, result_obj)
        return run_obj, [result_obj]

    mock_simulator.run_suite.side_effect = side_effect

    # Run Mutator first, then Reader
    graders = [MutatingGrader(), ReadingGrader()]
    engine = AssessmentEngine(simulator=mock_simulator, graders=graders)

    await engine.run_assay(corpus, "v")

    assert len(result_obj.scores) == 2
    mutator_score = result_obj.scores[0]
    reader_score = result_obj.scores[1]

    assert mutator_score.name == "Mutator"

    # Reader should FAIL if it detects mutation
    # Since we are passing the same object reference, and Mutator modified it,
    # Reader sees context['mutated'] = True
    assert reader_score.name == "Reader"
    assert reader_score.passed is False
    assert reader_score.reasoning is not None and "Inputs were mutated" in reader_score.reasoning
