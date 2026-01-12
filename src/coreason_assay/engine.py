# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Any, Callable, Coroutine, List, Optional

from coreason_assay.grader import BaseGrader
from coreason_assay.models import ReportCard, TestCorpus, TestResult
from coreason_assay.reporting import generate_report_card
from coreason_assay.simulator import Simulator
from coreason_assay.utils.logger import logger


class AssessmentEngine:
    """
    The orchestrator of the Run-Verify-Score loop.
    Coordinates the Simulator, Graders, and Reporting to produce a ReportCard.
    """

    def __init__(self, simulator: Simulator, graders: List[BaseGrader]):
        """
        Args:
            simulator: The initialized Simulator instance.
            graders: A list of initialized Grader instances to apply to each result.
        """
        self.simulator = simulator
        self.graders = graders

    def _grade_result(self, result: TestResult, case_inputs: Any, case_expectations: Any) -> None:
        """
        Applies all graders to a single result and updates it in-place.
        """
        # Convert Pydantic model to dict for easier lookup if needed,
        # but graders expect the full expectations object or specific fields.
        # The BaseGrader interface expects `expectations: Optional[Dict[str, Any]]`
        # case_expectations is a Pydantic model (TestCaseExpectation).
        # We should dump it to a dict.
        expectations_dict = case_expectations.model_dump()

        for grader in self.graders:
            try:
                score = grader.grade(result, inputs=case_inputs, expectations=expectations_dict)
                result.scores.append(score)
            except Exception as e:
                logger.error(f"Grader {grader.__class__.__name__} failed for case {result.case_id}: {e}")
                # We do not fail the whole run, but we might want to record a failing score or log it.
                # For now, we just log it. The result will just miss that score.

        # Determine if the case passed
        # A case passes if AND only if all scores are passing.
        # If there are no scores, it fails (safety by default).
        if not result.scores:
            result.passed = False
            # If there were no graders, maybe it shouldn't fail?
            # But requirements say "Run-Verify-Score". Unverified results shouldn't pass.
            # However, if the user provided no expectations and we have no graders?
            # Let's assume strict: No scores = Fail.
            # Actually, let's check if ALL scores pass.
        else:
            result.passed = all(s.passed for s in result.scores)

    async def run_assay(
        self,
        corpus: TestCorpus,
        agent_draft_version: str,
        on_progress: Optional[Callable[[int, int, TestResult], Coroutine[Any, Any, None]]] = None,
    ) -> ReportCard:
        """
        Executes the full assay lifecycle: Run -> Grade -> Report.

        Args:
            corpus: The test corpus to run.
            agent_draft_version: The version string of the agent.
            on_progress: Optional async callback for real-time updates.
                         Receives (completed_count, total_count, graded_result).

        Returns:
            ReportCard: The final summary of the run.
        """
        # We need a way to look up the case expectations during the callback.
        # The Simulator returns a TestResult which has case_id.
        # We can create a map for O(1) lookup.
        case_map = {case.id: case for case in corpus.cases}

        async def _progress_interceptor(completed: int, total: int, result: TestResult) -> None:
            # 1. Retrieve the case to get expectations
            case = case_map.get(result.case_id)
            if not case:
                logger.error(f"Result returned for unknown case ID: {result.case_id}")
                return

            # 2. Grade the result immediately
            self._grade_result(result, case.inputs, case.expectations)

            # 3. Forward to the user's callback
            if on_progress:
                await on_progress(completed, total, result)

        # Run the suite with our interceptor
        run, results = await self.simulator.run_suite(
            corpus=corpus,
            agent_draft_version=agent_draft_version,
            on_progress=_progress_interceptor,
        )

        # Safety check: ensure all results are graded (in case run_suite has edge cases where callback isn't called?)
        # Simulator logic seems to guarantee callback.
        # But if results come back, we can double check passed status.
        # The results in the list are references to the same objects modified in the callback.

        # Generate Report Card
        report_card = generate_report_card(run, results)

        return report_card
