# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import asyncio
import time
from typing import Any, Callable, Coroutine, List, Optional, Tuple
from uuid import UUID

from coreason_assay.interfaces import AgentRunner
from coreason_assay.models import (
    TestCase,
    TestCorpus,
    TestResult,
    TestResultOutput,
    TestRun,
    TestRunStatus,
)
from coreason_assay.utils.logger import logger


class Simulator:
    """
    The execution harness that runs the agent in a sandbox.
    """

    def __init__(self, runner: AgentRunner):
        """
        Args:
            runner: The concrete implementation of the AgentRunner protocol.
        """
        self.runner = runner

    async def run_case(self, case: TestCase, run_id: UUID) -> TestResult:
        """
        Runs a single test case through the agent runner.

        Args:
            case: The test case to execute.
            run_id: The ID of the parent TestRun.

        Returns:
            TestResult: The result of the execution (unscored).
        """
        logger.info(f"Running test case {case.id} for run {run_id}")

        start_time = time.perf_counter()

        try:
            # Prepare context (merge case context with any global context if needed)
            # For now, we just use the case context.
            context = case.inputs.context

            # Extract tool mocks
            tool_mocks = case.expectations.tool_mocks

            # Invoke the agent
            output = await self.runner.invoke(case.inputs, context, tool_mocks)

        except Exception as e:
            logger.exception(f"Error invoking agent for case {case.id}")
            # Return a failure result with the error message
            output = TestResultOutput(text=None, trace=f"Agent invocation failed: {str(e)}", structured_output=None)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Construct the result
        # Note: Scores are empty for now, will be filled by Grader later.
        # We store raw latency in metrics.
        result = TestResult(
            run_id=run_id,
            case_id=case.id,
            actual_output=output,
            metrics={"latency_ms": latency_ms},
            scores=[],
            passed=False,  # Default to False until graded
        )

        return result

    async def run_suite(
        self,
        corpus: TestCorpus,
        agent_draft_version: str,
        on_progress: Optional[Callable[[int, int, TestResult], Coroutine[Any, Any, None]]] = None,
    ) -> Tuple[TestRun, List[TestResult]]:
        """
        Runs an entire test corpus concurrently.

        Args:
            corpus: The TestCorpus to execute.
            agent_draft_version: Identifier for the agent version being tested.
            on_progress: Optional async callback (completed_count, total_count, last_result).

        Returns:
            Tuple[TestRun, List[TestResult]]: The finalized TestRun object and list of results.
        """
        test_run = TestRun(
            corpus_version=corpus.version,
            agent_draft_version=agent_draft_version,
            status=TestRunStatus.RUNNING,
        )

        logger.info(f"Starting TestRun {test_run.id} for corpus {corpus.id} ({len(corpus.cases)} cases)")

        results: List[TestResult] = []

        # If no cases, return immediately
        if not corpus.cases:
            test_run.status = TestRunStatus.DONE
            return test_run, results

        # Create tasks for all cases
        tasks = [self.run_case(case, test_run.id) for case in corpus.cases]

        total_cases = len(tasks)
        completed_cases = 0

        # Execute concurrently and process as they complete
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
            except Exception as e:
                # This catches errors in run_case itself (which already has a try/except),
                # or task scheduling errors. This is the ultimate fail-safe.
                logger.critical(f"Unexpected error in run_suite task execution: {e}")
                # We can't identify the case ID easily here if future just crashed,
                # but run_case is designed to be safe.
                # If we get here, something is very wrong with the event loop or runner.
                continue

            results.append(result)
            completed_cases += 1

            if on_progress:
                try:
                    await on_progress(completed_cases, total_cases, result)
                except Exception as e:
                    logger.error(f"Error in on_progress callback: {e}")

        test_run.status = TestRunStatus.DONE
        logger.info(f"Completed TestRun {test_run.id}. {completed_cases}/{total_cases} cases processed.")

        return test_run, results
