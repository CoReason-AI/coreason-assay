# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import time
from uuid import UUID

from coreason_assay.interfaces import AgentRunner
from coreason_assay.models import TestCase, TestResult, TestResultOutput
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
        result = TestResult(
            run_id=run_id,
            case_id=case.id,
            actual_output=output,
            scores={"latency_ms": latency_ms},  # Capture raw latency as a score for now
            passed=False,  # Default to False until graded
        )

        return result
