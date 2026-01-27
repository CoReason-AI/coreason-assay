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
from typing import Any, Dict
from uuid import uuid4

import pytest

from coreason_assay.interfaces import AgentRunner
from coreason_assay.models import (
    TestCase,
    TestCaseExpectation,
    TestCaseInput,
    TestCorpus,
    TestResult,
    TestResultOutput,
    TestRun,
    TestRunStatus,
)
from coreason_assay.simulator import Simulator

# --- Mocks ---


class AsyncSleepAgentRunner(AgentRunner):
    """
    Simulates a slow agent to test concurrency.
    """

    def __init__(self, delay_s: float, return_text: str = "Slow Response"):
        self.delay_s = delay_s
        self.return_text = return_text
        self.call_count = 0

    async def invoke(
        self, inputs: TestCaseInput, context: Dict[str, Any], tool_mocks: Dict[str, Any]
    ) -> TestResultOutput:
        self.call_count += 1
        await asyncio.sleep(self.delay_s)
        return TestResultOutput(text=self.return_text, trace="Slow trace", structured_output=None)


class FlakyAgentRunner(AgentRunner):
    """
    Simulates an agent that fails for specific cases.
    """

    def __init__(self, failure_trigger_prompt: str = "FAIL_ME"):
        self.failure_trigger_prompt = failure_trigger_prompt

    async def invoke(
        self, inputs: TestCaseInput, context: Dict[str, Any], tool_mocks: Dict[str, Any]
    ) -> TestResultOutput:
        if inputs.prompt == self.failure_trigger_prompt:
            raise RuntimeError("Intentional Crash")
        return TestResultOutput(text="OK", trace="OK Trace", structured_output=None)


# --- Fixtures ---


@pytest.fixture
def basic_corpus() -> TestCorpus:
    return TestCorpus(
        project_id="proj_123",
        name="Test Corpus",
        version="1.0.0",
        created_by="tester",
        cases=[
            TestCase(
                corpus_id=uuid4(),
                inputs=TestCaseInput(prompt="Case 1"),
                expectations=TestCaseExpectation(tone=None, text="Expected", schema_id=None, structure=None),
            ),
            TestCase(
                corpus_id=uuid4(),
                inputs=TestCaseInput(prompt="Case 2"),
                expectations=TestCaseExpectation(tone=None, text="Expected", schema_id=None, structure=None),
            ),
            TestCase(
                corpus_id=uuid4(),
                inputs=TestCaseInput(prompt="Case 3"),
                expectations=TestCaseExpectation(tone=None, text="Expected", schema_id=None, structure=None),
            ),
        ],
    )


# --- Tests ---


@pytest.mark.asyncio
async def test_run_suite_concurrency(basic_corpus: TestCorpus) -> None:
    """
    Verifies that run_suite executes cases concurrently.
    We have 3 cases. Each takes 0.1s. Total time should be roughly 0.1s, not 0.3s.
    """
    delay = 0.1
    runner = AsyncSleepAgentRunner(delay_s=delay)
    simulator = Simulator(runner)

    start_t = time.perf_counter()
    test_run, results = await simulator.run_suite(basic_corpus, agent_draft_version="0.0.1", run_by="tester")
    end_t = time.perf_counter()

    duration = end_t - start_t

    # Assertions
    assert len(results) == 3
    assert runner.call_count == 3
    assert isinstance(test_run, TestRun)
    assert test_run.status == TestRunStatus.DONE

    # Check concurrency: 3 * 0.1s = 0.3s.
    # With overhead, parallel execution should be well under 0.25s.
    # We use a conservative upper bound (0.28s) to avoid flakes on slow CI,
    # but strictly less than sequential execution.
    assert duration < (delay * 3 * 0.95), f"Execution took {duration}s, expected concurrent execution < {delay * 3}s"


@pytest.mark.asyncio
async def test_run_suite_progress_callback(basic_corpus: TestCorpus) -> None:
    """
    Verifies that the progress callback is called for each completed case.
    """
    runner = AsyncSleepAgentRunner(delay_s=0.01)
    simulator = Simulator(runner)

    callback_calls = []

    async def on_progress(completed: int, total: int, last_result: Any) -> None:
        callback_calls.append((completed, total, last_result.case_id))

    test_run, results = await simulator.run_suite(
        basic_corpus, agent_draft_version="0.0.1", run_by="tester", on_progress=on_progress
    )

    assert len(callback_calls) == 3
    # Check that 'completed' counts increment
    assert callback_calls[0][0] == 1
    assert callback_calls[1][0] == 2
    assert callback_calls[2][0] == 3
    # Check total is always 3
    assert callback_calls[0][1] == 3


@pytest.mark.asyncio
async def test_run_suite_failsafe_execution(basic_corpus: TestCorpus) -> None:
    """
    Verifies that if one case crashes, others complete, and the crashed one is marked as failed.
    """
    # Inject a failing case
    basic_corpus.cases[1].inputs.prompt = "FAIL_ME"

    runner = FlakyAgentRunner(failure_trigger_prompt="FAIL_ME")
    simulator = Simulator(runner)

    test_run, results = await simulator.run_suite(basic_corpus, agent_draft_version="0.0.1", run_by="tester")

    assert len(results) == 3
    assert test_run.status == TestRunStatus.DONE  # The run itself finishes

    # Find the failing result
    failing_result = next(r for r in results if r.case_id == basic_corpus.cases[1].id)
    success_result = next(r for r in results if r.case_id == basic_corpus.cases[0].id)

    assert failing_result.passed is False
    assert failing_result.actual_output.trace is not None
    assert "Intentional Crash" in failing_result.actual_output.trace

    assert success_result.actual_output.text == "OK"


@pytest.mark.asyncio
async def test_run_suite_empty_corpus() -> None:
    """
    Verifies behavior with an empty corpus.
    """
    empty_corpus = TestCorpus(project_id="p", name="n", version="v", created_by="u", cases=[])
    runner = AsyncSleepAgentRunner(0)
    simulator = Simulator(runner)

    test_run, results = await simulator.run_suite(empty_corpus, agent_draft_version="v1", run_by="tester")

    assert len(results) == 0
    assert test_run.status == TestRunStatus.DONE


@pytest.mark.asyncio
async def test_run_suite_on_progress_error(basic_corpus: TestCorpus) -> None:
    """
    Verifies that the run continues if the progress callback raises an exception.
    """
    runner = AsyncSleepAgentRunner(delay_s=0.01)
    simulator = Simulator(runner)

    async def on_progress_raising(completed: int, total: int, last_result: Any) -> None:
        raise RuntimeError("Callback Failed")

    test_run, results = await simulator.run_suite(
        basic_corpus, agent_draft_version="0.0.1", run_by="tester", on_progress=on_progress_raising
    )

    # Should complete all cases despite callback errors
    assert len(results) == 3
    assert test_run.status == TestRunStatus.DONE


@pytest.mark.asyncio
async def test_run_suite_critical_task_failure(basic_corpus: TestCorpus, mocker: Any) -> None:
    """
    Verifies the ultimate fail-safe where run_case itself (or the future) raises an unhandled exception.
    This simulates a failure in the task scheduling or a bug in run_case's own exception handling.
    """
    runner = AsyncSleepAgentRunner(delay_s=0.01)
    simulator = Simulator(runner)

    # Mock run_case to raise an exception *directly*, not returning a failed result.
    # This simulates a crash that bypasses run_case's internal try/except block.
    # We define a coroutine that raises when awaited.
    async def broken_method(*args: Any, **kwargs: Any) -> TestResult:
        raise RuntimeError("Catastrophic Failure")

    mocker.patch.object(simulator, "run_case", side_effect=broken_method)

    test_run, results = await simulator.run_suite(basic_corpus, agent_draft_version="0.0.1", run_by="tester")

    # The fail-safe catches the error and continues.
    # We now expect robust failure results to be generated.
    assert len(results) == 3
    assert test_run.status == TestRunStatus.DONE

    for result in results:
        assert result.passed is False
        assert result.actual_output.trace is not None
        assert "System Error" in result.actual_output.trace
        assert "Catastrophic Failure" in result.actual_output.trace
