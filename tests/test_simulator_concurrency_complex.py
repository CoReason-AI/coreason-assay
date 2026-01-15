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
from typing import Any, Dict, List
from uuid import uuid4

import pytest

from coreason_assay.interfaces import AgentRunner
from coreason_assay.models import (
    TestCase,
    TestCaseExpectation,
    TestCaseInput,
    TestCorpus,
    TestResultOutput,
    TestRunStatus,
)
from coreason_assay.simulator import Simulator


class SpyAgentRunner(AgentRunner):
    """
    Captures what was passed to invoke() and simulates various behaviors.
    """

    def __init__(self) -> None:
        self.invocations: List[Dict[str, Any]] = []
        # Lock not strictly needed for append in asyncio (atomic-ish), but good practice if logic grows
        self._lock = asyncio.Lock()

    async def invoke(
        self, inputs: TestCaseInput, context: Dict[str, Any], tool_mocks: Dict[str, Any]
    ) -> TestResultOutput:
        # Capture the inputs
        async with self._lock:
            self.invocations.append(
                {
                    "prompt": inputs.prompt,
                    "context": context,
                    "tool_mocks": tool_mocks,
                }
            )

        # Simulate behavior based on prompt content
        if "CRASH" in inputs.prompt:
            raise RuntimeError(f"Crashed on {inputs.prompt}")
        elif "SLOW" in inputs.prompt:
            await asyncio.sleep(0.05)
            return TestResultOutput(
                text=f"Slow response to {inputs.prompt}", trace="Slow Trace", structured_output=None
            )
        else:
            return TestResultOutput(
                text=f"Fast response to {inputs.prompt}", trace="Fast Trace", structured_output=None
            )


@pytest.mark.asyncio
async def test_run_suite_mixed_workload() -> None:
    """
    Tests a suite with a mix of fast, slow, and crashing agents.
    Verifies that all results are collected and mapped correctly.
    """
    case_count = 50
    cases = []

    # Generate 50 cases with mixed types
    for i in range(case_count):
        if i % 5 == 0:
            prompt = f"CRASH_{i}"
        elif i % 2 == 0:
            prompt = f"SLOW_{i}"
        else:
            prompt = f"FAST_{i}"

        cases.append(
            TestCase(
                corpus_id=uuid4(),
                inputs=TestCaseInput(prompt=prompt),
                expectations=TestCaseExpectation(tone=None, text="Expected", schema_id=None, structure=None),
            )
        )

    corpus = TestCorpus(project_id="p1", name="Mixed Corpus", version="v1", created_by="tester", cases=cases)

    runner = SpyAgentRunner()
    simulator = Simulator(runner)

    test_run, results = await simulator.run_suite(corpus, agent_draft_version="v1")

    assert test_run.status == TestRunStatus.DONE
    assert len(results) == case_count

    # Verification
    failed_count = 0
    slow_count = 0
    fast_count = 0

    for res in results:
        # We need to match result back to input prompt to verify behavior.
        # Since we didn't store prompt in result (it's in the case), we can look up by case_id.
        original_case = next(c for c in cases if c.id == res.case_id)
        prompt = original_case.inputs.prompt

        if "CRASH" in prompt:
            assert res.passed is False
            assert res.actual_output.text is None
            assert res.actual_output.trace is not None and "Crashed on" in res.actual_output.trace
            failed_count += 1
        elif "SLOW" in prompt:
            assert res.actual_output.text == f"Slow response to {prompt}"
            slow_count += 1
        elif "FAST" in prompt:
            assert res.actual_output.text == f"Fast response to {prompt}"
            fast_count += 1

    # Check counts
    # 0, 5, 10, ... 45 -> 10 crashes
    assert failed_count == 10
    # Remaining 40: evens are slow (minus crashes), odds are fast
    # Evens: 0, 2, 4... 48 (25 total). crash indices (0, 10, 20, 30, 40) are evens.
    # So 25 evens - 5 crashing evens = 20 slow.
    # Odds: 1, 3, 5... 49 (25 total). crash indices (5, 15, 25, 35, 45) are odds.
    # So 25 odds - 5 crashing odds = 20 fast.
    assert slow_count == 20
    assert fast_count == 20


@pytest.mark.asyncio
async def test_run_suite_context_isolation() -> None:
    """
    Verifies that running cases concurrently does not mix up their contexts.
    Each case gets a unique context ID, and we verify the runner received that exact ID.
    """
    case_count = 20
    cases = []

    for i in range(case_count):
        cases.append(
            TestCase(
                corpus_id=uuid4(),
                inputs=TestCaseInput(prompt=f"Case_{i}", context={"isolation_id": i, "common": "data"}),
                expectations=TestCaseExpectation(
                    tone=None, text="Expected", schema_id=None, structure=None, tool_mocks={"mock_id": i}
                ),
            )
        )

    corpus = TestCorpus(project_id="p1", name="Isolation Corpus", version="v1", created_by="tester", cases=cases)

    runner = SpyAgentRunner()
    simulator = Simulator(runner)

    # Randomize order in which they might finish by having random delays in the runner?
    # SpyAgentRunner defaults to fast, but context check is robust regardless of order.

    await simulator.run_suite(corpus, agent_draft_version="v1")

    assert len(runner.invocations) == case_count

    received_contexts = sorted([inv["context"]["isolation_id"] for inv in runner.invocations])
    received_mocks = sorted([inv["tool_mocks"]["mock_id"] for inv in runner.invocations])

    expected_ids = list(range(case_count))

    assert received_contexts == expected_ids
    assert received_mocks == expected_ids

    # Verify exact mapping (prompt X got context X)
    for inv in runner.invocations:
        # prompt is "Case_{i}"
        i = int(inv["prompt"].split("_")[1])
        assert inv["context"]["isolation_id"] == i
        assert inv["tool_mocks"]["mock_id"] == i
