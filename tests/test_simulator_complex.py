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
from typing import Any, Dict, Optional
from uuid import uuid4

import pytest

from coreason_identity.models import UserContext

from coreason_assay.interfaces import AgentRunner
from coreason_assay.models import (
    TestCase,
    TestCaseExpectation,
    TestCaseInput,
    TestResultOutput,
)
from coreason_assay.simulator import Simulator


class MockAgentRunner(AgentRunner):
    def __init__(self, return_text: Optional[str] = "Test Response", delay: float = 0.0) -> None:
        self.return_text = return_text
        self.delay = delay
        self.invoked = False
        self.last_inputs: Optional[TestCaseInput] = None
        self.last_context: Optional[UserContext] = None
        self.last_tool_mocks: Optional[Dict[str, Any]] = None

    async def invoke(
        self, inputs: TestCaseInput, user_context: UserContext, tool_mocks: Dict[str, Any]
    ) -> TestResultOutput:
        self.invoked = True
        self.last_inputs = inputs
        self.last_context = user_context
        self.last_tool_mocks = tool_mocks

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        return TestResultOutput(
            text=self.return_text,
            trace="Mock trace",
            structured_output={"complex": {"nested": [1, 2, 3]}} if self.return_text else None,
        )


@pytest.fixture
def base_test_case() -> TestCase:
    return TestCase(
        corpus_id=uuid4(),
        inputs=TestCaseInput(prompt="Base Prompt", context={"user_id": "base_user", "email": "base_user@coreason.ai"}),
        expectations=TestCaseExpectation(tone=None, text="Base Expectation", schema_id=None, structure=None),
    )


def test_simulator_empty_inputs(base_test_case: TestCase) -> None:
    """Test handling of empty strings and minimal valid context."""
    base_test_case.inputs.prompt = ""
    base_test_case.inputs.context = {"user_id": "test", "email": "test@coreason.ai"} # Minimal valid context
    base_test_case.expectations.tool_mocks = {}

    runner = MockAgentRunner(return_text="Response")
    simulator = Simulator(runner)
    run_id = uuid4()

    result = asyncio.run(simulator.run_case(base_test_case, run_id))

    assert result.passed is False  # default
    assert runner.last_inputs is not None
    assert runner.last_inputs.prompt == ""
    assert runner.last_context.user_id == "test"
    assert runner.last_context.email == "test@coreason.ai"
    assert runner.last_tool_mocks == {}
    assert result.actual_output.text == "Response"


def test_simulator_complex_structured_output(base_test_case: TestCase) -> None:
    """Test that complex nested structured output is preserved."""
    runner = MockAgentRunner(return_text="Complex")
    simulator = Simulator(runner)
    run_id = uuid4()

    result = asyncio.run(simulator.run_case(base_test_case, run_id))

    assert result.actual_output.structured_output == {"complex": {"nested": [1, 2, 3]}}


def test_simulator_tool_mocks_propagation(base_test_case: TestCase) -> None:
    """Test that tool mocks are correctly extracted from expectations and passed to runner."""
    mocks = {"database": {"error": "503 Service Unavailable"}, "search": {"results": ["A", "B"]}}
    base_test_case.expectations.tool_mocks = mocks

    runner = MockAgentRunner()
    simulator = Simulator(runner)
    run_id = uuid4()

    asyncio.run(simulator.run_case(base_test_case, run_id))

    assert runner.last_tool_mocks == mocks


def test_simulator_slow_execution(base_test_case: TestCase) -> None:
    """Test latency calculation for a slow agent."""
    delay = 0.1  # 100ms
    runner = MockAgentRunner(delay=delay)
    simulator = Simulator(runner)
    run_id = uuid4()

    result = asyncio.run(simulator.run_case(base_test_case, run_id))

    # Latency should be approximately the delay.
    # On Windows or busy CI, asyncio.sleep can be slightly imprecise.
    # We allow a small tolerance (e.g., 90% of delay is acceptable as 'slow enough' proof if clock skews,
    # but strictly it should be >= delay. However, asyncio.sleep isn't guaranteed to sleep *exactly* that long
    # or perf_counter might have slight variance relative to sleep?)
    # Actually, asyncio.sleep guarantees *at least* delay seconds.
    # But measured latency is wall clock.
    # If 99.936ms < 100ms, it means sleep returned early? Or float precision issues.
    # 99.936 is extremely close to 100.

    assert "latency_ms" in result.metrics
    # Allow 5ms tolerance for under-sleep/precision issues
    assert result.metrics["latency_ms"] >= (delay * 1000) - 5
    # Allow some overhead buffer (e.g., should be less than delay + 500ms)
    assert result.metrics["latency_ms"] < (delay * 1000) + 500


def test_simulator_very_fast_execution(base_test_case: TestCase) -> None:
    """Test latency calculation for instant execution."""
    runner = MockAgentRunner(delay=0)
    simulator = Simulator(runner)
    run_id = uuid4()

    result = asyncio.run(simulator.run_case(base_test_case, run_id))

    assert "latency_ms" in result.metrics
    assert result.metrics["latency_ms"] >= 0


def test_simulator_runner_returns_none_fields(base_test_case: TestCase) -> None:
    """Test handling when runner returns None for optional fields."""

    class NoneAgentRunner(AgentRunner):
        async def invoke(
            self, inputs: TestCaseInput, user_context: UserContext, tool_mocks: Dict[str, Any]
        ) -> TestResultOutput:
            return TestResultOutput(text=None, trace=None, structured_output=None)

    runner = NoneAgentRunner()
    simulator = Simulator(runner)
    run_id = uuid4()

    result = asyncio.run(simulator.run_case(base_test_case, run_id))

    assert result.actual_output.text is None
    assert result.actual_output.trace is None
    assert result.actual_output.structured_output is None
