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
    TestResult,
    TestResultOutput,
)
from coreason_assay.simulator import Simulator


class MockAgentRunner(AgentRunner):
    def __init__(self, return_text: str = "Test Response") -> None:
        self.return_text = return_text
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
        return TestResultOutput(text=self.return_text, trace=None, structured_output={"key": "value"})


class RaisingAgentRunner(AgentRunner):
    async def invoke(
        self, inputs: TestCaseInput, user_context: UserContext, tool_mocks: Dict[str, Any]
    ) -> TestResultOutput:
        raise RuntimeError("Simulated agent failure")


@pytest.fixture
def sample_test_case() -> TestCase:
    return TestCase(
        corpus_id=uuid4(),
        inputs=TestCaseInput(prompt="Hello", context={"user_id": "test_user", "email": "test_user@coreason.ai"}),
        expectations=TestCaseExpectation(
            tone=None, text="Hello back", schema_id=None, structure=None, tool_mocks={"db": "error"}
        ),
    )


def test_simulator_init() -> None:
    runner = MockAgentRunner()
    simulator = Simulator(runner)
    assert simulator.runner == runner


def test_simulator_run_case_success(sample_test_case: TestCase) -> None:
    runner = MockAgentRunner(return_text="Success Output")
    simulator = Simulator(runner)
    run_id = uuid4()

    # Use asyncio.run to execute the async method in the synchronous test
    result = asyncio.run(simulator.run_case(sample_test_case, run_id))

    # Verification of Runner Interaction
    assert runner.invoked is True
    assert runner.last_inputs == sample_test_case.inputs
    # Check that context was hydrated
    assert runner.last_context is not None
    assert runner.last_context.user_id == "test_user"
    assert runner.last_tool_mocks == sample_test_case.expectations.tool_mocks

    # Verification of Result Object
    assert isinstance(result, TestResult)
    assert result.run_id == run_id
    assert result.case_id == sample_test_case.id
    assert result.actual_output.text == "Success Output"
    assert result.actual_output.trace is None
    assert result.actual_output.structured_output == {"key": "value"}

    # Latency check (should be in metrics now)
    assert "latency_ms" in result.metrics
    assert result.metrics["latency_ms"] >= 0
    # Scores should be empty
    assert result.scores == []


def test_simulator_run_case_exception(sample_test_case: TestCase) -> None:
    runner = RaisingAgentRunner()
    simulator = Simulator(runner)
    run_id = uuid4()

    result = asyncio.run(simulator.run_case(sample_test_case, run_id))

    # Verification of Error Handling
    assert result.run_id == run_id
    assert result.case_id == sample_test_case.id
    assert result.actual_output.text is None
    assert result.actual_output.trace is None
    assert result.actual_output.error is not None
    assert "Agent invocation failed" in result.actual_output.error
    assert "Simulated agent failure" in result.actual_output.error

    # Latency should still be captured
    assert "latency_ms" in result.metrics
