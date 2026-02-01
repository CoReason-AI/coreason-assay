# Copyright (c) 2025 CoReason, Inc.

from typing import Any, Dict
from uuid import uuid4

import pytest
from coreason_identity.models import UserContext

from coreason_assay.interfaces import AgentRunner
from coreason_assay.models import TestCase, TestCaseExpectation, TestCaseInput, TestResultOutput
from coreason_assay.simulator import Simulator


class MockRunner(AgentRunner):
    async def invoke(
        self,
        inputs: TestCaseInput,
        user_context: UserContext,
        tool_mocks: Dict[str, Any],
    ) -> TestResultOutput:
        return TestResultOutput(error=None, text="OK", trace=None, structured_output=None)


@pytest.mark.asyncio
async def test_identity_hydration_failure() -> None:
    runner = MockRunner()
    simulator = Simulator(runner)

    # Create a case with invalid context (missing user_id)
    # This should trigger Pydantic ValidationError during UserContext.model_validate
    case = TestCase(
        corpus_id=uuid4(),
        inputs=TestCaseInput(prompt="p", context={}),
        expectations=TestCaseExpectation(
            text="e",
            schema_id=None,
            structure=None,
            reasoning=[],
            forbidden_content=[],
            tool_mocks={},
            tone=None,
        ),
    )

    result = await simulator.run_case(case, uuid4())

    assert result.passed is False
    assert result.actual_output.error is not None
    assert "Identity Hydration Failed" in result.actual_output.error
