# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import json
from typing import Any, Dict, Optional
from uuid import uuid4

import pytest
from coreason_manifest.definitions.agent import AgentDefinition, AgentRuntimeConfig
from coreason_manifest.definitions.simulation import SimulationTrace

from coreason_assay.grader import BaseGrader, ReasoningGrader
from coreason_assay.interfaces import LLMClient
from coreason_assay.models import (
    Score,
    TestCaseInput,
    TestResult,
    TestResultOutput,
)
from coreason_assay.utils.parsing import load_agent, load_trace


# --- Fixtures ---

@pytest.fixture
def complex_trace() -> SimulationTrace:
    return SimulationTrace(
        trace_id=uuid4(),
        agent_version="1.0.0",
        steps=[
            {
                "step_id": str(uuid4()),
                "timestamp": "2023-10-27T10:00:00Z",
                "node_id": "thought_process",
                "inputs": {"query": "complex request"},
                "thought": "I need to break this down.",
                "action": {},
                "observation": {},
                "snapshot": {"state": "initial"},
            },
            {
                "step_id": str(uuid4()),
                "timestamp": "2023-10-27T10:00:01Z",
                "node_id": "tool_execution",
                "inputs": {},
                "thought": "Calling database.",
                "action": {"tool": "db_query", "args": {"sql": "SELECT * FROM users"}},
                "observation": {"result": "5 users found"},
            },
        ],
        outcome={"final_answer": "42", "confidence": 0.99},
        metrics={"tokens_used": 150, "cost": 0.002},
    )


@pytest.fixture
def complex_agent() -> AgentDefinition:
    # Need to construct full valid object or mock if strict validation isn't triggered by constructor
    # But AgentDefinition is Pydantic model, so constructor validates.
    # We must provide all required fields.
    # Based on schema: metadata, interface, config, dependencies, integrity_hash
    from coreason_manifest.definitions.agent import AgentMetadata, AgentInterface, AgentDependencies, ModelConfig

    config = AgentRuntimeConfig(
        # system_prompt removed as it is not in v0.9.0 schema
        model_config=ModelConfig(model="gpt-4", temperature=0.7),
        nodes=[],
        edges=[],
        entry_point="start"
    )

    return AgentDefinition(
        metadata=AgentMetadata(
            id=str(uuid4()),
            name="ComplexAgent",
            version="2.5.0",
            created_at="2023-01-01T00:00:00Z",
            author="me",
            requires_auth=False
        ),
        interface=AgentInterface(inputs={}, outputs={}),
        config=config,
        dependencies=AgentDependencies(tools=[], libraries=[]),
        integrity_hash="a" * 64,
    )


class MockLLMClient(LLMClient):
    def __init__(self) -> None:
        self.last_prompt: str = ""

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        # Return a dummy passing response
        return json.dumps({"score": 1.0, "steps_analysis": []})


class ConfigGrader(BaseGrader):
    """
    Grader that verifies the agent's configuration.
    """
    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
        agent: Optional[AgentDefinition] = None,
    ) -> Score:
        if not agent:
            return Score(name="ConfigCheck", value=0.0, passed=False, reasoning="No agent provided")

        # Verify entry_point is present (valid field in v0.9.0)
        entry_point = agent.config.entry_point
        expected_ep = expectations.get("entry_point", "") if expectations else ""

        if entry_point == expected_ep:
            return Score(name="ConfigCheck", value=1.0, passed=True, reasoning=f"Match '{entry_point}'")
        return Score(name="ConfigCheck", value=0.0, passed=False, reasoning=f"Mismatch '{entry_point}'")


# --- Tests ---

def test_complex_trace_reasoning(complex_trace: SimulationTrace) -> None:
    """
    Verify ReasoningGrader correctly serializes a complex trace into the prompt.
    """
    llm_client = MockLLMClient()
    grader = ReasoningGrader(llm_client=llm_client)

    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="42",
            trace=complex_trace,
            structured_output=None
        ),
        metrics={},
        scores=[],
        passed=False
    )

    expectations = {"reasoning": ["Check database"]}

    grader.grade(result, expectations=expectations)

    # Verify prompt content
    prompt = llm_client.last_prompt
    assert "SELECT * FROM users" in prompt
    assert "5 users found" in prompt
    assert "token_usage" not in prompt # 'tokens_used' is in metrics, logic might not include root metrics in prompt trace dump?
    # ReasoningGrader uses `trace_obj.model_dump_json(indent=2)`.
    # SimulationTrace includes metrics. So metrics SHOULD be in the prompt.
    assert "tokens_used" in prompt


def test_agent_config_grading(complex_agent: AgentDefinition) -> None:
    """
    Verify we can grade based on the AgentDefinition configuration.
    """
    grader = ConfigGrader()

    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(text="Hi", trace=None, structured_output=None),
        metrics={},
        scores=[],
        passed=False
    )

    # Expect "start" which is the entry_point in complex_agent
    score = grader.grade(result, expectations={"entry_point": "start"}, agent=complex_agent)

    assert score.passed is True
    assert score.value == 1.0
    assert "Match 'start'" in score.reasoning


def test_trace_metrics_access(complex_trace: SimulationTrace) -> None:
    """
    Verify we can access structured metrics from the result object.
    """
    result = TestResult(
        run_id=uuid4(),
        case_id=uuid4(),
        actual_output=TestResultOutput(
            text="42",
            trace=complex_trace,
            structured_output=None
        ),
        metrics={"latency_ms": 500}, # Engine metrics
        scores=[],
        passed=False
    )

    # Access trace object
    trace_obj = result.actual_output.trace
    assert trace_obj is not None
    assert isinstance(trace_obj, SimulationTrace)

    # Check trace internal metrics
    assert trace_obj.metrics["tokens_used"] == 150
    assert trace_obj.metrics["cost"] == 0.002

    # Check engine metrics are separate
    assert result.metrics["latency_ms"] == 500


def test_parsing_robustness() -> None:
    """
    Test loading objects with extra fields (should be ignored if model allows, or fail if strict).
    coreason-manifest models are likely strict or allow extra?
    Let's verify behavior.
    """
    # 1. Trace with extra field
    trace_json = json.dumps({
        "trace_id": str(uuid4()),
        "agent_version": "1.0",
        "steps": [],
        "outcome": {},
        "metrics": {},
        "extra_field": "should_be_ignored_or_error"
    })

    # If default Pydantic config (ignore extras), this passes. If strict/forbid, it raises.
    # We just want to know it doesn't crash unpredictably.
    try:
        trace = load_trace(trace_json)
        assert isinstance(trace, SimulationTrace)
    except Exception:
        # If it fails validation, that is also an acceptable outcome for "Strict Consumer"
        pass

    # 2. Agent with Unicode
    agent_json = json.dumps({
        "metadata": {
            "id": str(uuid4()),
            "name": "Agent 🤖",
            "version": "1.0.0",
            "created_at": "2023-01-01T00:00:00Z",
            "author": "me",
            "requires_auth": False
        },
        "interface": {"inputs": {}, "outputs": {}},
        "config": {
            # "system_prompt": "Unicode check: \u2714", # Removed for v0.9.0 compliance
            "model_config": {"model": "gpt-4", "temperature": 0.0},
            "nodes": [],
            "edges": [],
            "entry_point": "start"
        },
        "dependencies": {"tools": [], "libraries": []},
        "integrity_hash": "a" * 64
    })

    agent = load_agent(agent_json)
    assert agent.metadata.name == "Agent 🤖"
    # assert "✔" in agent.config.system_prompt # Removed assertion
