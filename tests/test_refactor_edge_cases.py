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
import json
from typing import Any, Dict
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
from coreason_assay.utils.parsing import parse_json_from_llm_response

# --- Tests for Parsing ---


def test_parse_json_valid() -> None:
    text = '{"foo": "bar"}'
    assert parse_json_from_llm_response(text) == {"foo": "bar"}


def test_parse_json_markdown_block() -> None:
    text = '```json\n{"foo": "bar"}\n```'
    assert parse_json_from_llm_response(text) == {"foo": "bar"}


def test_parse_json_markdown_block_no_lang() -> None:
    text = '```\n{"foo": "bar"}\n```'
    assert parse_json_from_llm_response(text) == {"foo": "bar"}


def test_parse_json_surrounding_text_handled_crudely() -> None:
    # Our current parser is simple: it strips markdown tags if they start/end the string.
    # It does NOT extract JSON from the middle of text if not wrapped in blocks cleanly or if there is extra text.
    # But let's verify exact behavior.
    # If the user provides "Sure! ```json {} ```", startswith("```") will be False.
    # So it will try to parse "Sure! ...", which fails.
    # This is "Known Limitation" / "Standard Behavior" for the current implementation.
    text = 'Sure! {"foo": "bar"}'
    with pytest.raises(json.JSONDecodeError):
        parse_json_from_llm_response(text)


def test_parse_json_garbage() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_from_llm_response("Not JSON")


def test_parse_json_empty() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_from_llm_response("")


# --- Tests for Complex Concurrency ---


class MixedBehaviorAgent(AgentRunner):
    def __init__(self) -> None:
        pass

    async def invoke(
        self, inputs: TestCaseInput, context: Dict[str, Any], tool_mocks: Dict[str, Any]
    ) -> TestResultOutput:
        mode = inputs.prompt
        if mode == "FAST_OK":
            return TestResultOutput(text="OK", trace="Fast", structured_output=None)
        elif mode == "SLOW_OK":
            await asyncio.sleep(0.2)
            return TestResultOutput(text="OK", trace="Slow", structured_output=None)
        elif mode == "FAST_FAIL":
            raise RuntimeError("Fast Crash")
        elif mode == "SLOW_FAIL":
            await asyncio.sleep(0.2)
            raise RuntimeError("Slow Crash")
        else:
            return TestResultOutput(text="Unknown", trace="?", structured_output=None)


@pytest.mark.asyncio
async def test_simulator_mixed_workload_robustness() -> None:
    """
    Tests that the simulator handles a mix of passing, failing, and slow agents
    without losing results or crashing the run.
    """
    runner = MixedBehaviorAgent()
    simulator = Simulator(runner)

    # Create 10 cases with mixed types
    # 3 Fast OK
    # 3 Slow OK
    # 2 Fast Fail
    # 2 Slow Fail
    prompts = ["FAST_OK"] * 3 + ["SLOW_OK"] * 3 + ["FAST_FAIL"] * 2 + ["SLOW_FAIL"] * 2

    cases = [
        TestCase(
            corpus_id=uuid4(),
            inputs=TestCaseInput(prompt=p),
            expectations=TestCaseExpectation(text=None, schema_id=None, structure=None, tone=None),
        )
        for p in prompts
    ]

    corpus = TestCorpus(
        project_id="test",
        name="Mixed Corpus",
        version="1",
        created_by="me",
        cases=cases,
    )

    test_run, results = await simulator.run_suite(corpus, agent_draft_version="0.1")

    # Verify Run Status
    assert test_run.status == TestRunStatus.DONE
    assert len(results) == 10

    # Verify Results Breakdown
    fast_ok_count = sum(1 for r in results if r.actual_output.trace == "Fast")
    slow_ok_count = sum(1 for r in results if r.actual_output.trace == "Slow")

    # Failures handled by run_case have "Agent invocation failed"
    # Failures handled by _run_and_track (fail-safe) have "System Error"
    failures = [r for r in results if r.passed is False]

    # In this test, MixedBehaviorAgent raises inside invoke, so run_case catches it.
    agent_failures = [
        r for r in failures if r.actual_output.trace and "Agent invocation failed" in r.actual_output.trace
    ]

    assert fast_ok_count == 3
    assert slow_ok_count == 3
    assert len(agent_failures) == 4  # 2 Fast Fail + 2 Slow Fail

    # Verify error messages
    error_texts = [r.actual_output.trace for r in agent_failures]
    assert any("Fast Crash" in t for t in error_texts if t)
    assert any("Slow Crash" in t for t in error_texts if t)
