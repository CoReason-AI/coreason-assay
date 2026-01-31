# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from uuid import uuid4

import pytest
from coreason_manifest.definitions.simulation import SimulationTrace
from pydantic import ValidationError

from coreason_assay.models import (
    Score,
    TestCase,
    TestCaseExpectation,
    TestCaseInput,
    TestCorpus,
    TestResult,
    TestResultOutput,
    TestRun,
    TestRunStatus,
)


class TestModels:
    def test_test_case_creation(self) -> None:
        corpus_id = uuid4()
        inputs = TestCaseInput(prompt="Hello", files=["s3://bucket/file.pdf"])
        expectations = TestCaseExpectation(
            tone=None, text="World", reasoning=["Step 1"], schema_id=None, structure=None
        )

        test_case = TestCase(corpus_id=corpus_id, inputs=inputs, expectations=expectations)

        assert test_case.corpus_id == corpus_id
        assert test_case.inputs.prompt == "Hello"
        assert test_case.inputs.files == ["s3://bucket/file.pdf"]
        assert test_case.expectations.text == "World"
        assert test_case.expectations.reasoning == ["Step 1"]

    def test_test_corpus_creation(self) -> None:
        corpus = TestCorpus(project_id="proj-123", name="Golden Set", version="1.0", created_by="user@example.com")

        assert corpus.project_id == "proj-123"
        assert corpus.name == "Golden Set"
        assert corpus.version == "1.0"
        assert corpus.created_by == "user@example.com"
        assert corpus.cases == []

    def test_test_run_creation(self) -> None:
        run = TestRun(corpus_version="1.0", agent_draft_version="v2-draft")

        assert run.corpus_version == "1.0"
        assert run.agent_draft_version == "v2-draft"
        assert run.status == TestRunStatus.RUNNING

    def test_test_result_creation(self) -> None:
        run_id = uuid4()
        case_id = uuid4()
        trace = SimulationTrace(
            trace_id=uuid4(),
            agent_version="1.0",
            steps=[],
            outcome={},
            metrics={}
        )
        output = TestResultOutput(text="Response", trace=trace, structured_output=None)
        score = Score(name="accuracy", value=1.0, passed=True, reasoning="Perfect match")

        result = TestResult(run_id=run_id, case_id=case_id, actual_output=output, passed=True, scores=[score])

        assert result.run_id == run_id
        assert result.case_id == case_id
        assert result.actual_output.text == "Response"
        assert result.actual_output.trace == trace
        assert result.passed is True
        assert result.scores[0].name == "accuracy"
        assert result.scores[0].value == 1.0

    def test_test_result_error(self) -> None:
        run_id = uuid4()
        case_id = uuid4()
        output = TestResultOutput(text=None, trace=None, structured_output=None, error="Something went wrong")
        result = TestResult(run_id=run_id, case_id=case_id, actual_output=output, passed=False, scores=[])

        assert result.actual_output.error == "Something went wrong"

    def test_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            TestCaseInput(files="not a list")  # type: ignore

    def test_defaults(self) -> None:
        inputs = TestCaseInput(prompt="Hi")
        assert inputs.files == []
        assert inputs.context == {}

        expectations = TestCaseExpectation(tone=None, text=None, schema_id=None, structure=None)
        assert expectations.reasoning == []
        assert expectations.forbidden_content == []
