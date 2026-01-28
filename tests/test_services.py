# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Any, Dict, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from coreason_assay.grader import BaseGrader
from coreason_assay.interfaces import AgentRunner
from coreason_assay.models import (
    ReportCard,
    Score,
    TestCase,
    TestCaseExpectation,
    TestCaseInput,
    TestCorpus,
    TestResult,
)
from coreason_assay.services import run_suite, upload_bec
from coreason_identity.models import UserContext
from pytest_mock import MockerFixture


class MockGrader(BaseGrader):
    def grade(
        self,
        result: TestResult,
        inputs: Optional[TestCaseInput] = None,
        expectations: Optional[Dict[str, Any]] = None,
    ) -> Score:
        return cast(Score, MagicMock())


@pytest.fixture
def mock_bec_manager(mocker: MockerFixture) -> MagicMock:
    return cast(MagicMock, mocker.patch("coreason_assay.services.BECManager"))


@pytest.fixture
def mock_simulator(mocker: MockerFixture) -> MagicMock:
    return cast(MagicMock, mocker.patch("coreason_assay.services.Simulator"))


@pytest.fixture
def mock_engine(mocker: MockerFixture) -> MagicMock:
    return cast(MagicMock, mocker.patch("coreason_assay.services.AssessmentEngine"))


def test_upload_bec(mock_bec_manager: MagicMock, tmp_path: Any) -> None:
    # Setup
    mock_cases = [
        TestCase(
            corpus_id=uuid4(),
            inputs=TestCaseInput(prompt="p"),
            expectations=TestCaseExpectation(
                text="text",
                schema_id="schema",
                structure={},
                reasoning=[],
                forbidden_content=[],
                tool_mocks={},
                tone="",
            ),
        )
    ]
    mock_bec_manager.load_from_zip.return_value = mock_cases

    zip_path = tmp_path / "test.zip"
    zip_path.touch()
    extract_dir = tmp_path / "extract"

    # Mock UserContext
    # We use a dummy object if UserContext is not easily mockable, or specific mocking
    mock_context = MagicMock(spec=UserContext)
    mock_context.user_id = "user-1"

    # Execute
    corpus = upload_bec(
        file_path=zip_path,
        extraction_dir=extract_dir,
        project_id="proj-123",
        name="Test Corpus",
        version="v1.0",
        user_context=mock_context,
    )

    # Verify
    mock_bec_manager.load_from_zip.assert_called_once_with(zip_path, extract_dir)
    assert isinstance(corpus, TestCorpus)
    assert corpus.project_id == "proj-123"
    assert corpus.name == "Test Corpus"
    assert len(corpus.cases) == 1
    # Check if corpus_id was unified
    assert corpus.cases[0].corpus_id == corpus.id
    assert corpus.created_by == "user-1"


@pytest.mark.asyncio
async def test_run_suite(mock_simulator: MagicMock, mock_engine: MagicMock) -> None:
    # Setup
    corpus = TestCorpus(project_id="p", name="n", version="v", created_by="u", cases=[])
    mock_runner = AsyncMock(spec=AgentRunner)
    mock_graders: List[BaseGrader] = [MockGrader()]

    expected_report = ReportCard(
        run_id=uuid4(), total_cases=10, passed_cases=9, failed_cases=1, pass_rate=0.9, aggregates=[]
    )

    # Configure mock engine instance
    mock_engine_instance = mock_engine.return_value
    mock_engine_instance.run_assay = AsyncMock(return_value=expected_report)

    # Configure mock simulator instance (though not strictly accessed in run_suite logic before passed to engine)
    mock_sim_instance = mock_simulator.return_value

    # Execute
    report = await run_suite(
        corpus=corpus, agent_runner=mock_runner, agent_draft_version="draft-v1", graders=mock_graders
    )

    # Verify initialization
    mock_simulator.assert_called_once_with(runner=mock_runner)
    mock_engine.assert_called_once_with(simulator=mock_sim_instance, graders=mock_graders)

    # Verify execution
    mock_engine_instance.run_assay.assert_awaited_once_with(
        corpus=corpus, agent_draft_version="draft-v1", on_progress=None
    )

    assert report == expected_report


@pytest.mark.asyncio
async def test_run_suite_with_progress(mock_simulator: MagicMock, mock_engine: MagicMock) -> None:
    # Setup
    corpus = TestCorpus(project_id="p", name="n", version="v", created_by="u", cases=[])
    mock_runner = AsyncMock(spec=AgentRunner)
    mock_graders: List[BaseGrader] = []

    async def progress_cb(c: int, t: int, r: TestResult) -> None:
        pass

    mock_engine_instance = mock_engine.return_value
    mock_engine_instance.run_assay = AsyncMock(
        return_value=ReportCard(run_id=uuid4(), total_cases=0, passed_cases=0, failed_cases=0, pass_rate=0.0)
    )

    # Execute
    await run_suite(
        corpus=corpus, agent_runner=mock_runner, agent_draft_version="v1", graders=mock_graders, on_progress=progress_cb
    )

    # Verify
    mock_engine_instance.run_assay.assert_awaited_once_with(
        corpus=corpus, agent_draft_version="v1", on_progress=progress_cb
    )
