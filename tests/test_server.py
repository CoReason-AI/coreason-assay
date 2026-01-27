# Copyright (c) 2025 CoReason, Inc.

from typing import Any, Generator, Tuple, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from coreason_assay.interfaces import AgentRunner, LLMClient
from coreason_assay.models import ReportCard, TestCorpus
from coreason_assay.server import app, set_dependencies

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.fixture
def mock_upload_bec(mocker: Any) -> MagicMock:
    return cast(MagicMock, mocker.patch("coreason_assay.server.upload_bec"))


def test_upload_corpus(mock_upload_bec: MagicMock, mocker: Any) -> None:
    # Setup mock return
    corpus = TestCorpus(project_id="p1", name="Test Corpus", version="1.0", created_by="tester", cases=[])
    mock_upload_bec.return_value = corpus

    mock_unlink = mocker.patch("pathlib.Path.unlink")

    # Create dummy zip content
    file_content = b"PK\x03\x04dummyzipcontent"

    response = client.post(
        "/upload",
        files={"file": ("test.zip", file_content, "application/zip")},
        data={"project_id": "p1", "name": "Test Corpus", "version": "1.0", "author": "tester"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(corpus.id)
    mock_upload_bec.assert_called_once()
    # Verify zip unlink called
    mock_unlink.assert_called()


def test_upload_corpus_unlink_failure(mock_upload_bec: MagicMock, mocker: Any) -> None:
    mock_unlink = mocker.patch("pathlib.Path.unlink", side_effect=OSError("Cannot delete"))

    corpus = TestCorpus(project_id="p1", name="Test Corpus", version="1.0", created_by="tester", cases=[])
    mock_upload_bec.return_value = corpus

    file_content = b"dummy"
    response = client.post(
        "/upload",
        files={"file": ("test.zip", file_content, "application/zip")},
        data={"project_id": "p1", "name": "Test Corpus", "version": "1.0", "author": "tester"},
    )

    assert response.status_code == 200
    mock_unlink.assert_called()
    # Should swallow error and return success


def test_upload_corpus_failure(mock_upload_bec: MagicMock, mocker: Any) -> None:
    mock_upload_bec.side_effect = Exception("Upload failed")

    mock_rmtree = mocker.patch("shutil.rmtree")

    file_content = b"dummy"
    response = client.post(
        "/upload",
        files={"file": ("test.zip", file_content, "application/zip")},
        data={"project_id": "p1", "name": "Test Corpus", "version": "1.0", "author": "tester"},
    )

    assert response.status_code == 400
    assert "Failed to process upload" in response.json()["detail"]
    mock_rmtree.assert_called_once()


@pytest.fixture
def mock_dependencies() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    runner = MagicMock(spec=AgentRunner)
    llm = MagicMock(spec=LLMClient)
    set_dependencies(runner, llm)
    yield runner, llm
    set_dependencies(None, None)  # reset


def test_run_assay_missing_dependencies() -> None:
    # Ensure no dependencies set
    set_dependencies(None, None)

    corpus = TestCorpus(project_id="p1", name="Test Corpus", version="1.0", created_by="tester", cases=[])

    response = client.post(
        "/run",
        json={"corpus": corpus.model_dump(mode="json"), "agent_version": "1.0", "graders": {"Latency": {}}},
    )

    assert response.status_code == 503
    assert "AgentRunner not initialized" in response.json()["detail"]


def test_run_assay_missing_llm_client(mock_dependencies: Tuple[MagicMock, MagicMock]) -> None:
    runner, _ = mock_dependencies
    set_dependencies(runner, None)  # No LLM client

    corpus = TestCorpus(project_id="p1", name="Test Corpus", version="1.0", created_by="tester", cases=[])

    # Reasoning needs LLMClient
    response = client.post(
        "/run",
        json={"corpus": corpus.model_dump(mode="json"), "agent_version": "1.0", "graders": {"Reasoning": {}}},
    )

    assert response.status_code == 503
    assert "LLMClient not initialized" in response.json()["detail"]

    # Faithfulness needs LLMClient
    response = client.post(
        "/run",
        json={"corpus": corpus.model_dump(mode="json"), "agent_version": "1.0", "graders": {"Faithfulness": {}}},
    )
    assert response.status_code == 503

    # Tone needs LLMClient
    response = client.post(
        "/run",
        json={"corpus": corpus.model_dump(mode="json"), "agent_version": "1.0", "graders": {"Tone": {}}},
    )
    assert response.status_code == 503


def test_run_assay_all_graders(mock_dependencies: Tuple[MagicMock, MagicMock], mocker: Any) -> None:
    runner, llm = mock_dependencies

    mock_run = mocker.patch("coreason_assay.engine.AssessmentEngine.run_assay", new_callable=mocker.AsyncMock)

    report = ReportCard(run_id=uuid4(), total_cases=1, passed_cases=1, failed_cases=0, pass_rate=1.0, aggregates=[])
    mock_run.return_value = report

    corpus = TestCorpus(project_id="p1", name="Test Corpus", version="1.0", created_by="tester", cases=[])

    response = client.post(
        "/run",
        json={
            "corpus": corpus.model_dump(mode="json"),
            "agent_version": "1.0",
            "graders": {
                "Latency": {"threshold_ms": 1000},
                "Faithfulness": {},
                "JsonSchema": {},
                "ForbiddenContent": {},
                "Reasoning": {},
                "Tone": {},
                "UnknownGrader": {},
            },
        },
    )

    assert response.status_code == 200
    mock_run.assert_called_once()
