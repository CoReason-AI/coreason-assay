# Copyright (c) 2025 CoReason, Inc.

import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import cast

import pytest
from fastapi.testclient import TestClient

from coreason_assay.server import app, set_dependencies
from coreason_assay.models import TestCorpus, ReportCard, AggregateMetric
from coreason_assay.interfaces import AgentRunner, LLMClient

client = TestClient(app)

@pytest.fixture
def mock_upload_bec(mocker):
    # Mock services.upload_bec
    # Use cast to satisfy mypy strict checks on mocks
    return cast(MagicMock, mocker.patch("coreason_assay.server.upload_bec"))

@pytest.fixture
def mock_run_suite(mocker):
    # Mock services.run_suite
    # run_suite is async, so we need an async mock
    mock = AsyncMock()
    mocker.patch("coreason_assay.server.run_suite", side_effect=mock)
    return mock

@pytest.fixture
def mock_agent_runner():
    return MagicMock(spec=AgentRunner)

@pytest.fixture
def mock_llm_client():
    return MagicMock(spec=LLMClient)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "coreason-assay", "version": "0.2.0"}

def test_upload_corpus(mock_upload_bec):
    # Mock return value
    mock_corpus = TestCorpus(
        project_id="p1",
        name="test",
        version="v1",
        created_by="me",
        cases=[]
    )
    mock_upload_bec.return_value = mock_corpus

    # Prepare file upload
    files = {"file": ("corpus.zip", b"fake zip content", "application/zip")}
    data = {
        "project_id": "p1",
        "name": "test",
        "version": "v1",
        "author": "me"
    }

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 200
    assert response.json()["id"] == str(mock_corpus.id)

    # Verify mock call
    mock_upload_bec.assert_called_once()
    call_args = mock_upload_bec.call_args
    assert call_args.kwargs["project_id"] == "p1"
    # Check that file path passed to upload_bec exists (it is a temp file)
    file_path = call_args.kwargs["file_path"]
    assert isinstance(file_path, Path)
    # The file should exist during the call, but might be cleaned up?
    # No, in my implementation I keep it.
    assert file_path.exists()

def test_run_assay_no_deps(mock_run_suite):
    # Dependencies not set, should fail
    # We must reset deps because they are global
    set_dependencies(None, None) # type: ignore

    corpus_data = {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "project_id": "p1",
        "name": "test",
        "version": "v1",
        "created_by": "me",
        "cases": []
    }

    payload = {
        "corpus": corpus_data,
        "agent_version": "1.0.0",
        "graders": {}
    }

    response = client.post("/run", json=payload)
    assert response.status_code == 503
    assert "AgentRunner not initialized" in response.json()["detail"]

@pytest.mark.asyncio
async def test_run_assay_success(mock_run_suite, mock_agent_runner, mock_llm_client):
    # Set dependencies
    set_dependencies(mock_agent_runner, mock_llm_client)

    mock_report = ReportCard(
        run_id="123e4567-e89b-12d3-a456-426614174001",
        total_cases=10,
        passed_cases=9,
        failed_cases=1,
        pass_rate=0.9,
        aggregates=[AggregateMetric(name="Avg Latency", value=100.0, total_samples=10)]
    )
    mock_run_suite.return_value = mock_report

    corpus_data = {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "project_id": "p1",
        "name": "test",
        "version": "v1",
        "created_by": "me",
        "cases": []
    }

    payload = {
        "corpus": corpus_data,
        "agent_version": "1.0.0",
        "graders": {
            "Latency": {"threshold_ms": 2000.0},
            "Faithfulness": {}
        }
    }

    response = client.post("/run", json=payload)

    assert response.status_code == 200
    assert response.json()["pass_rate"] == 0.9

    mock_run_suite.assert_called_once()
    kwargs = mock_run_suite.call_args.kwargs
    assert kwargs["agent_draft_version"] == "1.0.0"
    assert len(kwargs["graders"]) == 2

    # Check grader types
    graders = kwargs["graders"]
    grader_names = {g.__class__.__name__ for g in graders}
    assert "LatencyGrader" in grader_names
    assert "FaithfulnessGrader" in grader_names

def test_run_assay_missing_llm_client(mock_run_suite, mock_agent_runner):
    # Set only agent runner
    set_dependencies(mock_agent_runner, None) # type: ignore

    corpus_data = {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "project_id": "p1",
        "name": "test",
        "version": "v1",
        "created_by": "me",
        "cases": []
    }

    payload = {
        "corpus": corpus_data,
        "agent_version": "1.0.0",
        "graders": {
            "Faithfulness": {}
        }
    }

    response = client.post("/run", json=payload)
    assert response.status_code == 503
    assert "LLMClient not initialized" in response.json()["detail"]
