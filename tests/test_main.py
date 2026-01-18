# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from coreason_assay.main import app
from coreason_assay.models import TestCorpus

runner = CliRunner()


def test_hello_world() -> None:
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello World!" in result.stdout


def test_upload_success() -> None:
    """Test the upload command with valid arguments."""
    mock_corpus = MagicMock(spec=TestCorpus)
    mock_corpus.name = "Test Corpus"
    mock_corpus.id = "123-uuid"
    mock_corpus.cases = [1, 2, 3]

    with patch("coreason_assay.main.upload_bec", return_value=mock_corpus) as mock_upload:
        result = runner.invoke(
            app,
            [
                "upload",
                "dummy.zip",
                "--project-id", "proj-1",
                "--name", "Corpus 1",
                "--version", "1.0",
                "--author", "tester",
                "--output", "out_dir",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully uploaded Corpus: Test Corpus" in result.stdout

        mock_upload.assert_called_once_with(
            file_path=Path("dummy.zip"),
            extraction_dir=Path("out_dir"),
            project_id="proj-1",
            name="Corpus 1",
            version="1.0",
            created_by="tester",
        )


def test_upload_failure() -> None:
    """Test the upload command when an error occurs."""
    with patch("coreason_assay.main.upload_bec", side_effect=Exception("Upload failed")):
        result = runner.invoke(app, ["upload", "dummy.zip"])

        assert result.exit_code == 1
        assert "Error: Upload failed" in result.stdout
