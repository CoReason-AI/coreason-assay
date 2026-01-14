# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import csv
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Generator
from uuid import uuid4

import pytest

from coreason_assay.bec_manager import BECManager


class TestBECManagerZipCoverage:
    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_zip_file_not_found(self, temp_dir: Path) -> None:
        """Test FileNotFoundError when ZIP path does not exist."""
        missing_zip = temp_dir / "nonexistent.zip"
        target_dir = temp_dir / "extracted"
        target_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="ZIP file not found"):
            BECManager.load_from_zip(missing_zip, target_dir)

    def test_zip_invalid_file(self, temp_dir: Path) -> None:
        """Test ValueError when ZIP file is corrupt."""
        bad_zip = temp_dir / "bad.zip"
        bad_zip.write_text("This is not a zip file")

        target_dir = temp_dir / "extracted"
        target_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid ZIP file"):
            BECManager.load_from_zip(bad_zip, target_dir)

    def test_zip_with_url_references(self, temp_dir: Path) -> None:
        """Test that URL references (s3://, http://) are ignored by path resolution."""
        manifest_path = temp_dir / "manifest.csv"

        # Create a manifest with a URL file reference
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["corpus_id", "prompt", "files", "expected_text"])
            writer.writeheader()
            writer.writerow(
                {
                    "corpus_id": str(uuid4()),
                    "prompt": "Test URL",
                    "files": json.dumps(["s3://bucket/file.pdf", "http://example.com/doc.pdf"]),
                    "expected_text": "Expectation",
                }
            )

        zip_path = temp_dir / "url_test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname="manifest.csv")

        target_dir = temp_dir / "extracted"
        target_dir.mkdir()

        cases = BECManager.load_from_zip(zip_path, target_dir)

        assert len(cases) == 1
        files = cases[0].inputs.files
        assert len(files) == 2
        assert files[0] == "s3://bucket/file.pdf"
        assert files[1] == "http://example.com/doc.pdf"
