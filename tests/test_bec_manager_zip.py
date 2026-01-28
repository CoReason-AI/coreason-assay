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


class TestBECManagerZip:
    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def dummy_pdf(self, temp_dir: Path) -> Path:
        p = temp_dir / "protocol.pdf"
        p.write_bytes(b"%PDF-1.4 dummy content")
        return p

    def create_csv_manifest(self, path: Path, file_ref: str = "protocol.pdf") -> None:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["corpus_id", "prompt", "files", "expected_text"])
            writer.writeheader()
            writer.writerow(
                {
                    "corpus_id": str(uuid4()),
                    "prompt": "Analyze this.",
                    "files": json.dumps([file_ref]),
                    "expected_text": "Looks good.",
                }
            )

    def create_jsonl_manifest(self, path: Path, file_ref: str = "protocol.pdf") -> None:
        data = {
            "corpus_id": str(uuid4()),
            "inputs": {"prompt": "Analyze this.", "files": [file_ref]},
            "expectations": {"text": "Looks good."},
        }
        with path.open("w") as f:
            f.write(json.dumps(data) + "\n")

    def test_load_valid_zip_csv(self, temp_dir: Path, dummy_pdf: Path) -> None:
        # Create manifest
        manifest_path = temp_dir / "manifest.csv"
        self.create_csv_manifest(manifest_path, file_ref="protocol.pdf")

        # Create ZIP
        zip_path = temp_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname="manifest.csv")
            zf.write(dummy_pdf, arcname="protocol.pdf")

        # Extract dir
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        cases = BECManager.load_from_zip(zip_path, extract_dir)

        assert len(cases) == 1
        assert len(cases[0].inputs.files) == 1
        # Check if file path is absolute and exists
        resolved_file = Path(cases[0].inputs.files[0])
        assert resolved_file.is_absolute()
        assert resolved_file.exists()
        assert resolved_file.name == "protocol.pdf"
        # Compare resolved paths to handle symlinks (e.g. /var vs /private/var on macOS)
        assert resolved_file.parent.resolve() == extract_dir.resolve()

    def test_load_valid_zip_jsonl(self, temp_dir: Path, dummy_pdf: Path) -> None:
        manifest_path = temp_dir / "manifest.jsonl"
        self.create_jsonl_manifest(manifest_path, file_ref="protocol.pdf")

        zip_path = temp_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname="manifest.jsonl")
            zf.write(dummy_pdf, arcname="protocol.pdf")

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        cases = BECManager.load_from_zip(zip_path, extract_dir)
        assert len(cases) == 1
        assert cases[0].inputs.files[0].endswith("protocol.pdf")

    def test_zip_missing_manifest(self, temp_dir: Path, dummy_pdf: Path) -> None:
        zip_path = temp_dir / "no_manifest.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(dummy_pdf, arcname="protocol.pdf")

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        with pytest.raises(ValueError, match="No manifest file"):
            BECManager.load_from_zip(zip_path, extract_dir)

    def test_zip_multiple_manifests(self, temp_dir: Path) -> None:
        zip_path = temp_dir / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.csv", "header\n")
            zf.writestr("b.jsonl", "{}\n")

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        with pytest.raises(ValueError, match="Ambiguous ZIP content"):
            BECManager.load_from_zip(zip_path, extract_dir)

    def test_zip_missing_asset(self, temp_dir: Path) -> None:
        # Manifest refers to missing.pdf
        manifest_path = temp_dir / "manifest.csv"
        self.create_csv_manifest(manifest_path, file_ref="missing.pdf")

        zip_path = temp_dir / "broken.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname="manifest.csv")

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Referenced asset not found"):
            BECManager.load_from_zip(zip_path, extract_dir)

    def test_zip_security_traversal(self, temp_dir: Path, dummy_pdf: Path) -> None:
        # Manifest refers to ../protocol.pdf
        manifest_path = temp_dir / "manifest.csv"
        self.create_csv_manifest(manifest_path, file_ref="../protocol.pdf")

        zip_path = temp_dir / "attack.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname="folder/manifest.csv")
            # We put the pdf in root, but manifest is in subfolder and tries to go up
            # Wait, `../protocol.pdf` relative to `folder/manifest.csv` is `protocol.pdf` (in root of extract dir).
            # That is actually valid.
            # To test security, we need to try to go OUT of extract dir.
            # ../../../etc/passwd style.
            zf.write(dummy_pdf, arcname="protocol.pdf")

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        # Let's try a path that definitely goes out
        manifest_path_2 = temp_dir / "bad_manifest.csv"
        self.create_csv_manifest(manifest_path_2, file_ref="../../outside.pdf")

        zip_path_2 = temp_dir / "attack_real.zip"
        with zipfile.ZipFile(zip_path_2, "w") as zf:
            zf.write(manifest_path_2, arcname="manifest.csv")

        with pytest.raises(ValueError, match="Security Error"):
            BECManager.load_from_zip(zip_path_2, extract_dir)

    def test_zip_nested_folder_structure(self, temp_dir: Path, dummy_pdf: Path) -> None:
        # ZIP Structure:
        # /root
        #   /data
        #     manifest.csv
        #     /files
        #       doc.pdf

        # Manifest refers to "files/doc.pdf"

        root = temp_dir / "root"
        root.mkdir()
        data = root / "data"
        data.mkdir()
        files = data / "files"
        files.mkdir()

        manifest_path = data / "manifest.csv"
        self.create_csv_manifest(manifest_path, file_ref="files/doc.pdf")

        shutil.copy(dummy_pdf, files / "doc.pdf")

        zip_path = temp_dir / "nested.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname="root/data/manifest.csv")
            zf.write(files / "doc.pdf", arcname="root/data/files/doc.pdf")

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        cases = BECManager.load_from_zip(zip_path, extract_dir)
        assert len(cases) == 1
        resolved = Path(cases[0].inputs.files[0])
        assert resolved.exists()
        assert resolved.name == "doc.pdf"
