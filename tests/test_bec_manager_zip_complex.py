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
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest

from coreason_assay.bec_manager import BECManager


class TestBECManagerZipComplex:
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def sensitive_file(self, temp_dir):
        # Create a file outside the extraction root
        p = temp_dir / "secret.txt"
        p.write_text("classified")
        return p

    def create_csv_manifest(self, path: Path, file_ref: str):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["corpus_id", "prompt", "files", "expected_text"]
            )
            writer.writeheader()
            writer.writerow({
                "corpus_id": str(uuid4()),
                "prompt": "Test",
                "files": json.dumps([file_ref]),
                "expected_text": "Expectation"
            })

    def test_zip_absolute_path_security(self, temp_dir, sensitive_file):
        """
        Verify that a manifest containing an absolute path to a file outside
        the extraction directory is rejected.
        """
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        manifest_path = temp_dir / "manifest.csv"
        # Use the absolute path of the sensitive file
        self.create_csv_manifest(manifest_path, file_ref=str(sensitive_file.resolve()))

        zip_path = temp_dir / "abs_path_attack.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname="manifest.csv")
            # We don't need to put the file in the zip, the manifest points to the one on disk

        with pytest.raises(ValueError, match="Security Error"):
            BECManager.load_from_zip(zip_path, extract_dir)

    def test_zip_unicode_filenames(self, temp_dir):
        """
        Test handling of unicode filenames in both manifest and ZIP structure.
        """
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        # Filename with Kanji and Emoji
        filename = "画像_🚀.png"
        manifest_name = "マニフェスト.csv"

        manifest_path = temp_dir / manifest_name
        self.create_csv_manifest(manifest_path, file_ref=filename)

        dummy_file = temp_dir / filename
        dummy_file.write_bytes(b"image data")

        zip_path = temp_dir / "unicode.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname=manifest_name)
            zf.write(dummy_file, arcname=filename)

        cases = BECManager.load_from_zip(zip_path, extract_dir)

        assert len(cases) == 1
        resolved = Path(cases[0].inputs.files[0])
        assert resolved.name == filename
        assert resolved.exists()

    def test_zip_mixed_separators(self, temp_dir):
        """
        Test that Windows-style backslashes in the manifest are handled correctly
        on Linux/Unix systems (converted to proper Path objects).
        """
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        # Manifest uses backslash
        manifest_path = temp_dir / "manifest.csv"
        self.create_csv_manifest(manifest_path, file_ref=r"subdir\doc.pdf")

        # Zip has standard forward slash
        zip_path = temp_dir / "mixed_sep.zip"

        # Create dummy file structure to zip
        content_dir = temp_dir / "content"
        content_dir.mkdir()
        subdir = content_dir / "subdir"
        subdir.mkdir()
        (subdir / "doc.pdf").write_text("content")

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname="manifest.csv")
            zf.write(subdir / "doc.pdf", arcname="subdir/doc.pdf")

        cases = BECManager.load_from_zip(zip_path, extract_dir)

        assert len(cases) == 1
        resolved = Path(cases[0].inputs.files[0])
        assert resolved.name == "doc.pdf"
        assert resolved.parent.name == "subdir"
        assert resolved.exists()

    def test_zip_symlink_security(self, temp_dir, sensitive_file):
        """
        Simulate a scenario where the extraction results in a symlink pointing outside.
        Since we can't easily create such a zip consistently, we manually 'poison'
        the target directory before calling load_from_zip, acting as if unzip did it.

        However, load_from_zip CLEANS the target? No, it just extracts over.

        We will patch zipfile.ZipFile.extractall to do nothing, and manually set up
        the extracted state with a symlink.
        """
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        # Create the 'poisoned' environment
        manifest_path = extract_dir / "manifest.csv"
        # Manifest points to "link_to_secret"
        self.create_csv_manifest(manifest_path, file_ref="link_to_secret")

        # Create symlink: extract_dir/link_to_secret -> sensitive_file
        link_path = extract_dir / "link_to_secret"
        try:
            os.symlink(sensitive_file, link_path)
        except OSError:
            pytest.skip("Symlinks not supported on this OS/env")

        # Create a dummy zip just to pass the file existence check
        zip_path = temp_dir / "dummy.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("dummy", "dummy")

        # Mock extractall to prevent overwriting our manual setup
        # We need to rely on the fact that load_from_zip finds the manifest we planted
        # But load_from_zip expects the manifest IN the zip to verify candidates.
        # So we must put the manifest in the zip too.
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(manifest_path, arcname="manifest.csv")

        # We patch extractall.
        # Actually, we can just let it extract. It will extract manifest.csv (overwriting).
        # It won't touch 'link_to_secret' because it's not in the zip.
        # But the manifest refers to it.
        # So `load_from_zip` will see "manifest.csv", parse it, see "link_to_secret",
        # resolve it to `extract_dir / link_to_secret` which is a symlink to `sensitive_file`.
        # `resolve()` will point to `sensitive_file`.
        # `relative_to` should fail.

        with pytest.raises(ValueError, match="Security Error"):
            BECManager.load_from_zip(zip_path, extract_dir)

    def test_zip_empty_file(self, temp_dir):
        """Test handling of an empty ZIP file."""
        zip_path = temp_dir / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        with pytest.raises(ValueError, match="No manifest file"):
            BECManager.load_from_zip(zip_path, extract_dir)
