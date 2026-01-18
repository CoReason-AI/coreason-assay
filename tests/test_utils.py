# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import logging
from pathlib import Path

from coreason_assay.utils.logger import logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    # Since the logger is initialized on import, we check side effects

    # Check if logs directory creation is handled
    # The new logger implementation uses settings to determine the path
    from coreason_assay.settings import settings
    log_path = settings.LOG_FILE.parent

    assert log_path.exists()
    assert log_path.is_dir()

    # Verify usage
    assert isinstance(logger, logging.Logger)
    assert logger.name == "coreason_assay"
    assert len(logger.handlers) >= 1


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None
