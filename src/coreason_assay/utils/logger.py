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
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from coreason_assay.settings import settings

# Create a custom logger
logger = logging.getLogger("coreason_assay")
logger.setLevel(settings.LOG_LEVEL)

# Prevent duplicate logs if reload happens
if not logger.handlers:
    # Formatter
    formatter = logging.Formatter(fmt=settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)

    # Console Handler (Stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_path = settings.LOG_FILE
    # Ensure logs directory exists
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=settings.LOG_MAX_BYTES, backupCount=settings.LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# Export logger
__all__ = ["logger"]
