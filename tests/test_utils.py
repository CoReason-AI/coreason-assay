# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import json
import logging
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from coreason_manifest.definitions.agent import AgentDefinition
from coreason_manifest.definitions.simulation import SimulationTrace

from coreason_assay.utils.logger import logger, setup_logger
from coreason_assay.utils.parsing import load_agent, load_trace


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
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


def test_logger_mkdir_logic() -> None:
    """
    Test that the logger setup logic attempts to create the directory if it doesn't exist.
    """
    with (
        patch("coreason_assay.utils.logger.settings") as mock_settings,
        patch("coreason_assay.utils.logger.RotatingFileHandler") as mock_handler,
    ):
        mock_path = MagicMock()
        mock_path.parent.exists.return_value = False
        mock_settings.LOG_FILE = mock_path
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.LOG_FORMAT = "%(message)s"
        mock_settings.LOG_DATE_FORMAT = "%Y-%m-%d"
        mock_settings.LOG_MAX_BYTES = 1000
        mock_settings.LOG_BACKUP_COUNT = 1

        # Clear handlers to force setup
        logger.handlers = []

        setup_logger()

        # Verify mkdir was called
        mock_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify handler creation
        mock_handler.assert_called_once()


def test_load_trace_valid() -> None:
    trace_data = {
        "trace_id": str(uuid4()),
        "agent_version": "1.0",
        "steps": [],
        "outcome": {},
        "metrics": {}
    }
    json_str = json.dumps(trace_data)
    trace = load_trace(json_str)
    assert isinstance(trace, SimulationTrace)
    assert str(trace.trace_id) == trace_data["trace_id"]


def test_load_agent_valid() -> None:
    # Minimal AgentDefinition
    agent_data = {
        "name": "Test Agent",
        "version": "1.0.0",
        "config": {
            "system_prompt": "You are a bot.",
            "model_config": {}
        },
        "integrity_hash": "hash123"
    }
    json_str = json.dumps(agent_data)
    agent = load_agent(json_str)
    assert isinstance(agent, AgentDefinition)
    assert agent.name == "Test Agent"
