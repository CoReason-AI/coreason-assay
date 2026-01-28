# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from abc import ABC, abstractmethod
from typing import Any, Dict

from coreason_identity.models import UserContext

from coreason_assay.models import TestCaseInput, TestResultOutput


class AgentRunner(ABC):
    """
    Abstract protocol for running an agent.
    The consuming application must provide a concrete implementation
    (e.g., via HTTP, subprocess, or direct import).
    """

    @abstractmethod
    async def invoke(
        self, inputs: TestCaseInput, user_context: UserContext, tool_mocks: Dict[str, Any]
    ) -> TestResultOutput:
        """
        Invokes the agent with the given inputs and context.

        Args:
            inputs: The simulated input for the agent (prompt, files, etc.).
            user_context: Validated user identity context.
            tool_mocks: Mock configuration for tools (e.g., error injection).

        Returns:
            TestResultOutput: The raw output from the agent (text, trace, structured_output).
        """
        pass  # pragma: no cover


class LLMClient(ABC):
    """
    Abstract protocol for interacting with an LLM provider.
    Used by Probabilistic Graders (LLM-as-a-Judge).
    """

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Generates a text completion for the given prompt.

        Args:
            prompt: The input prompt string.

        Returns:
            str: The LLM's response text.
        """
        pass  # pragma: no cover
