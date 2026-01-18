# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from typing import Optional

import pytest

from coreason_assay.interfaces import LLMClient


class MockLLMClient(LLMClient):
    """
    Concrete implementation of LLMClient for testing purposes.
    """

    def __init__(self, fixed_response: str = "Mock Response"):
        self.fixed_response = fixed_response
        self.last_prompt: Optional[str] = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.fixed_response


def test_llm_client_instantiation() -> None:
    client = MockLLMClient(fixed_response="Hello")
    assert isinstance(client, LLMClient)


def test_llm_client_complete() -> None:
    client = MockLLMClient(fixed_response="Success")
    response = client.complete("Test Prompt")

    assert response == "Success"
    assert client.last_prompt == "Test Prompt"


def test_abstract_class_enforcement() -> None:
    """
    Verify that LLMClient cannot be instantiated directly.
    """
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]
