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
from typing import Any, Dict


def parse_json_from_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parses a JSON object from a potentially Markdown-formatted LLM response.

    Args:
        response_text: The raw text response from the LLM.

    Returns:
        Dict[str, Any]: The parsed JSON object.

    Raises:
        json.JSONDecodeError: If parsing fails.
    """
    cleaned_response = response_text.strip()

    # Remove markdown code blocks if present
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]

    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]

    return json.loads(cleaned_response.strip())
