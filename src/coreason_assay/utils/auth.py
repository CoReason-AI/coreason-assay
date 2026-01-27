# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import os
from typing import List

from pydantic import BaseModel, Field

try:
    from coreason_identity.models import UserContext
except ImportError:
    # Mock for development/CI where the private package isn't available
    class UserContext(BaseModel):  # type: ignore
        user_id: str = Field(..., description="User ID")
        email: str = Field(..., description="Email")
        groups: List[str] = Field(default_factory=list, description="Groups")


def get_cli_context() -> UserContext:
    """
    Retrieves the UserContext for the current CLI session.

    In a real implementation, this would read from a token file, environment variables,
    or query an identity provider (IDP). For this MVP/refactor, we simulate it
    using environment variables or default to a system user.
    """
    # 1. Check for specific environment variables (e.g. from CI/CD)
    user_id = os.getenv("COREASON_USER_ID", os.getenv("USER", "unknown_user"))
    email = os.getenv("COREASON_USER_EMAIL", f"{user_id}@example.com")

    # 2. In the future, decode a JWT or reading ~/.coreason/credentials

    return UserContext(user_id=user_id, email=email, groups=["cli_users"])
