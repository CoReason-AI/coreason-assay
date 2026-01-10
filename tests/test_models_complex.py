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
from uuid import uuid4

import pytest
from pydantic import ValidationError

from coreason_assay.models import (
    TestCase,
    TestCaseExpectation,
    TestCaseInput,
    TestCorpus,
    TestRun,
    TestRunStatus,
)


class TestModelsComplex:
    def test_full_corpus_serialization_roundtrip(self) -> None:
        """
        Complex Scenario: Create a full TestCorpus with multiple TestCases,
        serialize it to JSON, and deserialize it back.
        """
        corpus_id = uuid4()
        case1_id = uuid4()
        case2_id = uuid4()

        case1 = TestCase(
            id=case1_id,
            corpus_id=corpus_id,
            inputs=TestCaseInput(
                prompt="Analyze this PDF",
                files=["s3://bucket/doc1.pdf"],
                context={"user_role": "doctor", "department": "cardiology"},
            ),
            expectations=TestCaseExpectation(
                text="The patient has hypertension.",
                reasoning=["Check vitals", "Compare to guidelines"],
                schema_id=None,
                structure=None,
            ),
        )

        case2 = TestCase(
            id=case2_id,
            corpus_id=corpus_id,
            inputs=TestCaseInput(
                prompt="What is the capital of France?",
                tool_outputs={"geo_api": {"lat": 48.8566, "long": 2.3522, "city": "Paris"}},
            ),
            expectations=TestCaseExpectation(
                structure={"city": "Paris", "country": "France"},
                tool_mocks={"geo_api": {"error": "timeout"}},
                text=None,
                schema_id=None,
            ),
        )

        corpus = TestCorpus(
            id=corpus_id,
            project_id="proj-alpha",
            name="Medical & General Knowledge",
            version="1.0.0",
            created_by="qa-lead@coreason.ai",
            cases=[case1, case2],
        )

        # 1. Dump to JSON string
        json_data = corpus.model_dump_json()

        # 2. Parse back to dict
        data_dict = json.loads(json_data)

        # 3. Validate back to Object
        restored_corpus = TestCorpus.model_validate(data_dict)

        assert restored_corpus.id == corpus_id
        assert len(restored_corpus.cases) == 2
        assert restored_corpus.cases[0].id == case1_id
        assert restored_corpus.cases[1].inputs.tool_outputs["geo_api"]["city"] == "Paris"
        assert restored_corpus.cases[1].expectations.tool_mocks["geo_api"]["error"] == "timeout"

    def test_complex_any_fields(self) -> None:
        """
        Edge Case: Verify that 'Dict[str, Any]' fields handle deeply nested and mixed types.
        """
        complex_context = {
            "history": [
                {"date": "2023-01-01", "event": "admitted"},
                {"date": "2023-01-02", "vitals": [98.6, 120, 80]},
            ],
            "metadata": {"source": "EHR", "verified": True, "confidence": 0.95},
        }

        inputs = TestCaseInput(prompt="Summarize history", context=complex_context)

        assert inputs.context["history"][1]["vitals"][1] == 120
        assert inputs.context["metadata"]["verified"] is True

        # Ensure it serializes correctly
        dumped = inputs.model_dump()
        assert dumped["context"] == complex_context

    def test_enum_validation_edge_cases(self) -> None:
        """
        Edge Case: Test invalid values for Enums.
        """
        # Valid status
        run = TestRun(corpus_version="1.0", agent_draft_version="v1", status=TestRunStatus.FAILED)
        assert run.status == TestRunStatus.FAILED

        # Invalid status via string injection (should fail validation)
        with pytest.raises(ValidationError) as excinfo:
            TestRun(
                corpus_version="1.0",
                agent_draft_version="v1",
                status="RunningFast",  # type: ignore
            )
        assert "Input should be 'Running', 'Done' or 'Failed'" in str(excinfo.value)

    def test_uuid_string_parsing(self) -> None:
        """
        Edge Case: Pydantic should automatically parse valid UUID strings into UUID objects.
        """
        valid_uuid_str = "123e4567-e89b-12d3-a456-426614174000"

        corpus = TestCorpus(
            id=valid_uuid_str,  # type: ignore
            project_id="p1",
            name="Test",
            version="1",
            created_by="me",
        )

        assert str(corpus.id) == valid_uuid_str

        # Invalid UUID string
        with pytest.raises(ValidationError):
            TestCorpus(
                id="not-a-uuid",  # type: ignore
                project_id="p1",
                name="Test",
                version="1",
                created_by="me",
            )
