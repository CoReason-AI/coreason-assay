# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TestCaseInput(BaseModel):
    """
    Simulated input for the agent.
    """

    prompt: str = Field(..., description="The user prompt text.")
    files: List[str] = Field(
        default_factory=list, description="List of file paths/URLs (e.g., S3 paths) mimicking RAG docs."
    )
    context: Dict[str, Any] = Field(default_factory=dict, description="Injected context like User Role, Date, Time.")
    tool_outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Mocked API responses to test how the agent handles data."
    )


class TestCaseExpectation(BaseModel):
    """
    Expected reality (Ground Truth) for the test case.
    """

    text: Optional[str] = Field(None, description="Expected final text (fuzzy match string).")
    schema_id: Optional[str] = Field(None, description="Expected JSON Schema ID for validation.")
    structure: Optional[Dict[str, Any]] = Field(None, description="Expected JSON structure.")
    reasoning: List[str] = Field(
        default_factory=list,
        description="List of required logical milestones (e.g., ['Identified Patient A', ...]).",
    )
    forbidden_content: List[str] = Field(default_factory=list, description="Negative constraints.")
    tool_mocks: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific mock options for tools during this test (e.g. error injection).",
    )


class TestCase(BaseModel):
    """
    A single test case within a corpus.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the test case.")
    corpus_id: UUID = Field(..., description="ID of the corpus this case belongs to.")
    inputs: TestCaseInput = Field(..., description="Simulated inputs.")
    expectations: TestCaseExpectation = Field(..., description="Expected outcomes.")


class TestCorpus(BaseModel):
    """
    A collection of test cases (Golden Data).
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the corpus.")
    project_id: str = Field(..., description="ID of the project.")
    name: str = Field(..., description="Name of the corpus.")
    version: str = Field(..., description="Version of the corpus.")
    created_by: str = Field(..., description="User who created the corpus.")
    cases: List[TestCase] = Field(default_factory=list, description="List of test cases in this corpus.")


class TestRunStatus(str, Enum):
    RUNNING = "Running"
    DONE = "Done"
    FAILED = "Failed"


class TestRun(BaseModel):
    """
    Represents an execution of a TestCorpus against a specific agent version.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the test run.")
    corpus_version: str = Field(..., description="Version of the corpus used.")
    agent_draft_version: str = Field(..., description="Version of the agent draft being tested.")
    status: TestRunStatus = Field(default=TestRunStatus.RUNNING, description="Current status of the run.")


class TestResultOutput(BaseModel):
    """
    The actual output produced by the agent.
    """

    text: Optional[str] = Field(None, description="The final text response.")
    trace: Optional[str] = Field(None, description="The execution trace/log.")
    structured_output: Optional[Any] = Field(None, description="The structured output (if any).")


class Score(BaseModel):
    """
    Represents a score for a specific dimension (e.g., Latency, Faithfulness).
    """

    name: str = Field(..., description="Name of the scoring dimension (e.g., 'Latency').")
    value: Union[float, int, bool] = Field(..., description="The quantitative score.")
    max_value: float = Field(default=1.0, description="The maximum possible value for this score.")
    passed: bool = Field(..., description="Whether this specific dimension passed.")
    reasoning: Optional[str] = Field(None, description="Explanation for the score.")


class TestResult(BaseModel):
    """
    The result of running a single TestCase.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the result.")
    run_id: UUID = Field(..., description="ID of the parent TestRun.")
    case_id: UUID = Field(..., description="ID of the TestCase.")
    actual_output: TestResultOutput = Field(..., description="The actual output from the agent.")
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Raw execution metrics (e.g., latency_ms, tokens)."
    )
    scores: List[Score] = Field(default_factory=list, description="List of graded scores.")
    passed: bool = Field(..., description="Whether the test passed based on criteria.")


class AggregateMetric(BaseModel):
    """
    Aggregated metric from a TestRun (e.g., Average Latency, Pass Rate).
    """

    name: str = Field(..., description="Name of the metric (e.g., 'Average Latency').")
    value: float = Field(..., description="The calculated aggregate value.")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g., 'ms', '%').")
    total_samples: int = Field(..., description="Number of data points included in this aggregate.")


class ReportCard(BaseModel):
    """
    The persistent result set summarizing a TestRun.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the report card.")
    run_id: UUID = Field(..., description="ID of the TestRun this report belongs to.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of generation."
    )

    # High-level stats
    total_cases: int = Field(..., description="Total number of test cases in the run.")
    passed_cases: int = Field(..., description="Number of cases that passed.")
    failed_cases: int = Field(..., description="Number of cases that failed.")
    pass_rate: float = Field(..., description="Global pass rate (0.0 to 1.0).")

    # Granular aggregates
    aggregates: List[AggregateMetric] = Field(default_factory=list, description="List of aggregated metrics.")


class DriftMetric(BaseModel):
    """
    Represents the difference for a specific metric between two runs.
    """

    name: str = Field(..., description="Name of the metric (e.g., 'Average Latency').")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g., 'ms', '%').")
    current_value: float = Field(..., description="Value in the current run.")
    previous_value: float = Field(..., description="Value in the previous run.")
    delta: float = Field(..., description="Absolute difference (current - previous).")
    pct_change: Optional[float] = Field(None, description="Percentage change relative to previous value (0.0 to 1.0+).")
    is_regression: bool = Field(..., description="True if this change is considered negative/bad.")


class DriftReport(BaseModel):
    """
    A report comparing two Test Runs to identify regressions.
    """

    current_run_id: UUID = Field(..., description="ID of the current TestRun.")
    previous_run_id: UUID = Field(..., description="ID of the previous TestRun.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of generation."
    )
    metrics: List[DriftMetric] = Field(default_factory=list, description="List of compared metrics.")
