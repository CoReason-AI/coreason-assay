# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

from pathlib import Path
from typing import Any, Callable, Coroutine, List, Optional, Union

from pydantic import BaseModel, Field

try:
    from coreason_identity.models import UserContext
except ImportError:
    # Mock for development/CI where the private package isn't available
    class UserContext(BaseModel):  # type: ignore
        user_id: str = Field(..., description="User ID")
        groups: List[str] = Field(default_factory=list, description="Groups")

from coreason_assay.bec_manager import BECManager
from coreason_assay.engine import AssessmentEngine
from coreason_assay.grader import BaseGrader
from coreason_assay.interfaces import AgentRunner
from coreason_assay.models import ReportCard, TestCorpus, TestResult
from coreason_assay.simulator import Simulator
from coreason_assay.utils.logger import logger


def upload_bec(
    file_path: Union[str, Path],
    extraction_dir: Union[str, Path],
    project_id: str,
    name: str,
    version: str,
    user_context: UserContext,
) -> TestCorpus:
    """
    Ingests a Benchmark Evaluation Corpus (BEC) from a ZIP file.
    Wraps BECManager to parse the file and constructs a TestCorpus object.

    Args:
        file_path: Path to the .zip file containing the corpus.
        extraction_dir: Directory where the zip contents will be extracted.
        project_id: ID of the project this corpus belongs to.
        name: Name of the corpus.
        version: Version string for the corpus.
        user_context: Context of the user creating the corpus.

    Returns:
        TestCorpus: The constructed test corpus with loaded test cases.
    """
    created_by = user_context.user_id
    logger.info(f"Uploading BEC from {file_path} (Project: {project_id}, Version: {version}, User: {created_by})")

    # Load test cases using BECManager
    # Note: BECManager.load_from_zip handles extraction and manifest parsing
    cases = BECManager.load_from_zip(file_path, extraction_dir)

    # Construct the TestCorpus object
    corpus = TestCorpus(
        project_id=project_id,
        name=name,
        version=version,
        created_by=created_by,
        cases=cases,
    )

    # Ensure all cases have the correct corpus_id
    # BECManager might have set it if present in CSV, but if not (or mismatched),
    # we should unify it with the corpus.id we just generated.
    for case in corpus.cases:
        case.corpus_id = corpus.id

    logger.info(f"Successfully created TestCorpus {corpus.id} with {len(cases)} cases.")
    return corpus


async def run_suite(
    corpus: TestCorpus,
    agent_runner: AgentRunner,
    agent_draft_version: str,
    graders: List[BaseGrader],
    user_context: UserContext,
    on_progress: Optional[Callable[[int, int, TestResult], Coroutine[Any, Any, None]]] = None,
) -> ReportCard:
    """
    Executes the full test suite for a given corpus against an agent.
    Coordinates the Simulator and AssessmentEngine.

    Args:
        corpus: The TestCorpus to execute.
        agent_runner: The implementation of the agent runner (adapter).
        agent_draft_version: The version string of the agent being tested.
        graders: List of graders to evaluate the results.
        user_context: Context of the user running the test.
        on_progress: Optional async callback for progress updates.

    Returns:
        ReportCard: The final graded report card.
    """
    run_by = user_context.user_id
    logger.info(f"Starting test suite run for Corpus {corpus.id} (Agent v{agent_draft_version}, User: {run_by})")

    # 1. Initialize Simulator with the provided AgentRunner
    simulator = Simulator(runner=agent_runner)

    # 2. Initialize AssessmentEngine with Simulator and Graders
    engine = AssessmentEngine(simulator=simulator, graders=graders)

    # 3. Run the Assay
    report_card = await engine.run_assay(
        corpus=corpus,
        agent_draft_version=agent_draft_version,
        run_by=run_by,
        on_progress=on_progress,
    )

    logger.info(f"Completed test suite run {report_card.run_id}. Pass Rate: {report_card.pass_rate:.2%}")
    return report_card
