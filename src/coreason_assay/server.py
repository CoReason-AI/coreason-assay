# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from coreason_assay.engine import AssessmentEngine
from coreason_assay.grader import (
    BaseGrader,
    FaithfulnessGrader,
    ForbiddenContentGrader,
    JsonSchemaGrader,
    LatencyGrader,
    ReasoningGrader,
    ToneGrader,
)
from coreason_assay.interfaces import AgentRunner, LLMClient
from coreason_assay.models import ReportCard, TestCorpus
from coreason_assay.services import upload_bec
from coreason_assay.simulator import Simulator

app = FastAPI(
    title="Coreason Assay QC Service",
    version="0.2.0",
    description="Quality Control (QC) Microservice for executing Cognitive Assays against AI agents."
)

# Global dependencies
_agent_runner: Optional[AgentRunner] = None
_llm_client: Optional[LLMClient] = None

def set_dependencies(runner: AgentRunner, llm_client: Optional[LLMClient] = None) -> None:
    """
    Injects concrete implementations of AgentRunner and LLMClient.
    This must be called before the server starts accepting /run requests.
    """
    global _agent_runner, _llm_client
    _agent_runner = runner
    _llm_client = llm_client

class RunRequest(BaseModel):
    """
    Request body for executing an assay.
    """
    corpus: TestCorpus = Field(..., description="The Test Corpus to execute.")
    agent_version: str = Field(..., description="Version of the agent draft being tested.")
    graders: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for graders (e.g., {'Latency': {'threshold_ms': 5000}, 'Faithfulness': {}})"
    )

@app.post("/upload", response_model=TestCorpus)
def upload_corpus(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    name: str = Form(...),
    version: str = Form(...),
    author: str = Form(...),
) -> TestCorpus:
    """
    Uploads a ZIP file containing a Benchmark Evaluation Corpus (BEC) and returns the parsed TestCorpus.
    """
    # Create a temporary directory for this upload
    # Note: We use mkdtemp to ensure the extracted files persist for subsequent /run calls.
    # In a production environment with shared storage, this should be replaced with object storage upload.
    upload_dir = Path(tempfile.mkdtemp(prefix="bec_upload_"))
    zip_path = upload_dir / (file.filename or "corpus.zip")

    try:
        with zip_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extraction_dir = upload_dir / "extracted"

        corpus = upload_bec(
            file_path=zip_path,
            extraction_dir=extraction_dir,
            project_id=project_id,
            name=name,
            version=version,
            created_by=author
        )

        # Cleanup the zip file to save space, keeping only extracted files
        try:
            zip_path.unlink()
        except OSError:
            pass # Best effort

        return corpus
    except Exception as e:
        # Cleanup the created directory on failure to prevent resource leaks
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to process upload: {str(e)}")

@app.post("/run", response_model=ReportCard)
async def run_assay(request: RunRequest) -> ReportCard:
    """
    Executes the assay for the provided corpus and agent version.
    """
    if _agent_runner is None:
        raise HTTPException(status_code=503, detail="AgentRunner not initialized. Server must be configured with a concrete AgentRunner.")

    # Build graders list
    graders_list: List[BaseGrader] = []

    for name, config in request.graders.items():
        if name == "Latency":
            threshold = config.get("threshold_ms", 5000.0)
            graders_list.append(LatencyGrader(threshold_ms=threshold))

        elif name == "JsonSchema":
            graders_list.append(JsonSchemaGrader())

        elif name == "ForbiddenContent":
            graders_list.append(ForbiddenContentGrader())

        elif name == "Reasoning":
            if not _llm_client:
                raise HTTPException(status_code=503, detail="LLMClient not initialized, required for ReasoningGrader.")
            graders_list.append(ReasoningGrader(llm_client=_llm_client))

        elif name == "Faithfulness":
            if not _llm_client:
                raise HTTPException(status_code=503, detail="LLMClient not initialized, required for FaithfulnessGrader.")
            graders_list.append(FaithfulnessGrader(llm_client=_llm_client))

        elif name == "Tone":
            if not _llm_client:
                raise HTTPException(status_code=503, detail="LLMClient not initialized, required for ToneGrader.")
            graders_list.append(ToneGrader(llm_client=_llm_client))

        else:
            # Skip unknown graders
            pass

    # Initialize Simulator
    simulator = Simulator(runner=_agent_runner)

    # Initialize AssessmentEngine
    engine = AssessmentEngine(simulator=simulator, graders=graders_list)

    # Run Assay
    report_card = await engine.run_assay(
        corpus=request.corpus,
        agent_draft_version=request.agent_version,
    )

    return report_card

@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "coreason-assay", "version": "0.2.0"}
