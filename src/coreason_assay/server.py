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
from typing import Annotated, Any, Dict, List, Optional

from coreason_identity.models import UserContext
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

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
from coreason_assay.services import run_suite, upload_bec
from coreason_assay.utils.logger import logger

app = FastAPI(title="CoReason Assay Service", version="0.4.0")

# Global dependencies
_agent_runner: Optional[AgentRunner] = None
_llm_client: Optional[LLMClient] = None


def set_dependencies(runner: AgentRunner, llm_client: LLMClient) -> None:
    """
    Injects concrete implementations of AgentRunner and LLMClient.
    This should be called by the application bootstrapping logic.
    """
    global _agent_runner, _llm_client
    _agent_runner = runner
    _llm_client = llm_client
    logger.info("Dependencies injected into Assessment Engine.")


class RunRequest(BaseModel):
    corpus: TestCorpus
    agent_version: str
    graders: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


@app.get("/health")  # type: ignore[misc]
def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "coreason-assay", "version": "0.4.0"}


@app.post("/upload", response_model=TestCorpus)  # type: ignore[misc]
def upload_corpus(
    file: Annotated[UploadFile, File(...)],
    project_id: Annotated[str, Form(...)],
    name: Annotated[str, Form(...)],
    version: Annotated[str, Form(...)],
    author: Annotated[str, Form(...)],
) -> TestCorpus:
    """
    Uploads a BEC ZIP file and ingests it.
    """
    # Create persistent temp dir for this version to ensure files exist for /run
    # We use a known prefix so we can potentially clean it up later or mounting volume
    base_dir = Path(tempfile.gettempdir()) / "coreason_assay_uploads" / project_id / version
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    zip_path = base_dir / "corpus.zip"
    extraction_dir = base_dir / "extracted"

    try:
        with zip_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    try:
        # Construct UserContext from the author field (Identity Hydration)
        # We synthesize an email since the legacy endpoint doesn't provide it.
        user_context = UserContext(user_id=author, email=f"{author}@coreason.ai")

        corpus = upload_bec(
            file_path=zip_path,
            extraction_dir=extraction_dir,
            project_id=project_id,
            name=name,
            version=version,
            user_context=user_context,
        )
        return corpus
    except Exception as e:
        logger.exception("Failed to upload/ingest corpus")
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/run", response_model=ReportCard)  # type: ignore[misc]
async def run_assay(request: RunRequest) -> ReportCard:
    """
    Executes the assay for the provided corpus and agent version.
    """
    if not _agent_runner:
        raise HTTPException(status_code=503, detail="AgentRunner not initialized. Server dependencies missing.")

    graders_list: List[BaseGrader] = []

    for name, config in request.graders.items():
        try:
            if name == "Latency":
                graders_list.append(LatencyGrader(**config))
            elif name == "JsonSchema":
                graders_list.append(JsonSchemaGrader())
            elif name == "ForbiddenContent":
                graders_list.append(ForbiddenContentGrader())
            elif name == "Reasoning":
                if not _llm_client:
                    raise HTTPException(status_code=503, detail="LLMClient not initialized.")
                graders_list.append(ReasoningGrader(llm_client=_llm_client))
            elif name == "Faithfulness":
                if not _llm_client:
                    raise HTTPException(status_code=503, detail="LLMClient not initialized.")
                graders_list.append(FaithfulnessGrader(llm_client=_llm_client))
            elif name == "Tone":
                if not _llm_client:
                    raise HTTPException(status_code=503, detail="LLMClient not initialized.")
                graders_list.append(ToneGrader(llm_client=_llm_client))
            else:
                logger.warning(f"Unknown grader requested: {name}")
        except TypeError as e:
            # Handle invalid config args
            raise HTTPException(status_code=400, detail=f"Invalid configuration for grader {name}: {e}") from e

    try:
        report = await run_suite(
            corpus=request.corpus,
            agent_runner=_agent_runner,
            agent_draft_version=request.agent_version,
            graders=graders_list,
        )
        return report
    except Exception as e:
        logger.exception("Failed to run assay")
        raise HTTPException(status_code=500, detail=str(e)) from e
