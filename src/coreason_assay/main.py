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
from typing import Optional
from uuid import UUID

import typer
from typing_extensions import Annotated

from coreason_assay.services import upload_bec
from coreason_assay.utils.logger import logger

app = typer.Typer(
    help="Coreason Assay CLI - Quality Control for AI Agents",
    add_completion=False,
)


@app.command()
def hello():
    """
    Sanity check command.
    """
    logger.info("Hello World!")
    typer.echo("Hello World!")


@app.command()
def upload(
    file_path: Annotated[Path, typer.Argument(..., help="Path to the ZIP file containing the BEC.")],
    project_id: Annotated[str, typer.Option("--project-id", "-p", help="Project ID")] = "default-project",
    name: Annotated[str, typer.Option("--name", "-n", help="Name of the corpus")] = "New Corpus",
    version: Annotated[str, typer.Option("--version", "-v", help="Version of the corpus")] = "1.0.0",
    created_by: Annotated[str, typer.Option("--author", "-a", help="Creator identifier")] = "cli_user",
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Extraction directory")] = Path("./data/extracted"),
):
    """
    Upload and digest a Benchmark Evaluation Corpus (BEC) from a ZIP file.
    """
    try:
        corpus = upload_bec(
            file_path=file_path,
            extraction_dir=output_dir,
            project_id=project_id,
            name=name,
            version=version,
            created_by=created_by,
        )
        typer.echo(f"Successfully uploaded Corpus: {corpus.name} (ID: {corpus.id}) with {len(corpus.cases)} cases.")
    except Exception as e:
        logger.exception("Failed to upload BEC")
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
