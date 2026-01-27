# Usage Guide

`coreason-assay` can be used either as a command-line interface (CLI) tool for local development or as a centralized microservice (Service C).

## CLI Usage

The CLI is useful for ad-hoc testing and local corpus management.

### Commands

*   `coreason-assay hello`: Sanity check command.
*   `coreason-assay upload`: Upload and ingest a BEC ZIP file.

```bash
# Example: Upload a corpus
poetry run coreason-assay upload ./path/to/corpus.zip --project-id "proj-123" --version "1.0.0"
```

## Server Usage (Service C)

When running in Server Mode (e.g., via Docker), the service exposes a REST API on port `8000`.

### Endpoints

#### `POST /upload`
Uploads a Benchmark Evaluation Corpus (BEC) ZIP file.

**Request:** `multipart/form-data`
*   `file`: The ZIP file containing the corpus and manifest.
*   `project_id`: String ID of the project.
*   `name`: Name of the corpus.
*   `version`: Version string.
*   `author`: Creator identifier.

**Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@./my_corpus.zip" \
  -F "project_id=proj-A" \
  -F "name=GoldenSet" \
  -F "version=v1.0" \
  -F "author=jdoe"
```

#### `POST /run`
Executes an assay (test suite) against a specific agent version.

**Request:** `application/json`
*   `corpus`: The full `TestCorpus` object (typically returned by `/upload`).
*   `agent_version`: Version string of the agent being tested.
*   `graders`: Dictionary of graders to apply, with their configurations.

**Example:**
```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "corpus": { ... },
    "agent_version": "0.5.0-draft",
    "graders": {
      "Latency": { "threshold_ms": 2000 },
      "Faithfulness": {},
      "Reasoning": {}
    }
  }'
```

#### `GET /health`
Returns the service health status and version.

**Example:**
```bash
curl http://localhost:8000/health
# Output: {"status": "healthy", "service": "coreason-assay", "version": "0.3.0"}
```

## Running the Server

You can run the server using Uvicorn directly or via Docker.

**Using Uvicorn:**
```bash
poetry run uvicorn coreason_assay.server:app --reload
```

**Using Docker:**
```bash
docker build -t coreason-assay:0.3.0 .
docker run -p 8000:8000 coreason-assay:0.3.0
```
