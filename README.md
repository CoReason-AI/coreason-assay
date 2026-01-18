# coreason-assay

**The Scientific Testing Engine for AI Agents.**

[![CI/CD](https://github.com/CoReason-AI/coreason-assay/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/CoReason-AI/coreason-assay/actions/workflows/ci-cd.yml)
[![Docker](https://github.com/CoReason-AI/coreason-assay/actions/workflows/docker.yml/badge.svg)](https://github.com/CoReason-AI/coreason-assay/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/CoReason-AI/coreason-assay/graph/badge.svg)](https://codecov.io/gh/CoReason-AI/coreason-assay)
[![PyPI version](https://badge.fury.io/py/coreason_assay.svg)](https://badge.fury.io/py/coreason_assay)
[![Python versions](https://img.shields.io/pypi/pyversions/coreason_assay.svg)](https://pypi.org/project/coreason_assay/)
[![License](https://img.shields.io/badge/license-Prosperity--3.0-blue)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

`coreason-assay` is the Quality Control (QC) laboratory of the CoReason platform. It provides a rigorous framework for evaluating the performance, safety, and alignment of AI agents before they are deployed to production.

## Features

-   **Benchmark Evaluation Corpus (BEC) Management**: Easily ingest test cases from CSV, JSONL, or ZIP archives.
-   **Simulation**: Run agents in a controlled sandbox with mocked tools and injected context.
-   **Glass Box Grading**: Evaluate not just the answer, but the reasoning process (Faithfulness, Alignment, Tone).
-   **Report Cards**: Generate detailed reports with drift detection and pass/fail metrics.

## Quick Start

### Installation

```sh
poetry install
```

### Usage

Run the CLI to upload a test corpus:

```sh
poetry run coreason-assay upload path/to/bec_archive.zip
```

### Documentation

For full documentation, including architecture details, usage guides, and examples, please visit the [docs](docs/) folder.

-   [Architecture](docs/architecture.md)
-   [Usage Guide](docs/usage.md)
-   [Vignette](docs/vignette.md)
-   [Product Requirements](docs/prd.md)

## License

This software is proprietary and dual-licensed. See `LICENSE` for details.
