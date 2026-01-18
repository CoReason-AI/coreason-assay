# coreason-assay

**The Scientific Testing Engine for AI Agents.**

[![CI](https://github.com/CoReason-AI/coreason_assay/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_assay/actions/workflows/ci.yml)

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
