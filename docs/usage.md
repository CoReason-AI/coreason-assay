# Usage Guide

## Installation

`coreason-assay` requires Python 3.12+ and Poetry.

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/CoReason-AI/coreason_assay.git
    cd coreason_assay
    ```

2.  **Install dependencies:**
    ```sh
    poetry install
    ```

## CLI Commands

The package provides a Command Line Interface (CLI) to interact with the system.

### Basic Check
To verify your installation:
```sh
poetry run coreason-assay hello
```

### Uploading a Benchmark Corpus (BEC)
Use the `upload` command to ingest a ZIP file containing your test cases and assets.

```sh
poetry run coreason-assay upload path/to/bec_archive.zip \
    --project-id "my-project" \
    --name "Protocol Tests" \
    --version "1.0.0" \
    --author "jane.doe"
```

**Arguments:**
*   `FILE_PATH`: Path to the ZIP file containing the BEC.

**Options:**
*   `--project-id`, `-p`: ID of the project (default: "default-project").
*   `--name`, `-n`: Name of the corpus (default: "New Corpus").
*   `--version`, `-v`: Version of the corpus (default: "1.0.0").
*   `--author`, `-a`: Creator identifier (default: "cli_user").
*   `--output`, `-o`: Directory to extract assets to (default: "./data/extracted").

## Data Formats

The BEC Manager supports loading test cases from **CSV**, **JSONL**, or a **ZIP archive**.

### ZIP Archive Structure
A valid ZIP archive must contain:
1.  **Manifest File:** A single `.csv` or `.jsonl` file defining the test cases.
2.  **Assets:** Any files referenced in the test cases (e.g., PDFs, images) must be included in the archive.

**Example Structure:**
```
my_tests.zip
├── manifest.csv
├── documents/
│   ├── protocol_A.pdf
│   └── protocol_B.pdf
└── images/
    └── chart.png
```

### CSV Format
When using CSV, complex nested fields (like lists or dictionaries) must be JSON-encoded strings.

| Column | Description | Example |
| :--- | :--- | :--- |
| `prompt` | User input text | "Analyze this patient." |
| `files` | JSON list of file paths (relative to ZIP root) | `["documents/protocol_A.pdf"]` |
| `context` | JSON dict of user context | `{"role": "doctor", "time": "14:00"}` |
| `tool_outputs` | JSON dict of mock tool responses | `{"search_db": [{"id": 1, "name": "Drug X"}]}` |
| `expected_text` | Expected text output (fuzzy match) | "The patient needs Drug X." |
| `expected_reasoning` | JSON list of required reasoning steps | `["Check vitals", "Prescribe"]` |
| `expected_structure` | JSON dict for exact structure match | `{"diagnosis": "Flu"}` |
| `forbidden_content` | JSON list of negative constraints | `["Do not mention pricing"]` |
| `tool_mocks` | JSON dict for error injection | `{"patient_db": "503 Error"}` |

### JSONL Format
Each line is a valid JSON object matching the `TestCase` schema. This format is preferred for complex data as it doesn't require double-encoding JSON strings.

```json
{
  "inputs": {
    "prompt": "Analyze this.",
    "files": ["documents/doc1.pdf"],
    "context": {"user": "admin"}
  },
  "expectations": {
    "text": "Analysis complete.",
    "reasoning": ["Step 1", "Step 2"]
  }
}
```
