# Coreason Kernel Integration

`coreason-assay` is a strict consumer of the **CoReason Kernel** via the `coreason-manifest` package. This ensures type-safe interoperability between the Foundry (which defines agents), the Simulator (which generates execution traces), and the Assay engine (which grades them).

## Dependency

The project depends on:
```toml
coreason-manifest = "^0.9.0"
```

## Key Data Structures

### 1. SimulationTrace

**Location:** `coreason_manifest.definitions.simulation.SimulationTrace`

The `SimulationTrace` replaces raw text logs as the primary artifact of an agent's execution. It provides a structured, immutable record of what happened.

**Structure:**
- `trace_id` (UUID): Unique identifier for the execution trace.
- `agent_version` (str): The version of the agent that ran.
- `steps` (List[SimulationStep]): An ordered sequence of atomic execution units.
    - `step_id` (UUID)
    - `timestamp` (datetime)
    - `node_id` (str): The graph node executed.
    - `inputs` (dict): Inputs to the step.
    - `thought` (str): Chain-of-thought reasoning.
    - `action` (dict): Tool calls or API requests.
    - `observation` (dict): Tool outputs.
- `outcome` (dict): The final result/output of the simulation.
- `metrics` (dict): Execution metrics (e.g., token usage, cost).

**Usage in Assay:**
- The `Simulator` produces a `TestResultOutput` containing a `trace` field of type `Optional[SimulationTrace]`.
- Graders (specifically `ReasoningGrader`) consume this structured object. It is serialized to JSON (via `model_dump_json()`) when constructed into LLM prompts for evaluation.

### 2. AgentDefinition

**Location:** `coreason_manifest.definitions.agent.AgentDefinition`

The `AgentDefinition` represents the static configuration and topology of the agent under test.

**Structure:**
- `name` (str): Agent name.
- `version` (str): Semantic version.
- `config` (AgentRuntimeConfig): The runtime configuration (formerly `topology`).
    - `system_prompt` (str): The system prompt template.
    - `model_config` (dict): LLM parameters (temp, top_p, etc.).
- `integrity_hash` (str): Hash for verifying the definition's integrity.

**Usage in Assay:**
- The `AssessmentEngine` and `Simulator` accept an optional `agent` argument.
- This object is passed down to `Grader.grade()` methods, allowing for **White-Box / Glass-Box Grading**.
- Example: A grader can verify if the agent's output adheres to constraints defined in `agent.config.system_prompt`.

## Workflow Integration

1.  **Input:** The CLI/Server receives a `SimulationTrace` (from a previous run) or generates one via the `Simulator`. It also receives the `AgentDefinition`.
2.  **Execution:** The `Simulator` runs the agent. If successful, it populates `TestResultOutput.trace` with a valid `SimulationTrace` object.
3.  **Grading:** The `AssessmentEngine` iterates over graders.
    - `ReasoningGrader` inspects `trace.steps` to validate logic.
    - Custom graders can inspect `agent.config` to validate alignment with the definition.
4.  **Reporting:** Results are aggregated into a `ReportCard`.

## Utilities

`src/coreason_assay/utils/parsing.py` provides helpers:
- `load_trace(json_str: str) -> SimulationTrace`
- `load_agent(json_str: str) -> AgentDefinition`

These ensure that any JSON input is strictly validated against the Kernel schema before processing.
