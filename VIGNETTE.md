# The Architecture and Utility of coreason-assay

### 1. The Philosophy (The Why)

In the deterministic world of traditional software, a unit test either passes or fails. `assert result == expected` is the law. But in the probabilistic domain of Generative AI, agents are "soft" logic engines. They don't just return values; they *think*, and their reasoning can be right for the wrong reasons, or wrong in subtle, semantic ways.

**coreason-assay** was born from the insight that evaluating an agent requires a "Scientific Testing Engine"—a Quality Control laboratory akin to pharmaceutical bio-assays. It is not enough to check if the JSON is valid; we must verify the "Cognitive Assay."

The package addresses three critical pain points in Agent Engineering:
1.  **Glass Box Grading:** We cannot treat agents as black boxes. To ensure reliability, we must grade the *thought process* (the "Chain of Thought") alongside the final answer. If an agent arrives at the correct conclusion but skips mandatory safety checks, it has failed.
2.  **The "Run-Verify-Score" Loop:** Effective evaluation requires a rigorous cycle of executing "Golden Data" (the Benchmark Evaluation Corpus or BEC), verifying structural integrity (Schema, Latency), and scoring semantic alignment (Faithfulness, Tone).
3.  **Collaborative QC:** Test cases are not just code; they are business assets. `coreason-assay` facilitates a workflow where SREs and Domain Experts define the "Expected Reality"—including complex mocked environments—and the system rigorously enforces it.

### 2. Under the Hood (The Dependencies & logic)

The architecture of `coreason-assay` is built on a foundation of strict typing and high concurrency, ensuring that the "Scientific Testing Engine" is both robust and fast.

*   **Pydantic (`pydantic`):** This is the bedrock of the system. From `TestCase` definitions to the final `ReportCard`, every entity is a strongly-typed model. This ensures that the "Golden Data" is validated before execution begins, and the resulting metrics are structured and queryable.
*   **Asyncio (`asyncio`):** Speed is critical in evaluation. The `Simulator` leverages Python's `asyncio` (specifically `TaskGroup`) to execute entire test corpora concurrently. This design choice transforms what could be an hour-long serial regression test into a rapid feedback loop, enabling real-time "progress bars" for developers.
*   **JSON Schema (`jsonschema`):** While LLMs produce text, systems consume structure. The package integrates `jsonschema` to provide deterministic validation of agent outputs, ensuring they adhere to strict API contracts.

**The Architectural Flow:**

1.  **The BEC Manager:** This component acts as the secure ingestion engine. It loads test cases from CSVs, JSONL files, or ZIP archives. crucially, it handles the "Simulated Reality"—resolving paths to mock files (like fake PDFs for RAG testing) and ensuring they are securely contained within the extraction root.
2.  **The Simulator:** The runner is agnostic to how the agent is deployed. It accepts an `AgentRunner` interface and executes it within a sandbox. It handles the mechanics of "Mocking"—injecting tool errors (like a database 503) or specific context—measuring raw performance metrics like latency along the way.
3.  **The Graders:** This is the core logic engine. The system uses a pluggable `BaseGrader` architecture.
    *   **Deterministic Graders** (`LatencyGrader`, `JsonSchemaGrader`) provide binary pass/fail signals based on hard constraints.
    *   **Probabilistic Graders** (`FaithfulnessGrader`, `ReasoningGrader`, `ToneGrader`) employ "LLM-as-a-Judge" patterns. They utilize an `LLMClient` to evaluate soft metrics, checking if the agent's reasoning trace aligns with the expected logical milestones or if the response contradicts the source documents.
4.  **The Assessment Engine:** The orchestrator that binds simulation and grading. It utilizes an event-driven approach to grade results immediately upon completion, allowing for streaming feedback.

### 3. In Practice (The How)

Using `coreason-assay` involves defining *how* to run your agent and *what* to grade. The following example demonstrates setting up a "Run-Verify-Score" loop with both structural and semantic checks.

```python
import asyncio
from typing import Any, Dict

from coreason_assay.engine import AssessmentEngine
from coreason_assay.grader import LatencyGrader, FaithfulnessGrader
from coreason_assay.interfaces import AgentRunner, LLMClient
from coreason_assay.models import TestCaseInput, TestResultOutput, TestCorpus, TestCase, TestCaseExpectation
from coreason_assay.simulator import Simulator

# 1. Define the Bridge to your Agent
class MyAgentRunner(AgentRunner):
    async def invoke(
        self, inputs: TestCaseInput, context: Dict[str, Any], tool_mocks: Dict[str, Any]
    ) -> TestResultOutput:
        # In a real scenario, you would call your agent's API or internal function here.
        # We simulate an agent response.
        return TestResultOutput(
            text="The patient's blood pressure is elevated.",
            trace="Checked patient vitals. BP is 140/90. Flagged as elevated.",
            structured_output={"status": "warning", "code": "BP_HIGH"}
        )

# 2. Define the Bridge for LLM-as-a-Judge (for semantic grading)
class MyLLMClient(LLMClient):
    def complete(self, prompt: str) -> str:
        # Connect to an LLM (e.g., OpenAI, Anthropic) to evaluate the response.
        # Simulating a judge's response:
        return '{"faithful": "true", "score": 1.0, "reasoning": "The response is supported by the context."}'

async def main():
    # 3. Setup the Environment
    runner = MyAgentRunner()
    llm_client = MyLLMClient()

    # 4. Construct the "Golden Data" (Or load via BECManager)
    corpus = TestCorpus(
        project_id="proj_001",
        name="Vital Signs Check",
        version="1.0",
        created_by="Dr. SRE",
        cases=[
            TestCase(
                corpus_id="corpus_123",
                inputs=TestCaseInput(prompt="Check vitals", context={"patient_id": "123"}),
                expectations=TestCaseExpectation(
                    text="elevated",
                    latency_threshold_ms=2000
                )
            )
        ]
    )

    # 5. Configure the QC Lab (The Graders)
    graders = [
        LatencyGrader(threshold_ms=500),  # Structural Check: Must be fast
        FaithfulnessGrader(llm_client=llm_client)  # Semantic Check: Must be true
    ]

    # 6. Run the Assay
    simulator = Simulator(runner=runner)
    engine = AssessmentEngine(simulator=simulator, graders=graders)

    print(f"Running Assay on {len(corpus.cases)} cases...")
    report_card = await engine.run_assay(corpus=corpus, agent_draft_version="v0.9-beta")

    # 7. Analyze Results
    print(f"\nReport Card ID: {report_card.id}")
    print(f"Pass Rate: {report_card.pass_rate * 100}%")
    for agg in report_card.aggregates:
        print(f" - {agg.name}: {agg.value}{agg.unit or ''}")

if __name__ == "__main__":
    asyncio.run(main())
```

In this snippet, we successfully decouple the *testing logic* from the *agent implementation*. The `AssessmentEngine` handles the complexity of execution and scoring, returning a comprehensive `ReportCard` that tells you not just *if* your agent failed, but *why*.
