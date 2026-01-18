# Product Requirements Document: coreason-assay

**Domain**: Agent Evaluation & Quality Control (QC)
**Architectural Role**: The Scientific Testing Engine
**Integration Target**: coreason-foundry (Source of Agents), coreason-api (UI Interface)

---

## 1. Executive Summary

coreason-assay is the Quality Control (QC) laboratory of the CoReason platform. Just as a pharmaceutical drug must pass bio-assays before release, a CoReason agent must pass "Cognitive Assays."

This package manages the ingestion of the **Benchmark Evaluation Corpus (BEC)**—the "Golden Data" used to test the agent. It executes agents in a controlled environment (isolating variables) and scores them against multi-dimensional metrics: **Faithfulness** (Did it stick to the source?), **Robustness** (Did it break on bad JSON?), and **Alignment** (Did it follow the Constitution?). It supports a multi-user environment where SREs can upload test cases, annotate expected reasoning paths, and view shared "Report Cards" in real-time.

## 2. Functional Philosophy

The agent must implement the **Run-Verify-Score Loop**:

1.  **Golden Data:** You cannot test without a baseline. The system must support complex test cases that mimic real-world ambiguity (Files + Text + Context).
2.  **Structural vs. Semantic:** We test both the *Container* (JSON Schema, Latency) and the *Content* (Medical Accuracy, Tone).
3.  **Glass Box Grading:** We do not just grade the final answer. We grade the *thought process*. If the agent arrived at the right answer for the wrong reason, it fails.
4.  **Collaborative QC:** Test suites are community assets. SRE A uploads the data; SRE B defines the expected output; SRE C runs the test.

---

## 3. Core Functional Requirements (Component Level)

### 3.1 The BEC Manager (The Data Loader)

**Concept:** A flexible ingestion engine for Test Cases.

*   **Corpus Structure:** It must support a "Test Case" entity containing:
    *   **Simulated Input:**
        *   user_prompt (Text).
        *   attached_files (PDFs/CSVs mimicking RAG docs).
        *   injected_context (User Role, Date, Time).
        *   tool_outputs (Mocked API responses to test how the agent handles data).
    *   **Expected Reality (The Ground Truth):**
        *   expected_final_text (Fuzzy match string).
        *   expected_structure (JSON Schema validation).
        *   expected_reasoning_steps (List of required logical milestones, e.g., ["Identified Patient A", "Checked Contraindications"]).
        *   forbidden_content (Negative constraints).
*   **Upload Mechanism:** Supports CSV/JSONL bulk uploads and "Single Case Creation" via UI.
*   **Versioning:** Test Corpora must be versioned. "BEC v1.0" is immutable once a test run starts.

### 3.2 The Simulator (The Runner)

**Concept:** The execution harness that runs the agent in a sandbox.

*   **Mocking:** It must be able to mock coreason-mcp tool calls. (We test the *agent's logic*, not the external database's uptime).
*   **Concurrent Execution:** It runs the test suite in parallel (using asyncio) to reduce "Time to Feedback."
*   **Real-Time Streaming:** It pipes execution logs back to coreason-foundry via WebSockets so users can watch the "Test Progress Bar."

### 3.3 The Grader (The Scorer)

**Concept:** A pluggable architecture for evaluating performance.

*   **Deterministic Graders:**
    *   **Schema Check:** Does output match Pydantic model? (Pass/Fail).
    *   **Latency Check:** Was it under 5000ms? (Pass/Fail).
    *   **Code Check:** Is the generated SQL valid? (Syntax Check).
*   **Probabilistic Graders (LLM-as-a-Judge):**
    *   **Faithfulness:** Does the answer contradict the provided attached_files?
    *   **Reasoning Alignment:** Did the agent follow the expected_reasoning_steps?
    *   **Tone Check:** Is the response "Professional" and "Empathetic"?

### 3.4 The Report Card (The Artifact)

**Concept:** The persistent result set.

*   **Scoring Matrix:** Aggregates individual scores into a high-level "pass rate" (e.g., "95% Accuracy, 100% JSON Validity").
*   **Regression flagging:** Compares current run vs. previous run. Flags "Drift" (e.g., "Latency increased by 200ms").
*   **Comment Support:** Allows users to comment on specific failed test cases (e.g., "This failure is acceptable due to new policy").

---

## 4. Integration Requirements (The Ecosystem)

*   **Source (Hook for coreason-foundry):**
    *   Fetches the *Draft Agent* logic from Foundry. It does not touch the Production codebase.
*   **UI (Hook for coreason-api):**
    *   Exposes endpoints for upload_bec, run_suite, get_results.
    *   Exposes WebSocket for live progress updates.

---

## 5. User Stories (Behavioral Expectations)

### Story A: The "Golden File" Upload

**Trigger:** SRE wants to test the "Protocol Analyzer" agent.
**Action:** SRE uploads a ZIP file containing 50 PDFs (Clinical Protocols) and a CSV manifest mapping filename -> expected_drug_name.
**System:** coreason-assay parses the CSV, creates 50 Test Cases, and links the PDFs as "Simulated RAG Documents."

### Story B: The "Mocked" Test

**Trigger:** SRE wants to test if the agent handles "Database Down" errors correctly.
**Action:** SRE edits a Test Case. In "Mock Options," they select "Tool: PatientDB" -> "Return Error: 503 Service Unavailable."
**Run:** The agent runs. It tries to call the tool. The Simulator intercepts and throws the 503.
**Verification:** The Grader checks if the agent replied "System unavailable, please try again" (Pass) or crashed (Fail).

### Story C: The "Reasoning Trace" Grading

**Trigger:** A complex Math Agent must show its work.
**Requirement:** Expected Reasoning: ["Convert units", "Calculate BMI", "Compare to Threshold"].
**Execution:** Agent outputs: "Patient is obese."
**Grading:** The LLM Judge scans the CoT. It sees the agent skipped "Convert units."
**Result:** Score: 50% (Correct Answer, Invalid Process). Feedback: "Reasoning Step Missed."

---

## 6. Data Schema (Conceptual)

*   **TestCorpus**: id, project_id, name, version, created_by
*   **TestCase**:
    *   id, corpus_id
    *   inputs: { prompt: "...", files: ["s3://..."], context: {...} }
    *   expectations: { text: "...", schema_id: "...", reasoning: ["..."], tool_mocks: {...} }
*   **TestRun**: id, corpus_version, agent_draft_version, status (Running/Done)
*   **TestResult**:
    *   id, run_id, case_id
    *   actual_output: { text: "...", trace: "..." }
    *   scores: { faithfulness: 0.9, schema: 1.0, latency: 400ms }
    *   pass: boolean
