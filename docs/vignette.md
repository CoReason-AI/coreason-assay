# Vignette: Testing a Medical Protocol Agent

This vignette demonstrates how to use `coreason-assay` to quality control a mission-critical agent: the **Medical Protocol Analyzer**.

## Scenario

**The Agent:** A tool designed to assist doctors by analyzing patient data against complex clinical protocols (PDFs) to recommend treatments.
**The Risk:** If the agent hallucinates a treatment or misses a contraindication, it could endanger a patient.
**The Solution:** We create a "Cognitive Assay" to rigorously test the agent's reasoning process, not just its final answer.

## 1. Defining the Test Case

We want to test if the agent correctly identifies a contraindication for "Drug A" when the patient has "Condition B", based on the provided "Protocol Alpha" document.

### The Input (Golden Data)

We prepare a ZIP file `protocol_tests.zip` containing:
1.  `protocols/protocol_alpha.pdf`: The official medical document.
2.  `manifest.csv`: The test definition.

**manifest.csv Content:**

| prompt | files | context | expected_text | expected_reasoning | forbidden_content |
| :--- | :--- | :--- | :--- | :--- | :--- |
| "Patient has Condition B. Can I prescribe Drug A?" | `["protocols/protocol_alpha.pdf"]` | `{"role": "Doctor"}` | "No, Drug A is contraindicated." | `["Check Protocol Alpha", "Identify Contraindications section", "Match Condition B to Drug A", "Conclude contraindicated"]` | `["Yes", "Safe to prescribe"]` |

## 2. Uploading the Corpus

We upload the test suite to the system using the CLI:

```bash
poetry run coreason-assay upload protocol_tests.zip \
    --name "Contraindication Check" \
    --version "1.0"
```

*The system parses the CSV, validates the paths, and stores the test case.*

## 3. Running the Simulation

The SRE triggers a test run against the `draft-v2` version of the agent.

*   **Simulator Action:**
    *   Spins up a sandbox.
    *   Injects the PDF and the prompt.
    *   Mocks the "Patient Database" tool to ensure deterministic behavior (if the agent tries to look up the patient).

## 4. The Report Card

After the run, `coreason-assay` generates a report card. Let's look at two possible outcomes.

### Scenario A: PASS ✅

**Agent Output:**
> "Based on Protocol Alpha, section 4.2, Drug A is contraindicated for patients with Condition B. Therefore, you should not prescribe it."

**Grading:**
*   **Faithfulness (LLM Judge):** Pass. The agent cited the correct document.
*   **Reasoning Alignment (LLM Judge):** Pass. The agent explicitly mentioned checking the protocol and finding the contraindication.
*   **Forbidden Content:** Pass. Did not say "Yes".

**Result:**
*   **Score:** 100%
*   **Status:** PASSED

### Scenario B: FAIL ❌ (Right Answer, Wrong Reason)

**Agent Output:**
> "No, don't prescribe it. I saw on a forum that it's bad."

**Grading:**
*   **Faithfulness (LLM Judge):** **FAIL**. The agent did not rely on the provided PDF.
*   **Reasoning Alignment:** **FAIL**. Missed the step "Check Protocol Alpha".
*   **Forbidden Content:** Pass.

**Result:**
*   **Score:** 33%
*   **Status:** **FAILED**
*   **Feedback:** "The agent arrived at the correct conclusion but failed to cite the provided protocol (Faithfulness Failure)."

## Conclusion

By enforcing **Reasoning Alignment**, we caught a dangerous behavior (hallucinating sources) that a simple text match ("No") would have missed. This is the power of Glass Box Grading.
