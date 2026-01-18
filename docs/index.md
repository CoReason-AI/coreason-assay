# Welcome to coreason-assay

**The Scientific Testing Engine for AI Agents.**

`coreason-assay` is the Quality Control (QC) laboratory of the CoReason platform. It provides a rigorous framework for evaluating the performance, safety, and alignment of AI agents before they are deployed to production.

## Why Cognitive Assays?

Just as a pharmaceutical drug must pass bio-assays before release, a CoReason agent must pass **Cognitive Assays**. We don't just check if the agent got the answer right; we verify *how* it got there.

*   **Faithfulness:** Did the agent stick to the provided source documents?
*   **Robustness:** Does it handle edge cases and bad data gracefully?
*   **Alignment:** Did it follow the required reasoning steps and ethical guidelines?

## Documentation Overview

*   [**Architecture**](architecture.md): Learn about the internal components (BEC Manager, Simulator, Grader).
*   [**Usage Guide**](usage.md): Instructions on how to install the package and use the CLI.
*   [**Vignette**](vignette.md): A step-by-step walkthrough of testing a Medical Protocol Analyzer agent.
*   [**Product Requirements**](prd.md): The original design specification and philosophy.

## Getting Started

To install the package and run your first test:

```sh
pip install coreason-assay
# OR
poetry add coreason-assay
```

Check out the [Usage Guide](usage.md) for detailed instructions.
