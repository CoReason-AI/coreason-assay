# Coreason Manifest v0.10.0 Proposal

## Rationale
During the integration of `coreason-assay` with `coreason-manifest` v0.9.0, we identified that the `AgentRuntimeConfig` schema is missing the `system_prompt` field, which was expected based on migration instructions ("Old: agent.topology.system_prompt -> New: agent.config.system_prompt"). This prevents Assay graders from validating the agent's core instructions.

## Proposed Changes

### 1. Restore `system_prompt` to `AgentRuntimeConfig`
Add an optional or required `system_prompt` field to `AgentRuntimeConfig` to serve as the global instruction set for the agent.

```python
class AgentRuntimeConfig(BaseModel):
    system_prompt: str = Field(..., description="The global system prompt/instruction for the agent.")
    # ... existing fields (nodes, edges, entry_point, llm_config)
```

### 2. (Alternative) Move `system_prompt` to `ModelConfig`
If the intent is to couple instructions with the model, add it to `ModelConfig`.

```python
class ModelConfig(BaseModel):
    model: str
    temperature: float
    system_prompt: Optional[str] = None
```

### 3. Usage in Assay
With this change, `coreason-assay` can implement "Glass Box" grading that verifies if the agent's behavior aligns with its `system_prompt`.

Example Grader Logic:
```python
def grade(self, result, agent):
    instruction = agent.config.system_prompt
    # Check if result.text follows instruction...
```
