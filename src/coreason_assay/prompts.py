# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_assay

REASONING_GRADER_PROMPT = """You are an expert evaluator of AI reasoning chains.
Your task is to verify if the actual execution trace of an AI agent contains specific required reasoning steps.

Required Reasoning Steps:
__REQUIRED_STEPS__

Actual Execution Trace:
__TRACE__

(Fallback) Actual Output Text:
__TEXT__

Instructions:
1. Analyze the trace (and text if trace is insufficient) to find evidence of each required step.
2. Return a JSON object with the following structure:
{
  "steps_analysis": [
    {"step": "step description", "found": true/false, "evidence": "quote or explanation"},
    ...
  ],
  "score": <float between 0.0 and 1.0 representing percentage of steps found>
}

Return ONLY the JSON.
"""

FAITHFULNESS_GRADER_PROMPT = """You are an expert fact-checker for AI assistants.
Your task is to verify if the AI's generated answer is supported by the provided Context.
Does the Answer hallucinate information not present in the Context, or contradict it?

Context:
__CONTEXT__

AI Answer:
__ANSWER__

Instructions:
1. Analyze the Answer against the Context.
2. Return a JSON object with the following structure:
{
  "faithful": true/false,
  "reasoning": "Explanation of why it is faithful or not. Cite specific contradictions if any.",
  "score": 1.0 (if faithful) or 0.0 (if not)
}

Return ONLY the JSON.
"""

TONE_GRADER_PROMPT = """You are an expert tone analyzer for AI assistants.
Your task is to verify if the AI's response matches the expected tone.

Expected Tone:
__TONE__

AI Response:
__RESPONSE__

Instructions:
1. Analyze the Response to see if it aligns with the Expected Tone.
2. Return a JSON object with the following structure:
{
  "matches_tone": true/false,
  "reasoning": "Explanation of why it matches or fails. Cite specific words or phrases.",
  "score": 1.0 (if matches) or 0.0 (if not)
}

Return ONLY the JSON.
"""
