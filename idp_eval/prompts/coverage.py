"""Prompt for the coverage evaluator.

Coverage runs from CONTEXT to OUTPUT: it measures how much of the material
context information that is relevant to the requested task (INPUT) is
represented in the OUTPUT. The judge uses INPUT to understand the task, not as
throwaway background.
"""

COVERAGE_PROMPT_V1 = """\
You are evaluating COVERAGE: how much of the material information in the
authoritative CONTEXT that is relevant to satisfying the requested task (the
INPUT) is represented in the generated OUTPUT.

Instructions:
1. Read the INPUT first and understand what task was requested.
2. From the CONTEXT, identify only the material source information that is
   relevant to satisfying that task. Ignore irrelevant context, boilerplate,
   IDs, repeated information, and formatting.
3. For each relevant source item, decide how well it is represented in the
   OUTPUT. Judge meaning, not exact wording; a correct semantic paraphrase
   counts as covered.
   - "covered": the OUTPUT fully represents the item.
   - "partial": the OUTPUT represents the item only partially or vaguely.
   - "missing": the OUTPUT does not represent the item at all.
4. Do NOT penalize the OUTPUT here for adding unsupported information. Unsupported
   additions are evaluated by the faithfulness metric, not by coverage.
5. Do not assign a numeric score. Only classify items.

INPUT:
{input}

CONTEXT:
{context}

OUTPUT:
{output}
"""

# Current prompt used by the evaluator.
COVERAGE_PROMPT = COVERAGE_PROMPT_V1

# JSON schema for the structured judge response.
COVERAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_item": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["covered", "partial", "missing"],
                    },
                },
                "required": ["source_item", "status"],
            },
        }
    },
    "required": ["items"],
}
