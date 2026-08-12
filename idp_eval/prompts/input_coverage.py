"""Prompt for the input coverage evaluator.

Coverage runs in the opposite direction to hallucination: from CONTEXT to
OUTPUT. The judge first identifies the important source information, then decides
how well each item is represented in the output.
"""

INPUT_COVERAGE_PROMPT_V1 = """\
You are evaluating how much of the important information in the authoritative
CONTEXT is represented in the generated OUTPUT. The INPUT describes what the
model was asked to do and is provided only for background.

Instructions:
1. Identify the important, distinct pieces of source information in the CONTEXT.
   Ignore trivia and boilerplate; focus on information a faithful output should
   preserve.
2. For each source item, decide how well it is represented in the OUTPUT.
   Judge meaning, not exact wording.
   - "covered": the OUTPUT fully represents the item.
   - "partial": the OUTPUT represents the item only partially or vaguely.
   - "missing": the OUTPUT does not represent the item at all.
3. Do not assign a numeric score. Only classify items.

INPUT:
{input}

CONTEXT:
{context}

OUTPUT:
{output}
"""

# Current prompt used by the evaluator.
INPUT_COVERAGE_PROMPT = INPUT_COVERAGE_PROMPT_V1

# JSON schema for the structured judge response.
INPUT_COVERAGE_SCHEMA = {
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
