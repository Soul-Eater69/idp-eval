"""Prompt for the hallucination evaluator.

Versioned so that ``hallucination_v1`` / ``hallucination_v2`` can be benchmarked
against the same golden dataset later.
"""

HALLUCINATION_PROMPT_V1 = """\
You are evaluating whether the claims made in a generated OUTPUT are supported
by the authoritative CONTEXT. The INPUT describes what the model was asked to do
and is provided only for background.

Instructions:
1. Break the OUTPUT into its distinct, atomic factual claims.
2. For each claim, decide its status relative to the CONTEXT only. Do not use
   outside knowledge.
   - "supported": the CONTEXT directly states or clearly implies the claim.
   - "unsupported": the CONTEXT neither states nor implies the claim.
   - "contradicted": the CONTEXT states something that conflicts with the claim.
3. Do not assign a numeric score. Only classify claims.

INPUT:
{input}

CONTEXT:
{context}

OUTPUT:
{output}
"""

# Current prompt used by the evaluator.
HALLUCINATION_PROMPT = HALLUCINATION_PROMPT_V1

# JSON schema for the structured judge response.
HALLUCINATION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["supported", "unsupported", "contradicted"],
                    },
                },
                "required": ["claim", "status"],
            },
        }
    },
    "required": ["claims"],
}
