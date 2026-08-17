"""Stage 2 prompt for binary classification of fixed instructions.

Classification receives the Python-assigned instruction IDs and generated
output. It must return one followed/violated judgment per fixed instruction.
"""

from __future__ import annotations

_INSTRUCTION_ADHERENCE_CLASSIFY_SYSTEM_V1 = """\
You determine whether a generated OUTPUT follows each item in a fixed list of
INSTRUCTIONS. The list is authoritative for this evaluation.

Do not add, remove, merge, split, or rewrite instructions. Return every supplied
instruction ``id`` unchanged, exactly once.

Classify each instruction as exactly one of:
- "followed": the OUTPUT satisfies the full material meaning of the instruction,
  including relevant qualifiers.
- "violated": the OUTPUT fails to satisfy, contradicts, or omits the material
  requirement.

This is intentionally binary; there is no intermediate status or credit. Judge
meaning rather than exact wording, but inspect deterministic constraints such as
counts, valid formats, required fields, word limits, ordering, languages, and
prohibitions strictly.

Examples:
- "Return exactly 3 recommendations." is followed only when the output contains
  exactly three recommendations.
- "Do not mention pricing." is violated when the output mentions "$20/month".
- "Respond in Spanish." is violated when the output is primarily English.
- "Return valid JSON with title and summary fields." is violated by malformed
  JSON or a missing required field.

Give one concise reason per judgment. Return only IDs, statuses, and reasons in
the structured schema. Do not return numeric scores, percentages, aggregate
ratings, or confidence values.\
"""

_INSTRUCTION_ADHERENCE_CLASSIFY_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[INSTRUCTIONS]
{instructions}

[OUTPUT]
{output}

[END DATA]"""

INSTRUCTION_ADHERENCE_CLASSIFY_PROMPT_V1 = [
    {"role": "system", "content": _INSTRUCTION_ADHERENCE_CLASSIFY_SYSTEM_V1},
    {"role": "user", "content": _INSTRUCTION_ADHERENCE_CLASSIFY_USER_TEMPLATE_V1},
]

INSTRUCTION_ADHERENCE_CLASSIFY_PROMPT = INSTRUCTION_ADHERENCE_CLASSIFY_PROMPT_V1

INSTRUCTION_ADHERENCE_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["followed", "violated"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["id", "status", "reason"],
            },
        }
    },
    "required": ["answers"],
}


def render_instruction_adherence_classify_prompt(
    instructions_json: str,
    output: str,
) -> list[dict[str, str]]:
    """Renders the classification prompt into a fresh message list."""
    rendered: list[dict[str, str]] = []
    for message in INSTRUCTION_ADHERENCE_CLASSIFY_PROMPT:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(
                instructions=instructions_json,
                output=output,
            )
        rendered.append({"role": message["role"], "content": content})
    return rendered
