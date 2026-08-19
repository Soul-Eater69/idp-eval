"""Stage 1 prompt for extracting explicit, checkable instructions.

Extraction receives only ``EvaluationCase.instructions``. It does not receive
the task input, context, or generated output, and it does not grade adherence.
"""

from __future__ import annotations

_INSTRUCTION_ADHERENCE_EXTRACT_SYSTEM_V1 = """\
You extract the distinct instructions that a generated output is expected to
follow. You receive only the explicit INSTRUCTIONS. Do not grade an output and do
not infer requirements from any other source.

WHAT COUNTS AS AN INSTRUCTION
Extract checkable required actions or content, prohibited content, formatting or
count constraints, structural or ordering requirements, style or language
requirements, scope restrictions, and mandatory conditions.

GRANULARITY
- Split an instruction when two parts can independently be followed or violated.
  For example, "Return JSON and do not mention internal IDs." becomes "Return
  JSON." and "Do not mention internal IDs.".
- Keep parts together when they define one coherent requirement whose pieces do
  not make sense to score independently. For example, "Return JSON with title
  and summary fields." may remain one schema-format instruction.
- Do not optimize for maximally atomic decomposition.
- Do not emit both an umbrella instruction and child instructions that already
  fully represent it.
- Do not emit semantic duplicates or equivalent restatements.
- Do not split instructions into meaningless lexical fragments.

QUALIFIERS
Preserve every qualifier that affects adherence, including exact counts, numeric
limits, language, ordering, scope, actors, timing, conditions, prohibitions, and
mandatory or optional wording. For example, preserve "exactly 3", "Spanish",
"no more than 200 words", "descending revenue", and "only active users".

GROUNDING
Do not turn explanatory text, examples, rationale, background, or user data into
instructions unless that text actually imposes a requirement on the output. Do
not invent unstated instructions.

OUTPUT
Return only the extracted instruction strings in the structured schema. Do not
return IDs, adherence judgments, scores, percentages, or confidence values.\
"""

_INSTRUCTION_ADHERENCE_EXTRACT_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[INSTRUCTIONS]
{instructions}

[END DATA]"""

INSTRUCTION_ADHERENCE_EXTRACT_PROMPT_V1 = [
    {"role": "system", "content": _INSTRUCTION_ADHERENCE_EXTRACT_SYSTEM_V1},
    {"role": "user", "content": _INSTRUCTION_ADHERENCE_EXTRACT_USER_TEMPLATE_V1},
]

INSTRUCTION_ADHERENCE_EXTRACT_PROMPT = INSTRUCTION_ADHERENCE_EXTRACT_PROMPT_V1

INSTRUCTION_ADHERENCE_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"instruction": {"type": "string"}},
                "required": ["instruction"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["instructions"],
    "additionalProperties": False,
}


def render_instruction_adherence_extract_prompt(
    instructions: str,
) -> list[dict[str, str]]:
    """Renders the extraction prompt into a fresh Phoenix-style message list."""
    rendered: list[dict[str, str]] = []
    for message in INSTRUCTION_ADHERENCE_EXTRACT_PROMPT:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(instructions=instructions)
        rendered.append({"role": message["role"], "content": content})
    return rendered
