"""Stage 2 prompt for coverage: binary classification of fixed requirements.

Classification receives the already-extracted REQUIREMENTS (each with a stable
id) and the generated OUTPUT. The requirement list is authoritative: the judge
must classify exactly those requirements and must NOT add, remove, merge, split,
or rewrite them. For each requirement it returns two booleans; Python derives the
covered/partial/missing status and all numeric scores.

The task (INPUT) is included only to clarify requirement semantics. The full
CONTEXT is deliberately NOT included here — extraction already used it.
"""

from __future__ import annotations

_COVERAGE_CLASSIFY_SYSTEM_V1 = """\
You classify how completely a generated OUTPUT represents each of a fixed list of
REQUIREMENTS. The REQUIREMENTS list is authoritative and complete for this
evaluation.

DO NOT add, remove, merge, split, or rewrite requirements. Classify exactly the
requirements provided, returning each requirement's ``id`` unchanged, exactly
once.

For each requirement, return two booleans:
- "meaningfully_present": true if any meaningful semantic part of the requirement
  is represented in the OUTPUT; false if the OUTPUT does not represent it in a
  meaningful way.
- "fully_present": true if the full material meaning of the requirement,
  including important qualifiers (numbers, percentages, thresholds, conditions,
  words like "real-time"/"only"/"within"), is represented in the OUTPUT.

Judge meaning, not wording. Semantic paraphrases count; exact wording is not
required. Examples:
- Requirement "Automate identity verification." with OUTPUT "Identity checks will
  be performed automatically." -> meaningfully_present=true, fully_present=true.
- Requirement "Reduce verification effort by 40%." with OUTPUT "Reduce
  verification effort." -> meaningfully_present=true, fully_present=false (the 40%
  qualifier is missing).
- Requirement "Reduce abandoned registrations." with OUTPUT containing no
  equivalent idea -> meaningfully_present=false, fully_present=false.

CONSISTENCY: if meaningfully_present is false, fully_present must also be false.
Never return meaningfully_present=false with fully_present=true.

This is a completeness (recall) judgment. Do NOT penalize the OUTPUT for adding
information that is not in the requirements; unsupported additions are out of
scope here. Provide a short reason for each requirement. Do NOT return any score,
percentage, or confidence — only the two booleans and a reason per requirement.\
"""

_COVERAGE_CLASSIFY_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[INPUT]
{input}

[REQUIREMENTS]
{requirements}

[OUTPUT]
{output}

[END DATA]"""

COVERAGE_CLASSIFY_PROMPT_V1 = [
    {"role": "system", "content": _COVERAGE_CLASSIFY_SYSTEM_V1},
    {"role": "user", "content": _COVERAGE_CLASSIFY_USER_TEMPLATE_V1},
]

COVERAGE_CLASSIFY_PROMPT = COVERAGE_CLASSIFY_PROMPT_V1

COVERAGE_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "meaningfully_present": {"type": "boolean"},
                    "fully_present": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "id",
                    "meaningfully_present",
                    "fully_present",
                    "reason",
                ],
            },
        }
    },
    "required": ["requirements"],
}


def render_coverage_classify_prompt(
    input_text: str,
    requirements_json: str,
    output: str,
) -> list[dict[str, str]]:
    """Renders the Stage 2 classification prompt into a fresh message list.

    A new list is built each call; the module-level template is never mutated.
    Only the user message's placeholders are filled. The full context is
    deliberately NOT part of this prompt.

    Args:
        input_text: The requested task, for semantic clarification only.
        requirements_json: The fixed extracted requirements serialized as a JSON
            array of ``{"id", "requirement"}`` objects.
        output: The generated content being evaluated.

    Returns:
        A list of ``{"role", "content"}`` message dicts.
    """
    rendered: list[dict[str, str]] = []
    for message in COVERAGE_CLASSIFY_PROMPT:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(
                input=input_text,
                requirements=requirements_json,
                output=output,
            )
        rendered.append({"role": message["role"], "content": content})
    return rendered
