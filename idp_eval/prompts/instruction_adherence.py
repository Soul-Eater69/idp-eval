"""Prompt and strict schemas for one-call instruction adherence."""

from __future__ import annotations


_INSTRUCTION_ADHERENCE_CONTRACT_V1 = """\
Evaluate whether the generated OUTPUT follows the explicit INSTRUCTIONS.

In one response:
1. identify all materially distinct, independently checkable output instructions;
   and
2. classify each identified instruction as exactly "followed" or "violated".

INSTRUCTION RULES
- An instruction is an obligation, required capability, constraint, prohibition,
  measurable target, output format, ordering rule, language/style requirement,
  or expected outcome that the OUTPUT must satisfy.
- Split compound text when its constraints can independently be followed or
  violated. Do not split into meaningless lexical fragments.
- Do not emit both an umbrella instruction and child instructions that already
  fully represent it. Do not emit semantic duplicates or synonym restatements.
- Preserve material qualifiers, including exact/at-least/maximum counts, ranges,
  word limits, percentages, timing, actors, scope, conditions, ordering,
  prohibitions, and mandatory/optional wording.
- Background, rationale, examples, explanatory text, source data, headings, and
  metadata are not instructions unless they impose an output requirement.
- Never invent an instruction that is not present in INSTRUCTIONS.

JUDGMENT RULES
- "followed" means the complete instruction is satisfied by the full OUTPUT.
- "violated" means any material part of the instruction is not satisfied.
- Universal qualifiers such as "each", "every", and "all" are followed only
  when every applicable output item satisfies the instruction.
- Exact counts, minimums, maximums, ranges, limits, required fields, output
  structure, ordering, language, style, and prohibitions must be judged using
  their stated semantics against the complete rendered OUTPUT.
- A prohibition is followed only when the prohibited content is absent.
- Use semantic meaning rather than exact wording where the instruction permits it.

Do not return IDs, item scores, an aggregate score, a percentage, confidence,
weights, partial statuses, or chain-of-thought reasoning. Python assigns IDs and
calculates all numeric scores."""

_INSTRUCTION_ADHERENCE_SYSTEM_COMPACT_V1 = (
    _INSTRUCTION_ADHERENCE_CONTRACT_V1
    + "\n\nReturn only instruction and status for each item. Do not return "
    "reasons or other prose."
)
_INSTRUCTION_ADHERENCE_SYSTEM_VERBOSE_V1 = (
    _INSTRUCTION_ADHERENCE_CONTRACT_V1
    + "\n\nAlso return a required reason string for every item. For followed "
    "items use an empty string. For violated items give a concise, non-empty "
    "description of the observable mismatch. Do not return other prose."
)

_INSTRUCTION_ADHERENCE_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[INSTRUCTIONS]
{instructions}

[OUTPUT]
{output}

[END DATA]"""

INSTRUCTION_ADHERENCE_PROMPT_COMPACT_V1 = [
    {"role": "system", "content": _INSTRUCTION_ADHERENCE_SYSTEM_COMPACT_V1},
    {"role": "user", "content": _INSTRUCTION_ADHERENCE_USER_TEMPLATE_V1},
]
INSTRUCTION_ADHERENCE_PROMPT_VERBOSE_V1 = [
    {"role": "system", "content": _INSTRUCTION_ADHERENCE_SYSTEM_VERBOSE_V1},
    {"role": "user", "content": _INSTRUCTION_ADHERENCE_USER_TEMPLATE_V1},
]


def _instruction_item_schema(*, include_reason: bool) -> dict:
    properties = {
        "instruction": {"type": "string"},
        "status": {"type": "string", "enum": ["followed", "violated"]},
    }
    required = ["instruction", "status"]
    if include_reason:
        properties["reason"] = {"type": "string"}
        required.append("reason")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _instruction_schema(*, include_reason: bool) -> dict:
    return {
        "type": "object",
        "properties": {
            "instructions": {
                "type": "array",
                "items": _instruction_item_schema(include_reason=include_reason),
            }
        },
        "required": ["instructions"],
        "additionalProperties": False,
    }


INSTRUCTION_ADHERENCE_SCHEMA_COMPACT = _instruction_schema(include_reason=False)
INSTRUCTION_ADHERENCE_SCHEMA_VERBOSE = _instruction_schema(include_reason=True)


def render_instruction_adherence_prompt(
    instructions: str, output: str, verbose: bool = False
) -> list[dict[str, str]]:
    """Renders a fresh one-call instruction-adherence prompt."""
    template = (
        INSTRUCTION_ADHERENCE_PROMPT_VERBOSE_V1
        if verbose
        else INSTRUCTION_ADHERENCE_PROMPT_COMPACT_V1
    )
    rendered: list[dict[str, str]] = []
    for message in template:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(instructions=instructions, output=output)
        rendered.append({"role": message["role"], "content": content})
    return rendered
