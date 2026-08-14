"""Prompt for the coverage evaluator.

Coverage runs from INPUT + CONTEXT to OUTPUT: it measures how completely the
generated OUTPUT represents the task-relevant information in the authoritative
CONTEXT. The judge uses INPUT to scope which context is relevant, derives atomic
task-relevant requirements, and classifies each against the OUTPUT. The judge
never produces a numeric score; Python aggregates the classifications.

The prompt is a Phoenix-style message list (separate system and user messages).
The system message holds only the evaluation rubric; the user message holds only
the data being evaluated. This Python file is the source of truth.

Prompt versioning: V1 kept the "material source item" framing; V2 switches to
recall-style atomic requirement decomposition. ``COVERAGE_PROMPT`` points at V2.
"""

from __future__ import annotations

# ============================================================================
# V1 (historical) - material source item framing. Kept for benchmarking.
# ============================================================================
_COVERAGE_SYSTEM_V1 = """\
You are a strict evaluator measuring COVERAGE of a generated output against an
authoritative context, for a specific requested task.

DEFINITION
Coverage measures how much of the material information in the authoritative
CONTEXT that is relevant to satisfying the requested task (the INPUT) is
represented in the generated OUTPUT.

Coverage does NOT measure hallucination, unsupported additions, factual
grounding, writing quality, style, formatting, or correctness against external
knowledge. Those belong to other metrics such as faithfulness.

Classify each material source item as "covered", "partial", or "missing".
Return one entry per material source item with a brief reason. Do NOT return any
numeric score, percentage, or weighting. Python computes the score.\
"""

_COVERAGE_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[INPUT]
{input}

[CONTEXT]
{context}

[OUTPUT]
{output}

[END DATA]"""

COVERAGE_PROMPT_V1 = [
    {"role": "system", "content": _COVERAGE_SYSTEM_V1},
    {"role": "user", "content": _COVERAGE_USER_TEMPLATE_V1},
]

COVERAGE_SCHEMA_V1 = {
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
                    "reason": {"type": "string"},
                },
                "required": ["source_item", "status"],
            },
        }
    },
    "required": ["items"],
}


# ============================================================================
# V2 (current) - recall-style atomic requirement decomposition.
# ============================================================================
_COVERAGE_SYSTEM_V2 = """\
You are a strict, auditable evaluator measuring COVERAGE of a generated output
against an authoritative context, for a specific requested task.

DEFINITION
Coverage answers: "How much of the task-relevant information in the supplied
CONTEXT is represented in the generated OUTPUT?" It measures completeness
(omissions), not correctness of additions.

Coverage does NOT measure any of the following (they belong to other metrics
such as faithfulness):
- hallucination or unsupported additions;
- factual grounding;
- writing quality, style, or formatting;
- correctness against external knowledge.
If the OUTPUT adds unsupported information, do NOT penalize it here; that is
faithfulness's job. Coverage is only about whether the required source
information is present.

STEP 1 - SCOPE THE TASK
1. Read the INPUT first and understand exactly what task was requested.
2. Determine which information in the CONTEXT is relevant to satisfying that
   task. Ignore context that is irrelevant to the task (e.g. employee names,
   office locations, IDs, formatting/metadata, unrelated history). Do not count
   irrelevant context as required coverage.

STEP 2 - DERIVE ATOMIC REQUIREMENTS
From the task-relevant context, derive atomic requirements/claims:
- Split a compound statement when it contains independently coverable semantic
  requirements. For example, "Reduce onboarding time by 25% and automate manual
  verification." becomes two requirements: "Reduce onboarding time by 25." and
  "Automate manual verification.".
- Preserve important qualifiers, quantities, constraints, entities, dates,
  thresholds, and conditions as part of the requirement. Examples that must stay
  attached: "25%", "real-time", "within 3 business days", "only administrators",
  "at least 95% accuracy", "before deployment".
- Do NOT fragment a sentence into meaningless lexical pieces (e.g. "reduce",
  "onboarding", "time").
- Do NOT duplicate semantically equivalent requirements; repeated context yields
  one requirement, not several.
- Do NOT invent requirements that are not supported by the CONTEXT.

STEP 3 - CLASSIFY EACH REQUIREMENT AGAINST THE OUTPUT
Classify each requirement as exactly one of:
- "covered": the OUTPUT represents the full semantic requirement, including
  material qualifiers. Exact wording is not required; semantically equivalent
  paraphrases count as covered. Example: requirement "Automate manual identity
  verification." with output "Identity verification will be automated." is
  covered.
- "partial": the OUTPUT captures a meaningful portion but omits or weakens a
  material part. Example: requirement "Reduce manual verification effort by 40%."
  with output "Reduce manual verification effort." is partial (the 40% target is
  missing). Prefer better atomic decomposition over using partial; do not mark
  partial merely because wording differs.
- "missing": the requirement is not meaningfully represented in the OUTPUT.

Provide a short, concise reason for every classification.

OUTPUT FORMAT
Return only the structured schema: one entry per requirement, each with the
requirement text, its status, and a reason. Do NOT return any aggregate score,
percentage, numeric coverage estimate, confidence, or weighting. Python computes
the final score deterministically from your classifications.\
"""

_COVERAGE_USER_TEMPLATE_V2 = """\
[BEGIN DATA]

[INPUT]
{input}

[CONTEXT]
{context}

[OUTPUT]
{output}

[END DATA]"""

COVERAGE_PROMPT_V2 = [
    {"role": "system", "content": _COVERAGE_SYSTEM_V2},
    {"role": "user", "content": _COVERAGE_USER_TEMPLATE_V2},
]

COVERAGE_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["covered", "partial", "missing"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["requirement", "status", "reason"],
            },
        }
    },
    "required": ["requirements"],
}


# Current prompt/schema used by the evaluator. Point these at a new version to
# benchmark without silently changing the metric definition.
COVERAGE_PROMPT = COVERAGE_PROMPT_V2
COVERAGE_SCHEMA = COVERAGE_SCHEMA_V2


def render_coverage_prompt(
    input_text: str,
    context: str,
    output: str,
) -> list[dict[str, str]]:
    """Renders the current coverage prompt into a fresh message list.

    A new list of message dicts is built on every call; the module-level
    ``COVERAGE_PROMPT`` template is never mutated. Only the user message's data
    placeholders are filled; the system rubric is passed through unchanged.

    Args:
        input_text: The requested task (``EvaluationCase.input``).
        context: The authoritative source information.
        output: The generated content being evaluated.

    Returns:
        A list of ``{"role", "content"}`` message dicts ready for the judge.
    """
    rendered: list[dict[str, str]] = []
    for message in COVERAGE_PROMPT:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(
                input=input_text,
                context=context,
                output=output,
            )
        rendered.append({"role": message["role"], "content": content})
    return rendered
