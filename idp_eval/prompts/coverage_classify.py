"""Stage 2 prompt for coverage: binary classification of fixed requirements.

Classification receives the already-extracted REQUIREMENTS (each with a stable
id) and the generated OUTPUT. The requirement list is authoritative: the judge
must classify exactly those requirements and must NOT add, remove, merge, split,
or rewrite them. For each requirement it returns two booleans; Python derives the
covered/partial/missing status and all numeric scores.

Two variants share one user template and one binary contract:

- **compact** (default): returns only ``id`` + the two booleans per requirement.
  This is the smallest output that still supports deterministic scoring and keeps
  Stage-2 responses small (no per-item prose), which helps reduce the risk of a
  classification request exceeding the gateway timeout (a mitigation, not a
  guarantee).
- **verbose**: additionally returns a short ``reason`` for requirements that are
  not fully represented (partial/missing). Covered items need no reason.

Both variants classify identically; verbosity changes only diagnostic prose, not
the booleans, statuses, or score. The overall ``result.explanation`` is generated
deterministically in Python, so the judge is never asked for it here.

The task (INPUT) is included only to clarify requirement semantics. The full
CONTEXT is deliberately NOT included here — extraction already used it. Output
uses stable ids only; requirement text is never echoed back.
"""

from __future__ import annotations

_CLASSIFY_CONTRACT = """\
You classify how completely a generated OUTPUT represents each of a fixed list of
REQUIREMENTS. The REQUIREMENTS list is authoritative and complete for this
evaluation.

DO NOT add, remove, merge, split, or rewrite requirements. Classify exactly the
requirements provided, returning each requirement's ``id`` unchanged, exactly
once. Refer to requirements only by ``id``; do not echo the requirement text.

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
scope here."""

_COVERAGE_CLASSIFY_SYSTEM_COMPACT_V1 = (
    _CLASSIFY_CONTRACT
    + """

Return ONLY each requirement's ``id`` and the two booleans. Do NOT return any
reason, explanation, score, percentage, or confidence."""
)

_COVERAGE_CLASSIFY_SYSTEM_VERBOSE_V1 = (
    _CLASSIFY_CONTRACT
    + """

Return each requirement's ``id`` and the two booleans. For any requirement that
is not fully represented (meaningfully_present=false, or fully_present=false),
also return a brief one-sentence ``reason`` naming what is missing. Fully
represented requirements need no reason. Do NOT return any score, percentage, or
confidence."""
)

_COVERAGE_CLASSIFY_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[INPUT]
{input}

[REQUIREMENTS]
{requirements}

[OUTPUT]
{output}

[END DATA]"""

COVERAGE_CLASSIFY_PROMPT_COMPACT_V1 = [
    {"role": "system", "content": _COVERAGE_CLASSIFY_SYSTEM_COMPACT_V1},
    {"role": "user", "content": _COVERAGE_CLASSIFY_USER_TEMPLATE_V1},
]

COVERAGE_CLASSIFY_PROMPT_VERBOSE_V1 = [
    {"role": "system", "content": _COVERAGE_CLASSIFY_SYSTEM_VERBOSE_V1},
    {"role": "user", "content": _COVERAGE_CLASSIFY_USER_TEMPLATE_V1},
]

# The default (compact) prompt is the stable public name.
COVERAGE_CLASSIFY_PROMPT_V1 = COVERAGE_CLASSIFY_PROMPT_COMPACT_V1
COVERAGE_CLASSIFY_PROMPT = COVERAGE_CLASSIFY_PROMPT_V1


def _classify_item_schema(*, include_reason: bool) -> dict:
    """Builds the per-item classification schema (with or without ``reason``)."""
    properties: dict = {
        "id": {"type": "string"},
        "meaningfully_present": {"type": "boolean"},
        "fully_present": {"type": "boolean"},
    }
    required = ["id", "meaningfully_present", "fully_present"]
    if include_reason:
        # Optional: only partial/missing items carry a reason.
        properties["reason"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


COVERAGE_CLASSIFY_SCHEMA_COMPACT = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": _classify_item_schema(include_reason=False),
        }
    },
    "required": ["requirements"],
}

COVERAGE_CLASSIFY_SCHEMA_VERBOSE = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": _classify_item_schema(include_reason=True),
        }
    },
    "required": ["requirements"],
}

# Default schema is the compact one.
COVERAGE_CLASSIFY_SCHEMA = COVERAGE_CLASSIFY_SCHEMA_COMPACT


def render_coverage_classify_prompt(
    input_text: str,
    requirements_json: str,
    output: str,
    verbose: bool = False,
) -> list[dict[str, str]]:
    """Renders the Stage 2 classification prompt into a fresh message list.

    A new list is built each call; the module-level templates are never mutated.
    Only the user message's placeholders are filled. The full context is
    deliberately NOT part of this prompt.

    Args:
        input_text: The requested task, for semantic clarification only.
        requirements_json: The fixed extracted requirements serialized as a JSON
            array of ``{"id", "requirement"}`` objects.
        output: The generated content being evaluated.
        verbose: When ``True``, use the variant that also requests a short reason
            for partial/missing requirements. Booleans/scores are identical
            either way.

    Returns:
        A list of ``{"role", "content"}`` message dicts.
    """
    prompt = (
        COVERAGE_CLASSIFY_PROMPT_VERBOSE_V1
        if verbose
        else COVERAGE_CLASSIFY_PROMPT_COMPACT_V1
    )
    rendered: list[dict[str, str]] = []
    for message in prompt:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(
                input=input_text,
                requirements=requirements_json,
                output=output,
            )
        rendered.append({"role": message["role"], "content": content})
    return rendered
