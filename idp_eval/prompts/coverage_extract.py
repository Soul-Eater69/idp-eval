"""Stage 1 prompt for coverage: task-relevant requirement extraction.

Extraction sees only INPUT (the task) and CONTEXT (the authoritative source). It
must NOT see the generated output and must NOT grade anything — it only derives
the atomic, task-relevant requirements the output should represent.

Phoenix-style message list: the system message holds the rubric; the user message
holds only the data. This file is the first stable prompt contract for the
two-stage coverage architecture.
"""

from __future__ import annotations

_COVERAGE_EXTRACT_SYSTEM_V1 = """\
You extract the materially distinct, independently checkable requirements that a
generated answer would need to represent in order to satisfy a user's task. You
are given the task (INPUT) and the authoritative source (CONTEXT). You are NOT
given any generated output, and you must NOT grade, score, or compare anything.

WHAT TO PRODUCE
Identify the task-relevant requirements from the CONTEXT that the answer should
represent to satisfy the task in INPUT. There is no target or maximum number of
requirements; do not drop a genuinely important requirement just because the
source is large.

RULES
- Read the task in INPUT and use it to decide what information in CONTEXT matters.
- Ignore context that is irrelevant to the requested task (e.g. unrelated
  history, employee names, office locations, IDs, formatting/metadata).
- Consolidate closely related statements that express one underlying
  requirement. Merge an obligation with its own qualifiers, limits, conditions,
  thresholds, or dependent constraints into a single requirement. Do NOT split a
  requirement merely because it carries several qualifiers. This is
  consolidation, NOT summarization: never drop a materially distinct requirement
  to shorten the list. There is no target and no maximum count.
- Keep obligations SEPARATE only when they can be satisfied independently.
  Example: "Reduce onboarding time by 25%." and "Support French localization."
  are two requirements (independently satisfiable). But "Retain the existing
  identity provider while preserving SSO continuity and avoiding user
  re-registration." is ONE requirement even if the source stated it across
  several sentences, because those parts are one underlying obligation.
- Preserve material qualifiers whose omission changes meaning: numbers,
  percentages, dates, thresholds, conditions, actors, prohibitions, dependencies,
  and words such as "real-time", "at least", "only", "before", "after", "within".
- Do NOT fragment a requirement into meaningless lexical pieces (e.g. "reduce",
  "onboarding", "time", "25%").
- Exclude repetition, examples, background commentary, incidental details, and
  non-material prose. Merge semantically equivalent statements.
- Never invent a requirement that is not grounded in the CONTEXT.
- Never refer to or assume any generated output; it is not provided.

OUTPUT
Return only the list of requirement strings in the structured schema. Do NOT
return any status, score, percentage, comparison, rationale, evidence, or
confidence.\
"""

_COVERAGE_EXTRACT_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[INPUT]
{input}

[CONTEXT]
{context}

[END DATA]"""

COVERAGE_EXTRACT_PROMPT_V1 = [
    {"role": "system", "content": _COVERAGE_EXTRACT_SYSTEM_V1},
    {"role": "user", "content": _COVERAGE_EXTRACT_USER_TEMPLATE_V1},
]

COVERAGE_EXTRACT_PROMPT = COVERAGE_EXTRACT_PROMPT_V1

COVERAGE_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"requirement": {"type": "string"}},
                "required": ["requirement"],
            },
        }
    },
    "required": ["requirements"],
}


def render_coverage_extract_prompt(
    input_text: str,
    context: str,
) -> list[dict[str, str]]:
    """Renders the Stage 1 extraction prompt into a fresh message list.

    A new list is built each call; the module-level template is never mutated.
    Only the user message's placeholders are filled. The output is deliberately
    NOT part of this prompt.

    Args:
        input_text: The requested task (``EvaluationCase.input``).
        context: The authoritative source information.

    Returns:
        A list of ``{"role", "content"}`` message dicts.
    """
    rendered: list[dict[str, str]] = []
    for message in COVERAGE_EXTRACT_PROMPT:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(input=input_text, context=context)
        rendered.append({"role": message["role"], "content": content})
    return rendered
