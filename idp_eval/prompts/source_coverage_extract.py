"""Stage 1 prompt for source coverage: whole-source item extraction.

Extraction sees ONLY the CONTEXT (the source/reference content). It must NOT see
any task/input, instructions, or generated output, and must NOT grade anything —
it only derives the important information a faithful representation of the source
should preserve.

Phoenix-style message list: the system message holds the rubric; the user message
holds only the source data. This file is the stable prompt contract for
``SourceCoverageEvaluator`` Stage 1.
"""

from __future__ import annotations

_SOURCE_COVERAGE_EXTRACT_SYSTEM_V1 = """\
You extract the important pieces of information from a source that a faithful
representation of that source should preserve. You are given only the SOURCE
(CONTEXT). You are NOT given any task, instructions, or generated output, and you
must NOT grade, score, or compare anything.

WHAT TO PRODUCE
Extract the materially important information items in the SOURCE that a good
representation should preserve.

WHAT COUNTS
An item is materially important if omitting it would change the meaning, an
obligation, a decision, a constraint, a prohibition, a dependency, a measurable
target, an actor, or an expected outcome. Extract only what the SOURCE states.
There is no target or maximum number of items; do not drop a genuinely important
item just because the source is large.

RULES
- Preserve material qualifiers: numbers, percentages, limits, thresholds,
  conditions, timing, actors/roles, prohibitions, and dependencies.
- Consolidate closely related statements that express one underlying item. Merge
  an obligation with its own qualifiers, limits, conditions, thresholds, or
  dependent constraints into a single item. Do NOT split an item merely because
  it carries several qualifiers. This is consolidation, NOT summarization: never
  drop a materially distinct item to shorten the list. There is no target and no
  maximum count.
- Keep items SEPARATE only when they are independently satisfiable. Example:
  "Reduce onboarding time by 25%." and "Support French localization." are two
  items; but "Retain the existing identity provider while preserving SSO
  continuity and avoiding user re-registration." is ONE item even if stated
  across several sentences, because those parts are one underlying obligation.
- Do NOT fragment one item into meaningless lexical pieces.
- Do NOT emit semantic duplicates or synonym restatements of an already-listed
  item, and do NOT emit a broad umbrella item alongside the specific items that
  already represent it.
- Exclude repetition, examples, background commentary, and trivial or incidental
  filler.
- Never invent an item that is not supported by the SOURCE.

OUTPUT
Return only the list of source item strings in the structured schema. Do NOT
return any status, score, percentage, comparison, rationale, evidence, or
confidence.\
"""

_SOURCE_COVERAGE_EXTRACT_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[CONTEXT]
{context}

[END DATA]"""

SOURCE_COVERAGE_EXTRACT_PROMPT_V1 = [
    {"role": "system", "content": _SOURCE_COVERAGE_EXTRACT_SYSTEM_V1},
    {"role": "user", "content": _SOURCE_COVERAGE_EXTRACT_USER_TEMPLATE_V1},
]

SOURCE_COVERAGE_EXTRACT_PROMPT = SOURCE_COVERAGE_EXTRACT_PROMPT_V1

SOURCE_COVERAGE_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "source_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source_item": {"type": "string"}},
                "required": ["source_item"],
            },
        }
    },
    "required": ["source_items"],
}


def render_source_coverage_extract_prompt(context: str) -> list[dict[str, str]]:
    """Renders the source-coverage extraction prompt into a fresh message list.

    A new list is built each call; the module-level template is never mutated.
    Only the user message's ``{context}`` placeholder is filled. No task/input,
    instructions, or output are part of this prompt.

    Args:
        context: The source/reference content (``EvaluationCase.context``).

    Returns:
        A list of ``{"role", "content"}`` message dicts.
    """
    rendered: list[dict[str, str]] = []
    for message in SOURCE_COVERAGE_EXTRACT_PROMPT:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(context=context)
        rendered.append({"role": message["role"], "content": content})
    return rendered
