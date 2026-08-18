"""One-call itemized prompt for whole-source coverage.

The judge receives the source ``CONTEXT`` and generated ``OUTPUT`` together. It
identifies materially important source items and classifies each item in one
structured response. This is faster than output-isolated two-stage extraction,
but the generated output can influence the denominator; task coverage retains
the two-stage architecture when denominator isolation is required.
"""

from __future__ import annotations

_SOURCE_COVERAGE_CONTRACT = """\
Evaluate how completely the generated OUTPUT represents the materially important
information in the SOURCE (CONTEXT).

In this single response:
1. identify every materially distinct important source item; and
2. classify how completely the OUTPUT represents each identified item.

SOURCE ITEM RULES
- Preserve every materially distinct important fact, obligation, decision,
  required capability, constraint, prohibition, dependency, measurable target,
  actor, and expected outcome.
- Preserve material qualifiers, including numbers, percentages, limits,
  thresholds, conditions, timing, actors/roles, prohibitions, and dependencies.
- Semantically consolidate repeated or dependent statements. Merge duplicate or
  equivalent statements and merge qualifiers into their underlying item.
- Consolidation is not summarization: never drop a materially distinct important
  item to shorten the list, and do not impose a maximum item count.
- Keep independently satisfiable or independently violatable items separate.
- Do not emit a broad umbrella item in addition to specific items that already
  fully represent it.
- Exclude repetition, examples, filler, incidental prose, and background that
  does not introduce materially important information.
- Never invent source content.

CLASSIFICATION RULES
For each source item return two booleans:
- "meaningfully_present": true when any meaningful semantic part of the item is
  represented in the OUTPUT; otherwise false.
- "fully_present": true only when the complete material meaning, including all
  important qualifiers, is represented in the OUTPUT.

Judge meaning rather than wording; semantic paraphrases count. If
meaningfully_present is false, fully_present must also be false. Unsupported
additions in the OUTPUT are out of scope for coverage and must not reduce the
classification.

Do not return an aggregate score, item score, percentage, confidence, or weight.
Python calculates all statuses and numeric scores."""

_SOURCE_COVERAGE_SYSTEM_COMPACT_V1 = (
    _SOURCE_COVERAGE_CONTRACT
    + """

Return only each ``source_item`` and its two booleans. Do not return reasons or
other prose."""
)

_SOURCE_COVERAGE_SYSTEM_VERBOSE_V1 = (
    _SOURCE_COVERAGE_CONTRACT
    + """

Also return a concise ``reason`` for each item that is partial or missing. Use an
empty reason for fully represented items. Do not return other prose."""
)

_SOURCE_COVERAGE_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[CONTEXT]
{context}

[OUTPUT]
{output}

[END DATA]"""

SOURCE_COVERAGE_PROMPT_COMPACT_V1 = [
    {"role": "system", "content": _SOURCE_COVERAGE_SYSTEM_COMPACT_V1},
    {"role": "user", "content": _SOURCE_COVERAGE_USER_TEMPLATE_V1},
]

SOURCE_COVERAGE_PROMPT_VERBOSE_V1 = [
    {"role": "system", "content": _SOURCE_COVERAGE_SYSTEM_VERBOSE_V1},
    {"role": "user", "content": _SOURCE_COVERAGE_USER_TEMPLATE_V1},
]

SOURCE_COVERAGE_PROMPT_V1 = SOURCE_COVERAGE_PROMPT_COMPACT_V1
SOURCE_COVERAGE_PROMPT = SOURCE_COVERAGE_PROMPT_V1


def _item_schema(*, include_reason: bool) -> dict:
    properties = {
        "source_item": {"type": "string"},
        "meaningfully_present": {"type": "boolean"},
        "fully_present": {"type": "boolean"},
    }
    required = ["source_item", "meaningfully_present", "fully_present"]
    if include_reason:
        properties["reason"] = {"type": "string"}
        required.append("reason")
    return {"type": "object", "properties": properties, "required": required}


SOURCE_COVERAGE_SCHEMA_COMPACT = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": _item_schema(include_reason=False),
        }
    },
    "required": ["items"],
}

SOURCE_COVERAGE_SCHEMA_VERBOSE = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": _item_schema(include_reason=True),
        }
    },
    "required": ["items"],
}

SOURCE_COVERAGE_SCHEMA = SOURCE_COVERAGE_SCHEMA_COMPACT


def render_source_coverage_prompt(
    context: str, output: str, verbose: bool = False
) -> list[dict[str, str]]:
    """Renders a fresh one-call source-coverage message list."""
    template = (
        SOURCE_COVERAGE_PROMPT_VERBOSE_V1
        if verbose
        else SOURCE_COVERAGE_PROMPT_COMPACT_V1
    )
    rendered: list[dict[str, str]] = []
    for message in template:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(context=context, output=output)
        rendered.append({"role": message["role"], "content": content})
    return rendered
