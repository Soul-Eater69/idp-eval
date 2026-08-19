"""Whole-source coverage prompts (shared by DAG and G-Eval modes).

``CoverageEvaluator`` answers "how much materially important information from the
source (context) is represented in the output?" Both modes use the SAME source-item
rubric so an A/B comparison isolates architecture, not rubric:

- **DAG** (default): Stage 1 sees the CONTEXT only and extracts the consolidated
  source items; Stage 2 classifies those fixed items against the OUTPUT. The
  output cannot influence the denominator.
- **G-Eval**: one call sees CONTEXT + OUTPUT and both identifies and classifies the
  source items together. Fewer calls / lower latency; the output is visible while
  identifying items.

Stage 1 (and the item-identification part of G-Eval) targets ~10 comprehensive,
independently checkable items via *semantic consolidation* — this is a soft target
reached by merging related meaning, never a destructive truncation. Python assigns
ids and computes all scores; the judge never returns a numeric coverage score.
"""

from __future__ import annotations

# Shared source-item rubric. ~10 is a semantic target reached by consolidation, not
# a hard cap and not "pick the top 10".
_COVERAGE_ITEM_RULES = """\
- Aim to represent the materially important source content in approximately 10
  comprehensive, independently checkable items. The ~10 is a semantic target
  reached by consolidation; it is NOT a hard maximum and NOT "pick the top 10".
- Consolidate related requirements, qualifiers, dependent constraints, and repeated
  statements when they belong to the same underlying obligation, fact, or outcome.
  Do not split one underlying requirement merely because it contains several
  qualifiers.
- Preserve every materially important fact, obligation, decision, required
  capability, constraint, prohibition, dependency, measurable target, actor, and
  expected outcome.
- Preserve material qualifiers: numbers, percentages, thresholds, timing, actors,
  conditions, limits, prohibitions, and dependencies.
- Keep independently satisfiable or independently violatable requirements separate.
- Do not create both a broad umbrella item and redundant child items.
- Exclude repetition, examples, filler, background commentary, and incidental prose.
- Never invent source information.
- If fewer than ~10 items suffice, return fewer. If more than ~10 are genuinely
  necessary to preserve materially distinct independent requirements, return more.
  Never drop a materially distinct important requirement to hit the target."""


# --- DAG Stage 1: context-only extraction -----------------------------------

_COVERAGE_EXTRACT_SYSTEM_V1 = (
    "Identify the materially important items in the SOURCE (CONTEXT) that a faithful "
    "representation should preserve. You are given ONLY the source; you are NOT given "
    "the generated answer and must not grade, classify, or compare anything.\n\n"
    "SOURCE ITEM RULES\n"
    + _COVERAGE_ITEM_RULES
    + "\n\nReturn only the source item strings."
)

_COVERAGE_EXTRACT_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

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
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source_item": {"type": "string"}},
                "required": ["source_item"],
            },
        }
    },
    "required": ["items"],
}


def render_coverage_extract_prompt(context: str) -> list[dict[str, str]]:
    """Renders the DAG Stage-1 extraction prompt (context only; no output)."""
    rendered: list[dict[str, str]] = []
    for message in COVERAGE_EXTRACT_PROMPT:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(context=context)
        rendered.append({"role": message["role"], "content": content})
    return rendered


# --- G-Eval: one-call identify + classify -----------------------------------

_COVERAGE_GEVAL_CONTRACT = (
    "Evaluate how completely the generated OUTPUT represents the materially "
    "important information in the SOURCE (CONTEXT).\n\n"
    "In this single response:\n"
    "1. identify the materially important source items; and\n"
    "2. classify how completely the OUTPUT represents each identified item.\n\n"
    "SOURCE ITEM RULES\n"
    + _COVERAGE_ITEM_RULES
    + "\n\nCLASSIFICATION RULES\n"
    "For each source item return two booleans:\n"
    '- "meaningfully_present": true when any meaningful semantic part of the item is '
    "represented in the OUTPUT; otherwise false.\n"
    '- "fully_present": true only when the complete material meaning, including all '
    "important qualifiers, is represented in the OUTPUT.\n\n"
    "Judge meaning rather than wording; semantic paraphrases count. If "
    "meaningfully_present is false, fully_present must also be false. Unsupported "
    "additions in the OUTPUT are out of scope and must not reduce the classification. "
    "Do not return an aggregate score, item score, percentage, confidence, or weight; "
    "Python calculates all statuses and numeric scores."
)

_COVERAGE_GEVAL_SYSTEM_COMPACT_V1 = (
    _COVERAGE_GEVAL_CONTRACT
    + "\n\nReturn only each source_item and its two booleans. Do not return reasons "
    "or other prose."
)
_COVERAGE_GEVAL_SYSTEM_VERBOSE_V1 = (
    _COVERAGE_GEVAL_CONTRACT
    + "\n\nAlso return a concise reason for each item that is partial or missing. Use "
    "an empty reason for fully represented items. Do not return other prose."
)

_COVERAGE_GEVAL_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[CONTEXT]
{context}

[OUTPUT]
{output}

[END DATA]"""

COVERAGE_GEVAL_PROMPT_COMPACT_V1 = [
    {"role": "system", "content": _COVERAGE_GEVAL_SYSTEM_COMPACT_V1},
    {"role": "user", "content": _COVERAGE_GEVAL_USER_TEMPLATE_V1},
]
COVERAGE_GEVAL_PROMPT_VERBOSE_V1 = [
    {"role": "system", "content": _COVERAGE_GEVAL_SYSTEM_VERBOSE_V1},
    {"role": "user", "content": _COVERAGE_GEVAL_USER_TEMPLATE_V1},
]


def _geval_item_schema(*, include_reason: bool) -> dict:
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


COVERAGE_GEVAL_SCHEMA_COMPACT = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": _geval_item_schema(include_reason=False)}},
    "required": ["items"],
}
COVERAGE_GEVAL_SCHEMA_VERBOSE = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": _geval_item_schema(include_reason=True)}},
    "required": ["items"],
}


def render_coverage_geval_prompt(
    context: str, output: str, verbose: bool = False
) -> list[dict[str, str]]:
    """Renders the one-call G-Eval prompt (context + output identify + classify)."""
    template = (
        COVERAGE_GEVAL_PROMPT_VERBOSE_V1 if verbose else COVERAGE_GEVAL_PROMPT_COMPACT_V1
    )
    rendered: list[dict[str, str]] = []
    for message in template:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(context=context, output=output)
        rendered.append({"role": message["role"], "content": content})
    return rendered
