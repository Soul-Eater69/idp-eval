"""Prompt for the coverage evaluator.

Coverage runs from CONTEXT to OUTPUT: it measures how much of the material
context information that is relevant to the requested task (INPUT) is
represented in the OUTPUT. The judge uses INPUT to understand and scope the task,
not as throwaway background.

The prompt is a Phoenix-style message list (separate system and user messages).
The system message holds only the evaluation rubric; the user message holds only
the data being evaluated. This Python file is the source of truth for v1.
"""

from __future__ import annotations

# --- System rubric ----------------------------------------------------------
# Contains no ``{}`` placeholders so it is safe to leave untouched during
# rendering.
_COVERAGE_SYSTEM_V1 = """\
You are a strict evaluator measuring COVERAGE of a generated output against an
authoritative context, for a specific requested task.

DEFINITION
Coverage measures how much of the material information in the authoritative
CONTEXT that is relevant to satisfying the requested task (the INPUT) is
represented in the generated OUTPUT.

Coverage does NOT measure any of the following (they belong to other metrics
such as faithfulness):
- hallucination;
- unsupported additions;
- factual grounding;
- writing quality;
- style;
- formatting quality;
- correctness against external knowledge.

MATERIAL INFORMATION
A source item is material if omitting it could change the meaning, requirement,
expected behavior, objective, constraint, decision, business rule, capability,
relationship, success condition, or expected outcome of the requested output.

Ignore and do not extract:
- boilerplate;
- irrelevant context;
- formatting;
- headings;
- IDs with no semantic importance;
- repeated information;
- purely stylistic wording;
- metadata that does not affect the requested output.

SCOPE THE CONTEXT WITH THE INPUT
1. Read the INPUT first.
2. Understand what task was requested.
3. Identify only the material information in the CONTEXT that is relevant to
   satisfying that task.
4. Ignore context that is unrelated to the requested task. Do not penalize the
   OUTPUT for omitting information that is outside the task scope.

For example, if the CONTEXT contains A, B, C, D, E but the INPUT asks only about
A and B, and the OUTPUT covers A and B, then coverage is complete. C, D, and E
are out of scope and must not be extracted as material items.

EXTRACT ATOMIC BUT MEANINGFUL SOURCE ITEMS
Keep source items atomic but semantically meaningful. Do not split a single
requirement or idea into multiple wording-level fragments merely because it
contains several words or clauses. For example,
"Users must be able to view and download invoices." must NOT be split into
meaningless fragments such as "users", "view", "download", "invoices". Split a
statement only if it contains genuinely independent semantic requirements.

CLASSIFICATION
Classify each material source item as exactly one of:
- "covered": the OUTPUT fully preserves the material meaning of the source item.
  Exact wording is not required; semantic paraphrases and equivalent expressions
  count as covered.
- "partial": the OUTPUT represents some meaningful part of the source item but
  omits, weakens, generalizes, or only vaguely represents another material part.
- "missing": the material source item is not meaningfully represented in the
  OUTPUT.

COVERAGE IS SEMANTIC, NOT LEXICAL
- Do not require exact phrase matching.
- Different wording can still be fully covered.
- Synonyms, paraphrases, restructuring, summarization, and equivalent
  expressions count when the material meaning is preserved.
- Do not use fuzzy string similarity as the decision rule.
- Judge semantic meaning.

COVERAGE VS FAITHFULNESS
Do not penalize the OUTPUT for adding unsupported information in this evaluator.
Unsupported additions are evaluated by the faithfulness metric.
- Coverage asks: "Did the OUTPUT omit important task-relevant source
  information?"
- Faithfulness asks: "Did the OUTPUT add unsupported information?"
Only classify source-to-output coverage.

OUTPUT FORMAT
Return one entry per material source item, each with the source item text, its
status, and a brief reason. Do NOT return any numeric score, percentage, or
weighting. Python computes the score from your classifications.\
"""

# --- User data template ------------------------------------------------------
# Holds only the data being evaluated; ``{input}``/``{context}``/``{output}`` are
# filled per call by ``render_coverage_prompt``.
_COVERAGE_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[INPUT]
{input}

[CONTEXT]
{context}

[OUTPUT]
{output}

[END DATA]"""

# Versioned message-list prompt. The user message still carries placeholders; it
# is rendered per call so the module-level template is never mutated.
COVERAGE_PROMPT_V1 = [
    {"role": "system", "content": _COVERAGE_SYSTEM_V1},
    {"role": "user", "content": _COVERAGE_USER_TEMPLATE_V1},
]

# Current prompt used by the evaluator. Point this at a new version (e.g.
# COVERAGE_PROMPT_V2) to benchmark prompts without silently changing the metric.
COVERAGE_PROMPT = COVERAGE_PROMPT_V1

# JSON schema for the structured judge response. ``reason`` is optional and kept
# only for explainability/debugging.
COVERAGE_SCHEMA = {
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
