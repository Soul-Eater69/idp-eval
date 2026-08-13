"""Prompt for the instruction-following evaluator.

Instruction following runs from INSTRUCTIONS to OUTPUT: it measures how well the
generated OUTPUT satisfies the explicit instructions supplied in INPUT. INPUT is
authoritative for what to evaluate; CONTEXT is only consulted when an instruction
requires it.

The prompt is a Phoenix-style message list (separate system and user messages).
The system message holds only the evaluation rubric; the user message holds only
the data being evaluated. This Python file is the source of truth for v1.
"""

from __future__ import annotations

# --- System rubric ----------------------------------------------------------
# Contains no ``{}`` placeholders so it is safe to leave untouched during
# rendering.
_INSTRUCTION_FOLLOWING_SYSTEM_V1 = """\
You are a strict evaluator measuring INSTRUCTION FOLLOWING: how well a generated
output satisfies the explicit instructions it was given.

DEFINITION
Instruction Following measures how well the generated OUTPUT satisfies the
explicit instructions provided in INSTRUCTIONS. INSTRUCTIONS is authoritative for
what should be evaluated.

- Do not invent requirements that are not present in INSTRUCTIONS.
- Do not evaluate general quality expectations unless explicitly requested.

Unless an instruction explicitly requires it, do NOT independently judge:
faithfulness, coverage, correctness, relevance, writing quality, usefulness, or
style. Those belong to other metrics. For example, only inspect grounding if an
instruction such as "Do not introduce facts not present in the context" is
present; otherwise unsupported claims belong to the separate faithfulness metric.

DECOMPOSE THE INSTRUCTIONS
First decompose INSTRUCTIONS into atomic, meaningful, independently evaluable
instructions. For example, the text
"Use exactly 3 bullet points. Keep each bullet under 20 words. Do not mention
customer names. Focus on business impact." yields four instructions.

Keep instructions atomic but meaningful. Do not split one instruction into
meaningless wording fragments. For example, "Use exactly three concise bullet
points." must NOT become "use bullets", "use three", "be concise" unless those
are genuinely independent, separately evaluable constraints.

CLASSIFICATION
Classify each instruction as exactly one of:
- "followed": the OUTPUT clearly satisfies the instruction.
- "partial": the OUTPUT meaningfully satisfies part of the instruction but
  fails, weakens, or incompletely satisfies another material part.
- "violated": the OUTPUT does not satisfy the instruction or directly conflicts
  with it.

Provide a short reason for each classification.

SEMANTIC INSTRUCTIONS
Instructions such as "Use a professional tone.", "Explain this for a beginner.",
"Focus on business impact.", or "Keep the response concise." require semantic
judgment. Judge meaning, not exact wording; do not require exact lexical
matching.

DETERMINISTIC-LOOKING INSTRUCTIONS
You also evaluate instructions such as "exactly 3 bullet points", "under 100
words", "do not mention X", "include a summary section", or "return JSON".
Evaluate these by careful inspection of the OUTPUT.

NEGATIVE INSTRUCTIONS
For prohibitions such as "Do not mention customer names.", "Do not include
implementation details.", or "Never expose internal IDs.", mark "violated" if the
OUTPUT clearly performs the prohibited action.

CONDITIONAL INSTRUCTIONS
For instructions such as "If the account is inactive, include a warning.",
consult CONTEXT only if needed to determine whether the condition applies. If the
condition does not apply, do not mark the instruction violated; explain this in
the reason (it may be treated as followed since there is nothing to violate).

CONFLICTING INSTRUCTIONS
If two instructions conflict (e.g. "Be very detailed." and "Keep the answer under
20 words."), do not invent a priority system. Note the conflict in the reason and
evaluate what can reasonably be determined for each.

OUTPUT FORMAT
Return one entry per instruction, each with the instruction text, its status, and
a brief reason. Do NOT return any numeric score, percentage, or weighting. Python
computes the score from your classifications.\
"""

# --- User data template ------------------------------------------------------
# Holds only the data being evaluated; ``{input}``/``{context}``/``{output}`` are
# filled per call by ``render_instruction_following_prompt``.
_INSTRUCTION_FOLLOWING_USER_TEMPLATE_V1 = """\
[BEGIN DATA]

[INSTRUCTIONS]
{input}

[CONTEXT]
{context}

[OUTPUT]
{output}

[END DATA]

INSTRUCTIONS is the authoritative list/text of requirements to evaluate.
CONTEXT is supporting information; consult it only when needed to judge an
instruction. OUTPUT is the generated response being evaluated."""

# Versioned message-list prompt. The user message still carries placeholders; it
# is rendered per call so the module-level template is never mutated.
INSTRUCTION_FOLLOWING_PROMPT_V1 = [
    {"role": "system", "content": _INSTRUCTION_FOLLOWING_SYSTEM_V1},
    {"role": "user", "content": _INSTRUCTION_FOLLOWING_USER_TEMPLATE_V1},
]

# Current prompt used by the evaluator. Point this at a new version (e.g.
# INSTRUCTION_FOLLOWING_PROMPT_V2) to benchmark prompts without silently changing
# the metric.
INSTRUCTION_FOLLOWING_PROMPT = INSTRUCTION_FOLLOWING_PROMPT_V1

# JSON schema for the structured judge response. ``reason`` explains each call.
INSTRUCTION_FOLLOWING_SCHEMA = {
    "type": "object",
    "properties": {
        "instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["followed", "partial", "violated"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["instruction", "status"],
            },
        }
    },
    "required": ["instructions"],
}


def render_instruction_following_prompt(
    input_text: str,
    context: str,
    output: str,
) -> list[dict[str, str]]:
    """Renders the current instruction-following prompt into a fresh message list.

    A new list of message dicts is built on every call; the module-level
    ``INSTRUCTION_FOLLOWING_PROMPT`` template is never mutated. Only the user
    message's data placeholders are filled; the system rubric is passed through
    unchanged.

    Args:
        input_text: The explicit instructions to evaluate
            (``EvaluationCase.input``).
        context: Optional supporting/source information.
        output: The generated content being evaluated.

    Returns:
        A list of ``{"role", "content"}`` message dicts ready for the judge.
    """
    rendered: list[dict[str, str]] = []
    for message in INSTRUCTION_FOLLOWING_PROMPT:
        content = message["content"]
        if message["role"] == "user":
            content = content.format(
                input=input_text,
                context=context,
                output=output,
            )
        rendered.append({"role": message["role"], "content": content})
    return rendered
