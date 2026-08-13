"""IDP LLM Evaluation Framework.

A reusable evaluation framework on top of Arize Phoenix that can evaluate any
generated AI output using a generic ``input`` / ``context`` / ``output`` triple.

v1 metrics:
    faithfulness: is the output grounded in the context? (detects hallucinated /
        unsupported additions). Higher is better.
    coverage: how much task-relevant context reached the output? (detects
        omissions). Higher is better.
    instruction_adherence: does the output obey the explicit instructions in
        input? Higher is better.

Which ``EvaluationCase`` fields each metric reads:
    faithfulness: input (task) + context + output, passed to Phoenix.
    coverage: input (task, used to scope relevant context) + context + output.
    instruction_adherence: instructions + context + output. It reads the
        dedicated ``instructions`` field, never ``input``.
"""

from idp_eval.evaluators import (
    CoverageEvaluator,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
)
from idp_eval.framework import EvaluationFramework
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator

__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "Evaluator",
    "EvaluationFramework",
    "FaithfulnessEvaluator",
    "CoverageEvaluator",
    "InstructionAdherenceEvaluator",
]

__version__ = "0.1.0"
