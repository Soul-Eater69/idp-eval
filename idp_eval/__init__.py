"""IDP LLM Evaluation Framework.

A reusable evaluation framework on top of Arize Phoenix that can evaluate any
generated AI output using a generic ``input`` / ``context`` / ``output`` triple.

v1 metrics:
    faithfulness: is the output grounded in the context? (detects hallucinated /
        unsupported additions). Higher is better.
    coverage: how much task-relevant context reached the output? (detects
        omissions). Higher is better.
    instruction_following: does the output obey the explicit instructions in
        input? Higher is better.

The meaning of ``EvaluationCase.input`` depends on the metric/application:
    faithfulness: task information passed to Phoenix alongside context/output.
    coverage: the task/request used to scope relevant context.
    instruction_following: the explicit instruction text to evaluate.
"""

from idp_eval.evaluators import (
    CoverageEvaluator,
    FaithfulnessEvaluator,
    InstructionFollowingEvaluator,
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
    "InstructionFollowingEvaluator",
]

__version__ = "0.1.0"
