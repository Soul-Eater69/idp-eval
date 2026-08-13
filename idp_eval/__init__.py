"""IDP LLM Evaluation Framework.

A reusable evaluation framework on top of Arize Phoenix that can evaluate any
generated AI output using a generic ``input`` / ``context`` / ``output`` triple.

v1 metrics:
    faithfulness: is the output grounded in the context? (detects hallucinated /
        unsupported additions). Higher is better.
    coverage: how much task-relevant context reached the output? (detects
        omissions). Higher is better.
"""

from idp_eval.evaluators import CoverageEvaluator, FaithfulnessEvaluator
from idp_eval.framework import EvaluationFramework
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator

__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "Evaluator",
    "EvaluationFramework",
    "FaithfulnessEvaluator",
    "CoverageEvaluator",
]

__version__ = "0.1.0"
