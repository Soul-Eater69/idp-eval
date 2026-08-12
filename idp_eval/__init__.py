"""Holistic LLM Evaluation Framework.

A reusable evaluation framework on top of Arize Phoenix that can evaluate any
generated AI output using a generic ``input`` / ``context`` / ``output`` triple.
"""

from idp_eval.evaluators import (
    FaithfulnessMetric,
    HallucinationEvaluator,
    InputCoverageEvaluator,
)
from idp_eval.framework import EvaluationFramework
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator

__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "Evaluator",
    "EvaluationFramework",
    "FaithfulnessMetric",
    "HallucinationEvaluator",
    "InputCoverageEvaluator",
]

__version__ = "0.1.0"
