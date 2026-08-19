"""Public API for the idp-eval framework."""

from idp_eval.evaluators import (
    CoverageEvaluator,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
    NDCGAtKEvaluator,
    RelevanceAtKEvaluator,
)
from idp_eval.framework import EvaluationFramework
from idp_eval.judges import (
    create_azure_judge,
    create_gateway_judge,
)
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.output import PersistenceError
from idp_eval.phoenix_client import register_tracing

__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "Evaluator",
    "EvaluationFramework",
    "FaithfulnessEvaluator",
    "CoverageEvaluator",
    "InstructionAdherenceEvaluator",
    "RelevanceAtKEvaluator",
    "NDCGAtKEvaluator",
    "create_gateway_judge",
    "create_azure_judge",
    "register_tracing",
    "PersistenceError",
]

__version__ = "0.1.0"
