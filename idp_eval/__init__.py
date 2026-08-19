"""IDP LLM Evaluation Framework.

A reusable evaluation framework on top of Arize Phoenix that can evaluate any
generated AI output using a generic ``input`` / ``context`` / ``output`` triple.

Metrics:
    faithfulness: is the output grounded in the context? (detects hallucinated /
        unsupported additions). Higher is better.
    coverage: one-call itemized judgment of how much of the whole source reached
        the output (detects omissions). Higher is better.
    instruction_adherence: does the output obey the explicit instructions in
        ``instructions``? Higher is better.
    relevance_at_{k} / ndcg_at_{k}: retrieval metrics over ranked
        ``retrieved_documents`` for a query (``input``). Relevance@K is the
        fraction of the top-K documents that are relevant (= Precision@K under
        binary relevance); nDCG@K is ranking quality from the same per-document
        relevance judgments. Higher is better.

Which ``EvaluationCase`` fields each metric reads:
    faithfulness: input (task) + context + output, passed to Phoenix.
    coverage: context + output (``input`` is ignored).
    instruction_adherence: instructions + output. It reads only the dedicated
        ``instructions`` field as its instruction source, never ``input`` or
        ``context``.
    relevance_at_{k} / ndcg_at_{k}: input (query) + retrieved_documents. No
        generated output is required.
"""

from idp_eval.evaluators import (
    CoverageEvaluator,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
    NDCGAtKEvaluator,
    RelevanceAtKEvaluator,
)
from idp_eval.framework import EvaluationFramework
from idp_eval.judge import JudgeConfig, create_judge
from idp_eval.judges import (
    AzureJudgeConfig,
    GatewayJudgeConfig,
    Judge,
    create_azure_judge,
    create_gateway_judge,
)
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.output import (
    ANNOTATOR_KINDS,
    EvaluationRecord,
    EvaluationWriter,
    ExcelEvaluationWriter,
    PersistenceError,
    PhoenixEvaluationWriter,
)
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
    "JudgeConfig",
    "GatewayJudgeConfig",
    "AzureJudgeConfig",
    "Judge",
    "create_judge",
    "create_gateway_judge",
    "create_azure_judge",
    "register_tracing",
    "ANNOTATOR_KINDS",
    "EvaluationRecord",
    "EvaluationWriter",
    "ExcelEvaluationWriter",
    "PhoenixEvaluationWriter",
    "PersistenceError",
]

__version__ = "0.1.0"
