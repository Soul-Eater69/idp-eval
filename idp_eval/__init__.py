"""IDP LLM Evaluation Framework.

A reusable evaluation framework on top of Arize Phoenix that can evaluate any
generated AI output using a generic ``input`` / ``context`` / ``output`` triple.

v1 metrics:
    faithfulness: is the output grounded in the context? (detects hallucinated /
        unsupported additions). Higher is better.
    source_coverage: one-call itemized judgment of how much of the whole source
        reached the output (detects omissions, task-agnostic). Higher is better.
    task_coverage: how much of the task-relevant source reached the output?
        (detects omissions, scoped by the task). Higher is better.
    instruction_adherence: does the output obey the explicit instructions in
        ``instructions``? Higher is better.
    relevance_at_{k} / ndcg_at_{k}: retrieval metrics over ranked
        ``retrieved_documents`` for a query (``input``). Relevance@K is the
        fraction of the top-K documents that are relevant (= Precision@K under
        binary relevance); nDCG@K is ranking quality from the same per-document
        relevance judgments. Higher is better.

Which ``EvaluationCase`` fields each metric reads:
    faithfulness: input (task) + context + output, passed to Phoenix.
    source_coverage: context + output (``input`` is ignored).
    task_coverage: input (task, used to scope relevant context) + context +
        output.
    instruction_adherence: instructions + output. It reads only the dedicated
        ``instructions`` field as its instruction source, never ``input`` or
        ``context``.
    relevance_at_{k} / ndcg_at_{k}: input (query) + retrieved_documents. No
        generated output is required.
"""

from idp_eval.evaluators import (
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
    NDCGAtKEvaluator,
    RelevanceAtKEvaluator,
    SourceCoverageEvaluator,
    TaskCoverageEvaluator,
)
from idp_eval.framework import EvaluationFramework
from idp_eval.judge import JudgeConfig, create_judge
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
    "TaskCoverageEvaluator",
    "SourceCoverageEvaluator",
    "InstructionAdherenceEvaluator",
    "RelevanceAtKEvaluator",
    "NDCGAtKEvaluator",
    "JudgeConfig",
    "create_judge",
    "register_tracing",
    "ANNOTATOR_KINDS",
    "EvaluationRecord",
    "EvaluationWriter",
    "ExcelEvaluationWriter",
    "PhoenixEvaluationWriter",
    "PersistenceError",
]

__version__ = "0.1.0"
