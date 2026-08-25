"""Deterministic evaluation fingerprints and persistence audit rendering."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from idp_eval.models import EvaluationCase, Evaluator
from idp_eval.rendering import render_value


def canonical_json(value: Any) -> str:
    """Serializes supported structured values deterministically for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def case_fingerprint(case: EvaluationCase) -> str:
    """Hashes exact score-relevant case content, preserving every list order."""
    return _sha256(
        {
            "input": case.input,
            "context": case.context,
            "output": case.output,
            "instructions": case.instructions,
            "retrieved_documents": case.retrieved_documents,
            "evaluation_scope": case.evaluation_scope,
        }
    )


def evaluation_fingerprint(
    case_hash: str, metric: str, evaluator: Evaluator
) -> str:
    """Hashes the case plus explicit evaluator identity/configuration."""
    return _sha256(
        {
            "case_fingerprint": case_hash,
            "metric": metric,
            "evaluator_type": (
                f"{type(evaluator).__module__}.{type(evaluator).__qualname__}"
            ),
            "evaluator_signature": evaluator.resume_signature(),
        }
    )


def external_evaluation_fingerprint(
    case_hash: str, metric: str, annotator_kind: str
) -> str:
    """Hashes an externally supplied evaluation without unsafe object reprs."""
    return _sha256(
        {
            "case_fingerprint": case_hash,
            "metric": metric,
            "evaluator_type": "external",
            "evaluator_signature": {
                "contract_version": 1,
                "annotator_kind": annotator_kind,
            },
        }
    )


def rendered_case_fields(case: EvaluationCase) -> dict[str, str | None]:
    """Returns human-readable audit fields without changing evaluator prompts."""
    return {
        "input": render_value(case.input) or None,
        "context": render_value(case.context) or None,
        "output": render_value(case.output) or None,
        "instructions": render_value(case.instructions) or None,
        "evaluation_scope": case.evaluation_scope,
        "retrieved_documents_json": (
            canonical_json(case.retrieved_documents)
            if case.retrieved_documents is not None
            else None
        ),
        "retrieved_documents": render_value(case.retrieved_documents) or None,
    }
