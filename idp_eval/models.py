"""Core data model and evaluator interface for the idp eval framework.

These types are intentionally generic. They carry no application-specific
vocabulary (no Jira, no RAG, no test cases). A generated output is described by
an ``input`` (task/request), optional explicit ``instructions``, authoritative
``context``, and the generated ``output``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from idp_eval.rendering import (
    StructuredValue,
    is_empty_value,
    validate_structured_value,
)

# Content fields that may hold structured values and are rendered for prompts.
_CONTENT_FIELDS = ("input", "context", "output", "instructions")


@dataclass
class EvaluationCase:
    """A single generated output to evaluate.

    Represents exactly ONE logical evaluation unit. ``input`` and
    ``instructions`` are distinct on purpose so the same case can be run through
    every metric without any field changing meaning:

    - ``input`` is the task/request (used by ``task_coverage`` to scope relevant
      context, and passed to Phoenix by ``faithfulness``).
    - ``instructions`` is the explicit instruction text evaluated by
      ``instruction_adherence``. It is never derived from ``input``.

    The content fields (``input`` / ``context`` / ``output`` / ``instructions``)
    may be **structured values** — nested ``dict`` / ``list`` of scalars — not
    just strings; they are rendered to readable text per evaluator. A ``list``
    output is a single structured output, not a request to evaluate many outputs
    (use ``EvaluationFramework.evaluate_many`` / ``evaluate_groups`` for that).
    All content fields are optional at the model level; each evaluator declares
    which it actually requires.

    Attributes:
        input: What the model was asked to do (the task/request).
        context: Authoritative source information.
        output: Generated content being evaluated.
        instructions: Explicit instructions to evaluate for adherence.
        case_id: Optional identifier for tracing and reporting.
        metadata: Optional free-form metadata carried alongside the case. Never
            injected into evaluator prompts.
        retrieved_documents: Optional ordered list of retrieved documents for the
            retrieval metrics (``relevance_at_k`` / ``ndcg_at_k``). **List order
            is the retrieval rank.** Each entry is a document string or a mapping
            with a text field (default key ``"text"``) plus optional
            ``document_id`` and ``score`` (similarity) metadata. Used only by the
            retrieval evaluators; other metrics ignore it.
    """

    input: StructuredValue = None
    context: StructuredValue = None
    output: StructuredValue = None
    instructions: StructuredValue = None
    case_id: str | None = None
    metadata: dict[str, Any] | None = None
    retrieved_documents: StructuredValue = None

    def __post_init__(self) -> None:
        """Structural (Level 1) validation of the structured content fields."""
        for name in _CONTENT_FIELDS:
            validate_structured_value(getattr(self, name), f"EvaluationCase.{name}")
        # Retrieved documents are structured values too (list of str/dict), but
        # are not rendered like the content fields — they feed the relevance
        # judge as individual document texts.
        validate_structured_value(
            self.retrieved_documents, "EvaluationCase.retrieved_documents"
        )


@dataclass
class EvaluationResult:
    """Normalized result returned by every evaluator.

    Keeping this stable and independent of Phoenix's internal ``Score`` object
    is what lets application code depend on our framework rather than Phoenix.

    Attributes:
        metric: Name of the metric that produced this result.
        score: Numeric score, typically in ``[0, 1]``. ``None`` if not scored.
        label: Human-readable label such as ``"high"`` or ``"unfaithful"``.
        explanation: Short natural-language justification for the score.
        details: Metric-specific structured detail (e.g. missing items).
    """

    metric: str
    score: float | None
    label: str | None
    explanation: str | None
    details: dict[str, Any] | None = None


class Evaluator(ABC):
    """Base interface for all evaluation metrics.

    A future metric only needs to implement this interface; the framework does
    not need to change to accommodate it.
    """

    # How this evaluator's results are produced, for publishing/annotation.
    # Built-in LLM judges are "LLM"; deterministic Python evaluators may set
    # "CODE". One of ``idp_eval.output.ANNOTATOR_KINDS``.
    annotator_kind: str = "LLM"

    # Case content fields this evaluator requires to be non-empty. The framework
    # enforces these (Level 2 validation) before the evaluator's first judge
    # call. Unused fields on a case are allowed and simply ignored.
    required_fields: tuple[str, ...] = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the metric name."""

    @abstractmethod
    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates one case and returns a normalized result."""

    def validate_case(self, case: EvaluationCase) -> None:
        """Checks that every :attr:`required_fields` value is non-empty.

        Raises a clear ``ValueError`` (naming the missing fields and showing
        present/missing for each requirement) before any judge work. Extra
        fields the evaluator does not use are ignored.
        """
        missing = [
            field
            for field in self.required_fields
            if is_empty_value(getattr(case, field, None))
        ]
        if not missing:
            return
        received = "\n".join(
            f"  {field}: "
            f"{'missing' if is_empty_value(getattr(case, field, None)) else 'present'}"
            for field in self.required_fields
        )
        names = ", ".join(f"`{field}`" for field in missing)
        raise ValueError(
            f"{type(self).__name__} requires non-empty {names}.\n\n"
            f"Received:\n{received}"
        )
