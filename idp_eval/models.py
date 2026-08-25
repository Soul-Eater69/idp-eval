"""Core data model and evaluator interface for idp-eval.

The model is domain-agnostic: a generated output is described by an ``input``,
optional ``instructions``, authoritative ``context``, and the ``output``.
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
_EVALUATION_SCOPES = ("combined", "individual", "both")


@dataclass
class EvaluationCase:
    """One generation to evaluate, optionally containing multiple outputs.

    Represents one generation request. Scope expansion may produce multiple
    logical evaluations, but ``input`` and ``instructions`` remain distinct so
    every metric receives fields with consistent meaning:

    - ``input`` is the task/request/query when one exists. Retrieval metrics use
      it as the query; other metrics may not need it.
    - ``context`` is authoritative source/reference evidence.
    - ``instructions`` contains explicit behavioral/output constraints. It is
      never derived from ``input``.
    - ``output`` is the generated result being evaluated.

    The content fields (``input`` / ``context`` / ``output`` / ``instructions``)
    may be **structured values** — nested ``dict`` / ``list`` of scalars — not
    just strings; they are rendered to readable text per evaluator. A ``list``
    output is one structured output by default. Set ``evaluation_scope`` to
    ``"individual"`` to evaluate each top-level item independently or ``"both"``
    to evaluate the list and every item.
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
        retrieved_documents: Optional ordered list of retrieved documents for
            retrieval and contextual retrieval metrics. **List order is the
            retrieval rank.** Each entry is a document string or a mapping with
            a text field (default key ``"text"``) plus optional ``document_id``
            and ``score`` (similarity) metadata. Used only by retrieval
            evaluators; other metrics ignore it.
        evaluation_scope: How a top-level list output is orchestrated:
            ``"combined"`` (default), ``"individual"``, or ``"both"``.
    """

    input: StructuredValue = None
    context: StructuredValue = None
    output: StructuredValue = None
    instructions: StructuredValue = None
    case_id: str | None = None
    metadata: dict[str, Any] | None = None
    retrieved_documents: StructuredValue = None
    evaluation_scope: str = "combined"

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
        if self.evaluation_scope not in _EVALUATION_SCOPES:
            allowed = ", ".join(repr(value) for value in _EVALUATION_SCOPES)
            raise ValueError(
                f"Unknown evaluation_scope {self.evaluation_scope!r}; allowed "
                f"values are: {allowed}."
            )


@dataclass
class EvaluationResult:
    """Normalized result returned by every evaluator.

    Keeping this stable and independent of Phoenix's internal ``Score`` object
    is what lets application code depend on our framework rather than Phoenix.

    Attributes:
        metric: Name of the metric that produced this result.
        score: Numeric score, typically in ``[0, 1]``. ``None`` if not scored.
        label: Human-readable metric-specific label.
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
    _llm: Any | None = None

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

    def _bind_judge(self, judge: Any) -> None:
        """Binds a framework judge only when none was supplied explicitly."""
        if self._llm is None:
            self._llm = judge

    def _require_judge(self) -> Any:
        """Returns the configured judge or raises at the first judge operation."""
        if self._llm is None:
            raise ValueError(
                f"{type(self).__name__} requires a judge. Pass it directly or "
                "provide judge=... to EvaluationFramework."
            )
        return self._llm

    def resume_signature(self) -> dict[str, Any]:
        """Returns explicit score-affecting configuration for resume identity.

        Custom configurable evaluators should override this method. The
        evaluator class and metric name are fingerprinted separately, so the
        safe default only needs a manually bumped contract version.
        """
        return {"contract_version": 1}

    @staticmethod
    def judge_resume_signature(judge: Any) -> dict[str, Any]:
        """Returns safe, stable judge identity without endpoint/auth values."""
        signature: dict[str, Any] = {
            "type": f"{type(judge).__module__}.{type(judge).__qualname__}"
        }
        for name in ("model", "provider", "client"):
            try:
                value = getattr(judge, name, None)
            except Exception:  # pragma: no cover - defensive custom property
                continue
            if isinstance(value, (str, int, float, bool)):
                signature[name] = value
        return signature
