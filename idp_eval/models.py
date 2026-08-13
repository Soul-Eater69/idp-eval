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


@dataclass
class EvaluationCase:
    """A single generated output to evaluate.

    ``input`` and ``instructions`` are distinct on purpose so the same case can
    be run through every metric without any field changing meaning:

    - ``input`` is the task/request (used by ``coverage`` to scope relevant
      context, and passed to Phoenix by ``faithfulness``).
    - ``instructions`` is the explicit instruction text evaluated by
      ``instruction_adherence``. It is never derived from ``input``.

    Attributes:
        input: What the model was asked to do (the task/request).
        context: Authoritative source information.
        output: Generated content being evaluated.
        instructions: Explicit instructions to evaluate for adherence. Optional;
            required only by the ``instruction_adherence`` metric.
        case_id: Optional identifier for tracing and reporting.
        metadata: Optional free-form metadata carried alongside the case.
    """

    input: str
    context: str
    output: str
    instructions: str | None = None
    case_id: str | None = None
    metadata: dict[str, Any] | None = None


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

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the metric name."""

    @abstractmethod
    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates one case and returns a normalized result."""
