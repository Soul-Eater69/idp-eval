"""The evaluation framework: a thin orchestrator over a set of evaluators."""

from __future__ import annotations

from typing import Union

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator

# An entry may be an evaluator class (constructed with the shared judge) or an
# already-constructed evaluator instance (backward-compatible).
EvaluatorEntry = Union[type[Evaluator], Evaluator]


class EvaluationFramework:
    """Runs evaluation metrics for generated AI content.

    The preferred API passes evaluator *classes* plus one shared ``judge``; the
    framework constructs each with ``evaluator_class(llm=judge)``. Passing
    already-constructed evaluator instances is still supported. Application code
    should call :meth:`evaluate` rather than talking to Phoenix directly, which
    keeps the backend swappable.
    """

    def __init__(
        self,
        evaluators: list[EvaluatorEntry],
        judge=None,
    ):
        """Initializes the framework.

        Args:
            evaluators: Evaluator classes and/or instances to register. Classes
                are instantiated as ``cls(llm=judge)``. The caller chooses
                exactly which evaluators are active; nothing is auto-enabled.
            judge: The shared judge LLM injected into evaluator classes. Required
                only when at least one entry is a class.

        Raises:
            TypeError: If an entry is neither an ``Evaluator`` subclass nor an
                ``Evaluator`` instance.
            ValueError: If a class entry is given without a ``judge``, if an
                evaluator constructor fails, or if two evaluators share a name.
        """
        self._evaluators: dict[str, Evaluator] = {}
        for entry in evaluators:
            evaluator = self._instantiate(entry, judge)
            if evaluator.name in self._evaluators:
                raise ValueError(
                    f"Duplicate evaluator name: {evaluator.name!r}"
                )
            self._evaluators[evaluator.name] = evaluator

    @staticmethod
    def _instantiate(entry: EvaluatorEntry, judge) -> Evaluator:
        """Turns an entry into an evaluator instance."""
        if isinstance(entry, type):
            if not issubclass(entry, Evaluator):
                raise TypeError(
                    f"Evaluator class must subclass Evaluator: {entry!r}"
                )
            if judge is None:
                raise ValueError(
                    f"A judge is required to construct evaluator class "
                    f"{entry.__name__!r}; pass judge=create_judge()."
                )
            try:
                return entry(llm=judge)
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise ValueError(
                    f"Failed to construct evaluator {entry.__name__!r}: {exc}"
                ) from exc

        if isinstance(entry, Evaluator):
            return entry

        raise TypeError(
            "Evaluator entry must be an Evaluator subclass or instance, got "
            f"{entry!r}"
        )

    @property
    def metrics(self) -> list[str]:
        """Returns the names of all registered metrics."""
        return list(self._evaluators)

    def evaluate(
        self,
        case: EvaluationCase,
        metrics: list[str] | None = None,
    ) -> dict[str, EvaluationResult]:
        """Evaluates one case with the selected metrics.

        Args:
            case: The case to evaluate.
            metrics: Metric names to run. Runs all registered metrics when
                ``None``.

        Returns:
            A mapping of metric name to its :class:`EvaluationResult`.

        Raises:
            KeyError: If a requested metric name is not registered.
        """
        selected = metrics if metrics is not None else self.metrics

        results: dict[str, EvaluationResult] = {}
        for metric_name in selected:
            if metric_name not in self._evaluators:
                raise KeyError(f"Unknown metric: {metric_name!r}")
            results[metric_name] = self._evaluators[metric_name].evaluate(case)

        return results
