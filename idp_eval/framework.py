"""The evaluation framework: a thin orchestrator over a set of evaluators."""

from __future__ import annotations

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator


class EvaluationFramework:
    """Runs evaluation metrics for generated AI content.

    Application code should call :meth:`evaluate` rather than talking to Phoenix
    directly. That keeps the backend swappable.
    """

    def __init__(self, evaluators: list[Evaluator]):
        """Initializes the framework.

        Args:
            evaluators: Evaluators to register, keyed by their ``name``.

        Raises:
            ValueError: If two evaluators share the same name.
        """
        self._evaluators: dict[str, Evaluator] = {}
        for evaluator in evaluators:
            if evaluator.name in self._evaluators:
                raise ValueError(
                    f"Duplicate evaluator name: {evaluator.name!r}"
                )
            self._evaluators[evaluator.name] = evaluator

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
