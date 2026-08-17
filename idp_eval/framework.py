"""The evaluation framework: orchestrates evaluators, tracing, and output.

The framework owns the case-level trace (one ``EvaluationCase`` -> one trace),
runs the selected evaluators (whose real judge calls become child spans), and
publishes the resulting :class:`EvaluationResult` objects once to the configured
output(s). Evaluation is never re-run for a second destination.
"""

from __future__ import annotations

from typing import Union

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.output import (
    EvaluationRecord,
    PersistenceError,
    build_writers,
    validate_annotator_kind,
)

# An entry may be an evaluator class (constructed with the shared judge) or an
# already-constructed evaluator instance (backward-compatible).
EvaluatorEntry = Union[type[Evaluator], Evaluator]


class EvaluationFramework:
    """Runs evaluation metrics for generated AI content.

    The preferred API passes evaluator *classes* plus one shared ``judge``; the
    framework constructs each with ``evaluator_class(llm=judge)``. Passing
    already-constructed instances is still supported.

    Output is optional and orthogonal to evaluation: ``output`` selects where the
    computed results are published (``"phoenix"``, ``"excel"``, ``"both"``, or
    ``None``). With ``None`` the framework still returns Python results and, if
    ``register_tracing`` was called, still emits the case trace — it just logs no
    result annotations/rows.
    """

    def __init__(
        self,
        evaluators: list[EvaluatorEntry],
        judge=None,
        output: str | None = None,
        excel_path: str | None = None,
    ):
        """Initializes the framework.

        Args:
            evaluators: Evaluator classes and/or instances to register. Classes
                are instantiated as ``cls(llm=judge)``.
            judge: The shared judge LLM injected into evaluator classes. Required
                only when at least one entry is a class.
            output: Where to publish results: ``None``, ``"phoenix"``,
                ``"excel"``, or ``"both"``.
            excel_path: Destination ``.xlsx`` path; required when ``output``
                includes Excel.

        Raises:
            TypeError: If an entry is neither an ``Evaluator`` subclass nor
                instance.
            ValueError: For a class entry without a judge, a duplicate metric
                name, an unknown output mode, or a missing Excel path.
        """
        self._evaluators: dict[str, Evaluator] = {}
        for entry in evaluators:
            evaluator = self._instantiate(entry, judge)
            if evaluator.name in self._evaluators:
                raise ValueError(
                    f"Duplicate evaluator name: {evaluator.name!r}"
                )
            self._evaluators[evaluator.name] = evaluator

        self._writers = build_writers(output, excel_path)

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
        run_name: str | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, EvaluationResult]:
        """Evaluates one case (one trace) with the selected metrics.

        Args:
            case: The case to evaluate.
            metrics: Metric names to run. Runs all registered metrics when
                ``None``.
            run_name: Optional benchmark run name, attached to the trace and
                every published row.
            dataset_name: Optional dataset name, attached likewise.

        Returns:
            A mapping of metric name to its :class:`EvaluationResult`.

        Raises:
            KeyError: If a requested metric name is not registered.
            PersistenceError: If evaluation succeeds but a configured writer
                fails; the computed results are preserved on the exception.
        """
        selected = metrics if metrics is not None else self.metrics
        for metric_name in selected:
            if metric_name not in self._evaluators:
                raise KeyError(f"Unknown metric: {metric_name!r}")

        with tracing.case_evaluation_span(
            case.case_id, run_name, dataset_name
        ) as handle:
            results: dict[str, EvaluationResult] = {}
            for metric_name in selected:
                results[metric_name] = self._evaluators[metric_name].evaluate(
                    case
                )

            # Supplemental, compact result attributes on the open root span
            # (native Phoenix annotations, below, are the canonical record).
            for metric_name in selected:
                result = results[metric_name]
                tracing.annotate_current_span(
                    metric=result.metric,
                    score=result.score,
                    label=result.label,
                    explanation=result.explanation,
                    annotator_kind=self._evaluators[metric_name].annotator_kind,
                )
            span_id = handle.span_id
            trace_id = handle.trace_id

        # The root span is now closed/exported; persist native annotations
        # against its span id.
        records = [
            EvaluationRecord.from_result(
                results[name],
                annotator_kind=self._evaluators[name].annotator_kind,
                case_id=case.case_id,
                run_name=run_name,
                dataset_name=dataset_name,
                trace_id=trace_id,
                span_id=span_id,
            )
            for name in selected
        ]
        self._publish(records, results)

        return results

    def evaluate_batch(
        self,
        cases: list[EvaluationCase],
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
    ) -> list[dict[str, EvaluationResult]]:
        """Evaluates many cases, one trace each (no batch root span).

        ``run_name`` / ``dataset_name`` only group the per-case traces and rows
        via metadata.
        """
        return [
            self.evaluate(
                case,
                metrics=metrics,
                run_name=run_name,
                dataset_name=dataset_name,
            )
            for case in cases
        ]

    def log_evaluation(
        self,
        result: EvaluationResult,
        *,
        case: EvaluationCase | None = None,
        case_id: str | None = None,
        annotator_kind: str = "CODE",
        run_name: str | None = None,
        dataset_name: str | None = None,
    ) -> EvaluationResult:
        """Publishes an externally computed result through the same output layer.

        Lets callers log their own evaluations (deterministic Python checks,
        human review, etc.) without subclassing an evaluator.

        Args:
            result: The already-computed result to publish.
            case: Optional case, used only to resolve ``case_id``.
            case_id: Explicit case id (wins over ``case.case_id``).
            annotator_kind: One of ``"LLM"``, ``"CODE"``, ``"HUMAN"``.
            run_name: Optional run name metadata.
            dataset_name: Optional dataset name metadata.

        Returns:
            The same ``result``, for convenience.
        """
        validate_annotator_kind(annotator_kind)
        resolved_case_id = (
            case_id if case_id is not None
            else (case.case_id if case is not None else None)
        )
        with tracing.case_evaluation_span(
            resolved_case_id, run_name, dataset_name
        ) as handle:
            tracing.annotate_current_span(
                metric=result.metric,
                score=result.score,
                label=result.label,
                explanation=result.explanation,
                annotator_kind=annotator_kind,
            )
            span_id = handle.span_id
            trace_id = handle.trace_id

        record = EvaluationRecord.from_result(
            result,
            annotator_kind=annotator_kind,
            case_id=resolved_case_id,
            run_name=run_name,
            dataset_name=dataset_name,
            trace_id=trace_id,
            span_id=span_id,
        )
        self._publish([record], {result.metric: result})
        return result

    def log_custom_evaluation(
        self,
        name: str,
        score: float | None = None,
        label: str | None = None,
        explanation: str | None = None,
        details: dict | None = None,
        *,
        kind: str = "CODE",
        case: EvaluationCase | None = None,
        case_id: str | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
    ) -> EvaluationResult:
        """Convenience wrapper: builds an :class:`EvaluationResult` and logs it."""
        result = EvaluationResult(
            metric=name,
            score=score,
            label=label,
            explanation=explanation,
            details=details,
        )
        return self.log_evaluation(
            result,
            case=case,
            case_id=case_id,
            annotator_kind=kind,
            run_name=run_name,
            dataset_name=dataset_name,
        )

    def _publish(
        self,
        records: list[EvaluationRecord],
        results: dict[str, EvaluationResult],
    ) -> None:
        """Writes records to all configured writers, preserving results on error.

        Raises:
            PersistenceError: If any writer fails. Evaluation is never re-run.
        """
        if not self._writers:
            return
        try:
            for writer in self._writers:
                writer.write(records)
        except Exception as exc:  # noqa: BLE001 - re-raised with results attached
            raise PersistenceError(
                f"Failed to persist evaluation results: {exc}", results=results
            ) from exc
