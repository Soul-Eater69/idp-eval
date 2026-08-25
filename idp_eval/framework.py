"""The evaluation framework: orchestrates evaluators, tracing, and output.

The framework expands an ``EvaluationCase`` according to its output scope, owns
one trace per resulting logical evaluation, runs selected evaluators (whose real
judge calls become child spans), and publishes each result once to the configured
output(s). Evaluation is never re-run for a second destination.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Iterable, TypeAlias

from tqdm.auto import tqdm

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.operational_errors import classify_operational_error
from idp_eval.output import (
    EvaluationCheckpoint,
    EvaluationRecord,
    PersistenceError,
    build_writers,
    validate_report_fields,
    validate_annotator_kind,
)
from idp_eval.rendering import is_empty_value, render_value
from idp_eval.resume import (
    case_fingerprint,
    evaluation_fingerprint,
    external_evaluation_fingerprint,
    rendered_case_fields,
)

# An entry may be an evaluator class (constructed with the shared judge) or an
# already-constructed evaluator instance (backward-compatible).
EvaluatorEntry = type[Evaluator] | Evaluator

# Default global cap on simultaneous judge calls in an async evaluation run.
# This protects any configured backend from unbounded request fan-out.
DEFAULT_MAX_CONCURRENCY = 4

EvaluationResults: TypeAlias = dict[str, EvaluationResult]
ScopedEvaluationResults: TypeAlias = dict[
    str, EvaluationResults | list[EvaluationResults] | None
]
EvaluationReturn: TypeAlias = EvaluationResults | ScopedEvaluationResults


@dataclass
class _LogicalRun:
    """Internal result for one already-expanded evaluator-facing case."""

    results: EvaluationResults
    records: list[EvaluationRecord]
    resumed_metrics: int = 0
    evaluated_metrics: int = 0
    operational_errors: int = 0


@dataclass
class _CaseRun:
    """Internal result and operational counts for one original case."""

    results: EvaluationReturn
    resumed_metrics: int = 0
    evaluated_metrics: int = 0
    operational_errors: int = 0

    @property
    def fully_resumed(self) -> bool:
        return self.resumed_metrics > 0 and self.evaluated_metrics == 0


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
        resume: bool = False,
        report_fields: list[str] | tuple[str, ...] | None = None,
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
            resume: Reuse exact successful evaluations from ``excel_path`` and
                upsert rerun errors. Requires Excel output.
            report_fields: Ordered case-content fields shown in the visible
                Excel report. Reporting-only; does not affect evaluation or
                resume fingerprints.

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

        validated_report_fields = validate_report_fields(report_fields)
        self._writers = build_writers(
            output,
            excel_path,
            resume=resume,
            report_fields=validated_report_fields,
        )
        self._checkpoint = next(
            (
                writer
                for writer in self._writers
                if isinstance(writer, EvaluationCheckpoint)
            ),
            None,
        )
        self._resume = resume

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
                    f"{entry.__name__!r}; pass a configured judge."
                )
            try:
                return entry(llm=judge)
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise ValueError(
                    f"Failed to construct evaluator {entry.__name__!r}: {exc}"
                ) from exc

        if isinstance(entry, Evaluator):
            # Instances constructed without a judge (e.g. RelevanceAtKEvaluator(k=5)
            # needs its ``k`` at construction, so it is passed as an instance) can
            # receive the shared judge here. ``_bind_judge`` only binds when unset,
            # so an explicitly-provided judge always wins.
            if judge is not None:
                bind_judge = getattr(entry, "_bind_judge", None)
                if callable(bind_judge):
                    bind_judge(judge)
            return entry

        raise TypeError(
            "Evaluator entry must be an Evaluator subclass or instance, got "
            f"{entry!r}"
        )

    @property
    def metrics(self) -> list[str]:
        """Returns the names of all registered metrics."""
        return list(self._evaluators)

    def _select(self, metrics: list[str] | None) -> list[str]:
        """Resolves the metric subset to run against the configured evaluators.

        ``None`` selects all configured evaluators. A ``metrics`` list is an
        optional subset filter — it never instantiates unconfigured evaluators.

        Raises:
            KeyError: If a requested metric was not configured on the framework.
        """
        if metrics is None:
            return list(self._evaluators)
        for metric_name in metrics:
            if metric_name not in self._evaluators:
                raise KeyError(
                    f"Unknown metric {metric_name!r}; configured metrics: "
                    f"{sorted(self._evaluators)}"
                )
        return list(metrics)

    @staticmethod
    def _validate_max_concurrency(max_concurrency: int) -> None:
        """Rejects a non-positive-integer concurrency limit with a clear error."""
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError(
                "max_concurrency must be a positive integer, got "
                f"{max_concurrency!r}."
            )

    @staticmethod
    def _validate_on_error(on_error: str) -> None:
        if on_error not in ("raise", "continue"):
            raise ValueError(
                f"Unknown on_error {on_error!r}; expected 'raise' or 'continue'."
            )

    def _retrieval_metrics(self, selected: list[str]) -> list[str]:
        """Selected metric names that share the per-case document-relevance pass."""
        return [
            name
            for name in selected
            if getattr(self._evaluators[name], "uses_retrieval_relevance", False)
        ]

    def _build_relevance_pass(self, case: EvaluationCase, retrieval: list[str]):
        """Builds the shared relevance pass and the deepest rank it must judge.

        Returns ``(pass, max_depth)``. The lead retrieval evaluator provides the
        relevance judge (retrieval metrics are configured with the same judge);
        ``max_depth`` is the deepest ``effective_k`` across the selected retrieval
        metrics, so documents are judged once up to that rank and reused.
        """
        lead = self._evaluators[retrieval[0]]
        relevance_pass = lead._build_relevance_pass(case)
        max_depth = max(
            self._evaluators[name]._relevance_depth(case) for name in retrieval
        )
        return relevance_pass, max_depth

    def _validate_case(self, case: EvaluationCase, selected: list[str]) -> None:
        """Level 2 validation: every selected evaluator's required fields.

        Only the selected evaluators are checked, so a case missing a field an
        unselected evaluator would need is still valid for this call.
        """
        for metric_name in selected:
            self._evaluators[metric_name].validate_case(case)

    def evaluate(
        self,
        case: EvaluationCase,
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        on_error: str = "raise",
    ) -> EvaluationReturn:
        """Evaluates a case using its combined/individual/both output scope.

        Args:
            case: The case to evaluate.
            metrics: Metric names to run. Runs all registered metrics when
                ``None``.
            run_name: Optional benchmark run name, attached to the trace and
                every published row.
            dataset_name: Optional dataset name, attached likewise.
            on_error: ``"raise"`` preserves fail-fast behavior. ``"continue"``
                converts only recognized operational failures into metric error
                results.

        Returns:
            Combined scope returns the backward-compatible metric mapping.
            Individual/both scopes return a mapping with ``combined`` and
            ``individual`` entries.

        Raises:
            KeyError: If a requested metric name is not registered.
            ValueError: If the case is missing a field a selected evaluator
                requires (raised before any judge call).
            PersistenceError: If evaluation succeeds but a configured writer
                fails; the computed results are preserved on the exception.
        """
        self._validate_on_error(on_error)
        selected = self._select(metrics)
        scope, planned_cases = self._scope_plan(case)
        for planned_case in planned_cases:
            self._validate_case(planned_case, selected)

        return self._run_original_sync(
            scope,
            planned_cases,
            selected,
            run_name=run_name,
            dataset_name=dataset_name,
            on_error=on_error,
        ).results

    def _run_original_sync(
        self,
        scope: str,
        planned_cases: list[EvaluationCase],
        selected: list[str],
        *,
        run_name: str | None,
        dataset_name: str | None,
        on_error: str,
    ) -> _CaseRun:
        """Runs and immediately persists each logical part of one input case."""
        logical_runs: list[_LogicalRun] = []
        for planned_case in planned_cases:
            run = self._run_one(
                planned_case,
                selected,
                run_name=run_name,
                dataset_name=dataset_name,
                on_error=on_error,
            )
            self._publish(run.records, run.results)
            logical_runs.append(run)
        return _CaseRun(
            results=self._shape_scope_results(
                scope, [run.results for run in logical_runs]
            ),
            resumed_metrics=sum(run.resumed_metrics for run in logical_runs),
            evaluated_metrics=sum(run.evaluated_metrics for run in logical_runs),
            operational_errors=sum(
                run.operational_errors for run in logical_runs
            ),
        )

    def _run_one(
        self,
        case: EvaluationCase,
        selected: list[str],
        *,
        run_name: str | None,
        dataset_name: str | None,
        on_error: str,
    ) -> _LogicalRun:
        """Runs one expanded case, reusing exact successful checkpoints."""

        resumed = self._load_resumed_results(
            case, selected, run_name, dataset_name
        )
        to_run = [name for name in selected if name not in resumed]
        if selected and not to_run:
            return _LogicalRun(
                results={name: resumed[name] for name in selected},
                records=[],
                resumed_metrics=len(resumed),
            )

        with tracing.case_evaluation_span(
            case.case_id,
            run_name,
            dataset_name,
            self._descriptive_input(case),
        ) as handle:
            # One batched relevance call for all selected retrieval metrics,
            # judged once through the deepest requested effective K.
            retrieval = self._retrieval_metrics(to_run)
            relevance_pass = None
            new_results: dict[str, EvaluationResult] = {}
            if retrieval:
                relevance_pass, max_depth = self._build_relevance_pass(
                    case, retrieval
                )
                try:
                    relevance_pass.run(max_depth)
                except Exception as exc:
                    error = self._operational_results(
                        retrieval, exc, on_error=on_error
                    )
                    new_results.update(error)
                    relevance_pass = None

            for metric_name in to_run:
                if metric_name in new_results:
                    continue
                evaluator = self._evaluators[metric_name]
                try:
                    if relevance_pass is not None and getattr(
                        evaluator, "uses_retrieval_relevance", False
                    ):
                        new_results[metric_name] = evaluator.evaluate_shared(
                            case, relevance_pass
                        )
                    else:
                        new_results[metric_name] = evaluator.evaluate(case)
                except Exception as exc:
                    new_results.update(
                        self._operational_results(
                            [metric_name], exc, on_error=on_error
                        )
                    )

            # Supplemental, compact result attributes on the open root span
            # (native Phoenix annotations, below, are the canonical record).
            for metric_name in to_run:
                result = new_results[metric_name]
                tracing.annotate_current_span(
                    metric=result.metric,
                    score=result.score,
                    label=result.label,
                    explanation=result.explanation,
                    annotator_kind=self._evaluators[metric_name].annotator_kind,
                )
            span_id = handle.span_id
            trace_id = handle.trace_id

        results = {
            name: resumed[name] if name in resumed else new_results[name]
            for name in selected
        }
        records = self._build_records(
            case,
            to_run,
            new_results,
            run_name,
            dataset_name,
            trace_id,
            span_id,
        )
        return _LogicalRun(
            results=results,
            records=records,
            resumed_metrics=len(resumed),
            evaluated_metrics=len(to_run),
            operational_errors=sum(
                result.label == "error" for result in new_results.values()
            ),
        )

    def _operational_results(
        self, metrics: list[str], exc: Exception, *, on_error: str
    ) -> dict[str, EvaluationResult]:
        """Converts only known operational failures when explicitly requested."""
        info = classify_operational_error(exc)
        if on_error == "raise" or info is None:
            raise exc
        return {
            metric: EvaluationResult(
                metric=metric,
                score=None,
                label="error",
                explanation=info.explanation,
                details=dict(info.details()),
            )
            for metric in metrics
        }

    def _evaluation_fingerprints(
        self, case: EvaluationCase, selected: list[str]
    ) -> tuple[str, dict[str, str]]:
        case_hash = case_fingerprint(case)
        return case_hash, {
            name: evaluation_fingerprint(
                case_hash, name, self._evaluators[name]
            )
            for name in selected
        }

    def _load_resumed_results(
        self,
        case: EvaluationCase,
        selected: list[str],
        run_name: str | None,
        dataset_name: str | None,
    ) -> EvaluationResults:
        if not self._resume or self._checkpoint is None:
            return {}
        _, fingerprints = self._evaluation_fingerprints(case, selected)
        results: EvaluationResults = {}
        for name in selected:
            result = self._checkpoint.load_successful_result(
                run_name=run_name,
                dataset_name=dataset_name,
                case_id=case.case_id,
                evaluation_fingerprint=fingerprints[name],
                metric=name,
            )
            if result is not None:
                results[name] = result
        return results

    @staticmethod
    def _scope_plan(case: EvaluationCase) -> tuple[str, list[EvaluationCase]]:
        """Expands one scoped case into ordinary evaluator-facing cases."""
        scope = case.evaluation_scope
        if scope == "combined":
            return scope, [case]
        if not isinstance(case.output, list) or not case.output:
            raise ValueError(
                f"evaluation_scope={scope!r} requires output to be a non-empty "
                "list."
            )

        individual = [
            EvaluationFramework._copy_case(
                case,
                output=output,
                case_id=(
                    f"{case.case_id}:{index}"
                    if case.case_id is not None
                    else None
                ),
            )
            for index, output in enumerate(case.output)
        ]
        if scope == "individual":
            return scope, individual
        combined = EvaluationFramework._copy_case(
            case, output=case.output, case_id=case.case_id
        )
        return scope, [combined, *individual]

    @staticmethod
    def _copy_case(
        case: EvaluationCase, *, output, case_id: str | None
    ) -> EvaluationCase:
        """Copies shared fields into an evaluator-facing combined case."""
        return EvaluationCase(
            input=case.input,
            context=case.context,
            output=output,
            instructions=case.instructions,
            case_id=case_id,
            metadata=dict(case.metadata) if case.metadata is not None else None,
            retrieved_documents=case.retrieved_documents,
            evaluation_scope="combined",
        )

    @staticmethod
    def _shape_scope_results(
        scope: str, results: list[EvaluationResults]
    ) -> EvaluationReturn:
        """Builds the public backward-compatible or scoped return shape."""
        if scope == "combined":
            return results[0]
        if scope == "individual":
            return {"combined": None, "individual": results}
        return {"combined": results[0], "individual": results[1:]}

    def _build_records(
        self,
        case: EvaluationCase,
        selected: list[str],
        results: dict[str, EvaluationResult],
        run_name: str | None,
        dataset_name: str | None,
        trace_id: str | None,
        span_id: str | None,
    ) -> list[EvaluationRecord]:
        """Normalizes one case's results into publishable records (in order)."""
        case_hash, fingerprints = self._evaluation_fingerprints(case, selected)
        audit = rendered_case_fields(case)
        return [
            EvaluationRecord.from_result(
                results[name],
                annotator_kind=self._evaluators[name].annotator_kind,
                status=(
                    "error"
                    if results[name].label == "error"
                    and results[name].details
                    and results[name].details.get("status") == "error"
                    else "success"
                ),
                case_id=case.case_id,
                input=audit["input"],
                run_name=run_name,
                dataset_name=dataset_name,
                case_fingerprint=case_hash,
                evaluation_fingerprint=fingerprints[name],
                context=audit["context"],
                output=audit["output"],
                instructions=audit["instructions"],
                evaluation_scope=audit["evaluation_scope"],
                retrieved_documents_json=audit["retrieved_documents_json"],
                retrieved_documents=audit["retrieved_documents"],
                metadata=(dict(case.metadata) if case.metadata is not None else None),
                trace_id=trace_id,
                span_id=span_id,
            )
            for name in selected
        ]

    @staticmethod
    def _descriptive_input(case: EvaluationCase) -> str | None:
        """Renders optional case input for traces/reports, never metric prompts."""
        if is_empty_value(case.input):
            return None
        return render_value(case.input)

    @staticmethod
    def _progress_case_label(case: EvaluationCase, index: int) -> str:
        """Returns a safe progress identifier without rendering case content."""
        return case.case_id if case.case_id is not None else f"index {index}"

    @staticmethod
    def _progress_metric_results(
        results: EvaluationReturn, metric: str
    ) -> list[EvaluationResult]:
        """Collects one metric's results from combined or scoped output."""
        direct = results.get(metric)
        if isinstance(direct, EvaluationResult):
            return [direct]

        collected: list[EvaluationResult] = []
        combined = results.get("combined")
        if isinstance(combined, dict):
            combined_result = combined.get(metric)
            if isinstance(combined_result, EvaluationResult):
                collected.append(combined_result)
        individual = results.get("individual")
        if isinstance(individual, list):
            for item_results in individual:
                item_result = item_results.get(metric)
                if isinstance(item_result, EvaluationResult):
                    collected.append(item_result)
        return collected

    @staticmethod
    def _progress_score(result: EvaluationResult) -> str:
        """Formats a normalized result compactly, including not-applicable."""
        if result.label == "error":
            error_type = (
                result.details.get("error_type")
                if isinstance(result.details, dict)
                else None
            )
            return f"ERROR ({error_type or 'operational'})"
        score = "None" if result.score is None else f"{result.score:.3f}"
        label = "None" if result.label is None else result.label
        return f"{score} ({label})"

    @classmethod
    def _progress_metric_summary(
        cls, results: EvaluationReturn, selected: list[str]
    ) -> str:
        """Formats every selected metric without exposing evaluator payloads."""
        summaries: list[str] = []
        for metric in selected:
            values = [
                cls._progress_score(result)
                for result in cls._progress_metric_results(results, metric)
            ]
            rendered = values[0] if len(values) == 1 else f"[{'; '.join(values)}]"
            summaries.append(f"{metric}={rendered}")
        return " | ".join(summaries)

    @staticmethod
    def _progress_error(exc: Exception) -> str:
        """Builds a bounded error summary and suppresses likely secret text."""
        error_type = type(exc).__name__
        message = " ".join(str(exc).split())
        sensitive_markers = (
            "authorization",
            "bearer ",
            "api key",
            "api_key",
            "client_secret",
            "password",
            "token",
        )
        if not message or any(
            marker in message.lower() for marker in sensitive_markers
        ):
            return error_type
        if len(message) > 160:
            message = f"{message[:157]}..."
        return f"{error_type}: {message}"

    @staticmethod
    def _progress_elapsed(seconds: float) -> str:
        """Formats total wall time for the final progress summary."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        rounded = int(round(seconds))
        minutes, seconds_part = divmod(rounded, 60)
        hours, minutes_part = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes_part}m {seconds_part}s"
        return f"{minutes_part}m {seconds_part}s"

    @classmethod
    def _write_progress_success(
        cls,
        progress,
        *,
        completed: int,
        total: int,
        case: EvaluationCase,
        index: int,
        duration: float,
        results: EvaluationReturn,
        selected: list[str],
        fully_resumed: bool = False,
        operational_errors: int = 0,
    ) -> None:
        """Writes one case completion line without corrupting the bar."""
        metrics = cls._progress_metric_summary(results, selected)
        suffix = f" | {metrics}" if metrics else ""
        if fully_resumed:
            symbol = "↷"
            state = "resumed from checkpoint"
        elif operational_errors:
            symbol = "⚠"
            state = "completed with operational error"
        else:
            symbol = "✓"
            state = "completed"
        tqdm.write(
            f"{symbol} [{completed}/{total}] "
            f"case={cls._progress_case_label(case, index)} "
            f"{state} in {duration:.1f}s{suffix}",
            file=progress.fp,
        )

    @classmethod
    def _write_progress_failure(
        cls,
        progress,
        *,
        position: int,
        total: int,
        case: EvaluationCase,
        index: int,
        duration: float,
        exc: Exception,
    ) -> None:
        """Writes one bounded failure line before preserving the exception."""
        tqdm.write(
            f"✗ [{position}/{total}] "
            f"case={cls._progress_case_label(case, index)} "
            f"failed after {duration:.1f}s: {cls._progress_error(exc)}",
            file=progress.fp,
        )

    def evaluate_many(
        self,
        cases: "Iterable[EvaluationCase]",
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        show_progress: bool = False,
        on_error: str = "raise",
    ) -> list[EvaluationReturn]:
        """Evaluates many cases independently, honoring each output scope.

        Each expanded logical evaluation keeps its own root
        ``idp_eval.evaluate`` trace, metric results, and output rows/annotations.
        There is no batch-level root span; ``run_name`` / ``dataset_name`` only
        group the traces and rows via metadata.

        Failure behavior is **fail fast**: the whole batch is validated for the
        selected evaluators *before any judge call*, so a malformed later case
        never causes paid work on earlier cases. Invalid cases raise rather than
        being silently skipped or coerced to not-applicable.

        Returns:
            One result mapping per case, in input order.

        Args:
            show_progress: When true, display one tqdm progress update and a
                compact metric summary per original input case.

        Raises:
            KeyError: If a requested metric was not configured.
            ValueError: If any case is missing a required field for a selected
                evaluator; the message names the offending ``case_id`` (or index).
        """
        case_list = list(cases)
        self._validate_on_error(on_error)
        selected = self._select(metrics)

        # Pre-validate the entire batch before any judge work (fail fast).
        plans: list[tuple[str, list[EvaluationCase]]] = []
        for index, case in enumerate(case_list):
            try:
                plan = self._scope_plan(case)
                planned_cases = plan[1]
                for planned_case in planned_cases:
                    self._validate_case(planned_case, selected)
                plans.append(plan)
            except ValueError as exc:
                label = case.case_id if case.case_id is not None else f"index {index}"
                raise ValueError(f"Case {label}: {exc}") from exc

        if not show_progress:
            return [
                self._run_original_sync(
                    scope,
                    planned_cases,
                    selected,
                    run_name=run_name,
                    dataset_name=dataset_name,
                    on_error=on_error,
                ).results
                for scope, planned_cases in plans
            ]

        total = len(case_list)
        batch_started = time.perf_counter()
        progress = tqdm(total=total, desc="Evaluating cases", unit="case")
        results_list: list[EvaluationReturn] = []
        resumed_cases = 0
        operational_error_cases = 0
        try:
            for index, (case, (scope, planned_cases)) in enumerate(
                zip(case_list, plans)
            ):
                case_started = time.perf_counter()
                try:
                    run = self._run_original_sync(
                        scope,
                        planned_cases,
                        selected,
                        run_name=run_name,
                        dataset_name=dataset_name,
                        on_error=on_error,
                    )
                except Exception as exc:
                    self._write_progress_failure(
                        progress,
                        position=int(progress.n) + 1,
                        total=total,
                        case=case,
                        index=index,
                        duration=time.perf_counter() - case_started,
                        exc=exc,
                    )
                    raise
                results_list.append(run.results)
                resumed_cases += int(run.fully_resumed)
                operational_error_cases += int(run.operational_errors > 0)
                progress.update(1)
                self._write_progress_success(
                    progress,
                    completed=int(progress.n),
                    total=total,
                    case=case,
                    index=index,
                    duration=time.perf_counter() - case_started,
                    results=run.results,
                    selected=selected,
                    fully_resumed=run.fully_resumed,
                    operational_errors=run.operational_errors,
                )
        finally:
            progress.close()

        tqdm.write(
            f"Evaluation complete: {total}/{total} cases in "
            f"{self._progress_elapsed(time.perf_counter() - batch_started)} | "
            f"completed={total - resumed_cases} | resumed={resumed_cases} | "
            f"operational_errors={operational_error_cases}",
            file=progress.fp,
        )
        return results_list

    def evaluate_groups(
        self,
        groups: "Iterable[dict]",
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        on_error: str = "raise",
    ) -> list[dict[str, EvaluationResult]]:
        """Convenience orchestration: fan grouped outputs into single cases.

        Each group is a mapping with an ``outputs`` list plus optional shared
        ``input`` / ``context`` / ``instructions`` / ``metadata``, an optional
        ``group_id``, and optional ``case_ids`` aligned with ``outputs``. Every
        output becomes one ordinary :class:`EvaluationCase`; the resulting cases
        are passed to :meth:`evaluate_many`. A list inside one output remains
        part of that single structured output.

        Case ids: ``case_ids[i]`` if given, else ``f"{group_id}:{i}"`` when a
        ``group_id`` is present, else ``f"{group_index}:{i}"``. ``group_id`` is
        carried on ``case.metadata`` and is never injected into prompts. Input
        groups and output objects are not mutated.

        Returns:
            One result mapping per fanned-out case, in group-then-output order.
        """
        cases: list[EvaluationCase] = []
        for group_index, group in enumerate(groups):
            cases.extend(self._fan_out_group(group, group_index))
        return self.evaluate_many(
            cases,
            metrics=metrics,
            run_name=run_name,
            dataset_name=dataset_name,
            on_error=on_error,
        )

    async def a_evaluate_groups(
        self,
        groups: "Iterable[dict]",
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        on_error: str = "raise",
    ) -> list[dict[str, EvaluationResult]]:
        """Fans grouped outputs into cases, then evaluates them concurrently.

        Every output is an ordinary independent case with its own root trace and
        judge call. The framework-level semaphore bounds all concurrent judge
        work, and results remain in group-then-output order.
        """
        cases: list[EvaluationCase] = []
        for group_index, group in enumerate(groups):
            cases.extend(self._fan_out_group(group, group_index))
        return await self.a_evaluate_many(
            cases,
            metrics=metrics,
            run_name=run_name,
            dataset_name=dataset_name,
            max_concurrency=max_concurrency,
            on_error=on_error,
        )

    @staticmethod
    def _fan_out_group(group: dict, group_index: int) -> list[EvaluationCase]:
        """Expands one grouped record into ordinary per-output cases."""
        if "output" in group and "outputs" not in group:
            raise ValueError(
                "Grouped records use 'outputs' (a list); got singular 'output'. "
                "Use evaluate_many for individual cases."
            )
        outputs = group.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("Each group must provide a non-empty 'outputs' list.")

        group_id = group.get("group_id")
        case_ids = group.get("case_ids")
        if case_ids is not None and (
            not isinstance(case_ids, list) or len(case_ids) != len(outputs)
        ):
            raise ValueError(
                "Group 'case_ids' length must match the number of 'outputs'."
            )

        shared = {
            field: group[field]
            for field in ("input", "context", "instructions")
            if field in group
        }
        source_metadata = group.get("metadata")
        if source_metadata is not None and not isinstance(source_metadata, dict):
            raise ValueError("Group 'metadata' must be a mapping when provided.")
        metadata = dict(source_metadata or {})
        if group_id is not None:
            metadata["group_id"] = group_id

        cases: list[EvaluationCase] = []
        for output_index, output in enumerate(outputs):
            if case_ids is not None:
                case_id = case_ids[output_index]
            elif group_id is not None:
                case_id = f"{group_id}:{output_index}"
            else:
                case_id = f"{group_index}:{output_index}"
            cases.append(
                EvaluationCase(
                    output=output,
                    case_id=case_id,
                    metadata=dict(metadata) if metadata else None,
                    **shared,
                )
            )
        return cases

    # --- async evaluation ---------------------------------------------------

    async def a_evaluate(
        self,
        case: EvaluationCase,
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        on_error: str = "raise",
    ) -> EvaluationReturn:
        """Async counterpart of :meth:`evaluate`, including output scope.

        Judge calls are bounded by a per-call global concurrency limiter.
        Evaluators without an async path run their synchronous ``evaluate`` in a
        worker thread under the same limiter. Validation still runs before any
        judge call.
        """
        self._validate_max_concurrency(max_concurrency)
        self._validate_on_error(on_error)
        selected = self._select(metrics)
        scope, planned_cases = self._scope_plan(case)
        for planned_case in planned_cases:
            self._validate_case(planned_case, selected)
        limiter = asyncio.Semaphore(max_concurrency)
        publish_lock = asyncio.Lock()
        runs = await asyncio.gather(
            *(
                self._a_run_and_publish(
                    planned_case,
                    selected,
                    run_name,
                    dataset_name,
                    limiter,
                    on_error,
                    publish_lock,
                )
                for planned_case in planned_cases
            )
        )
        return self._shape_scope_results(scope, [run.results for run in runs])

    async def a_evaluate_many(
        self,
        cases: "Iterable[EvaluationCase]",
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        show_progress: bool = False,
        on_error: str = "raise",
    ) -> list[EvaluationReturn]:
        """Async counterpart of :meth:`evaluate_many`: cases run concurrently.

        Each expanded logical evaluation is independent (own validation, root
        ``idp_eval.evaluate`` trace, and results/rows/annotations). A single shared
        ``asyncio.Semaphore(max_concurrency)`` caps total simultaneous judge calls
        across every case so a large batch never floods the configured backend.

        Failure behavior mirrors :meth:`evaluate_many`: the whole batch is
        validated up front (fail fast, case-aware error). Results are returned in
        input order; each completed logical case is persisted immediately through
        one serialized publisher so concurrent tasks never mutate writers at the
        same time. When ``show_progress`` is true, progress advances as original
        input cases finish while return ordering remains unchanged.
        """
        self._validate_max_concurrency(max_concurrency)
        self._validate_on_error(on_error)
        case_list = list(cases)
        selected = self._select(metrics)

        plans: list[tuple[str, list[EvaluationCase]]] = []
        for index, case in enumerate(case_list):
            try:
                plan = self._scope_plan(case)
                for planned_case in plan[1]:
                    self._validate_case(planned_case, selected)
                plans.append(plan)
            except ValueError as exc:
                label = case.case_id if case.case_id is not None else f"index {index}"
                raise ValueError(f"Case {label}: {exc}") from exc

        limiter = asyncio.Semaphore(max_concurrency)
        publish_lock = asyncio.Lock()
        flat_cases = [case for _, cases in plans for case in cases]

        async def run_and_publish(planned_case: EvaluationCase) -> _LogicalRun:
            return await self._a_run_and_publish(
                planned_case,
                selected,
                run_name,
                dataset_name,
                limiter,
                on_error,
                publish_lock,
            )

        if not show_progress:
            runs = await asyncio.gather(
                *(run_and_publish(case) for case in flat_cases)
            )
        else:
            total = len(case_list)
            batch_started = time.perf_counter()
            progress = tqdm(total=total, desc="Evaluating cases", unit="case")
            output_file = progress.fp
            case_started = [time.perf_counter() for _ in case_list]
            case_offsets: list[int] = []
            owner_indices: list[int] = []
            remaining: list[int] = []
            offset = 0
            for owner_index, (_, planned_cases) in enumerate(plans):
                case_offsets.append(offset)
                count = len(planned_cases)
                remaining.append(count)
                owner_indices.extend([owner_index] * count)
                offset += count

            stored_runs: list[_LogicalRun | None] = [None] * len(flat_cases)
            failure_reported = [False] * total

            async def run_with_progress(
                flat_index: int,
                owner_index: int,
                planned_case: EvaluationCase,
            ) -> None:
                try:
                    run = await run_and_publish(planned_case)
                except Exception as exc:
                    if not failure_reported[owner_index]:
                        failure_reported[owner_index] = True
                        self._write_progress_failure(
                            progress,
                            position=int(progress.n) + 1,
                            total=total,
                            case=case_list[owner_index],
                            index=owner_index,
                            duration=(
                                time.perf_counter() - case_started[owner_index]
                            ),
                            exc=exc,
                        )
                    raise

                stored_runs[flat_index] = run
                remaining[owner_index] -= 1
                if remaining[owner_index] == 0:
                    scope, planned_cases = plans[owner_index]
                    start = case_offsets[owner_index]
                    scoped_runs: list[_LogicalRun] = []
                    for scoped_index in range(start, start + len(planned_cases)):
                        stored_run = stored_runs[scoped_index]
                        assert stored_run is not None
                        scoped_runs.append(stored_run)
                    shaped = self._shape_scope_results(
                        scope, [run.results for run in scoped_runs]
                    )
                    progress.update(1)
                    self._write_progress_success(
                        progress,
                        completed=int(progress.n),
                        total=total,
                        case=case_list[owner_index],
                        index=owner_index,
                        duration=time.perf_counter() - case_started[owner_index],
                        results=shaped,
                        selected=selected,
                        fully_resumed=(
                            sum(run.resumed_metrics for run in scoped_runs) > 0
                            and sum(run.evaluated_metrics for run in scoped_runs)
                            == 0
                        ),
                        operational_errors=sum(
                            run.operational_errors for run in scoped_runs
                        ),
                    )

            try:
                await asyncio.gather(
                    *(
                        run_with_progress(flat_index, owner_index, case)
                        for flat_index, (owner_index, case) in enumerate(
                            zip(owner_indices, flat_cases)
                        )
                    )
                )
            finally:
                progress.close()

            runs = []
            for stored_run in stored_runs:
                assert stored_run is not None
                runs.append(stored_run)

        flat_results = [run.results for run in runs]

        results_list: list[EvaluationReturn] = []
        offset = 0
        for scope, planned_cases in plans:
            count = len(planned_cases)
            results_list.append(
                self._shape_scope_results(
                    scope, flat_results[offset : offset + count]
                )
            )
            offset += count
        if show_progress:
            resumed_cases = 0
            operational_error_cases = 0
            offset = 0
            for _, planned_cases in plans:
                scoped_runs = runs[offset : offset + len(planned_cases)]
                resumed_cases += int(
                    sum(run.resumed_metrics for run in scoped_runs) > 0
                    and sum(run.evaluated_metrics for run in scoped_runs) == 0
                )
                operational_error_cases += int(
                    any(run.operational_errors for run in scoped_runs)
                )
                offset += len(planned_cases)
            tqdm.write(
                f"Evaluation complete: {len(case_list)}/{len(case_list)} cases in "
                f"{self._progress_elapsed(time.perf_counter() - batch_started)} | "
                f"completed={len(case_list) - resumed_cases} | "
                f"resumed={resumed_cases} | "
                f"operational_errors={operational_error_cases}",
                file=output_file,
            )
        return results_list

    async def _a_run_and_publish(
        self,
        case: EvaluationCase,
        selected: list[str],
        run_name: str | None,
        dataset_name: str | None,
        limiter: asyncio.Semaphore,
        on_error: str,
        publish_lock: asyncio.Lock,
    ) -> _LogicalRun:
        """Runs one logical case and checkpoints it before returning."""
        run = await self._a_run_case(
            case,
            selected,
            run_name,
            dataset_name,
            limiter,
            on_error,
        )
        async with publish_lock:
            self._publish(run.records, run.results)
        return run

    async def _a_run_case(
        self,
        case: EvaluationCase,
        selected: list[str],
        run_name: str | None,
        dataset_name: str | None,
        limiter: asyncio.Semaphore,
        on_error: str,
    ) -> _LogicalRun:
        """Runs one case's selected evaluators within its own root trace.

        Concurrent cases each get an independent root span/trace because each is
        a separate task with its own OpenTelemetry context.
        """
        resumed = self._load_resumed_results(
            case, selected, run_name, dataset_name
        )
        to_run = [name for name in selected if name not in resumed]
        if selected and not to_run:
            return _LogicalRun(
                results={name: resumed[name] for name in selected},
                records=[],
                resumed_metrics=len(resumed),
            )

        with tracing.case_evaluation_span(
            case.case_id,
            run_name,
            dataset_name,
            self._descriptive_input(case),
        ) as handle:
            # One batched relevance call for all selected retrieval metrics. The
            # complete call consumes one slot from the shared judge limiter.
            retrieval = self._retrieval_metrics(to_run)
            relevance_pass = None
            new_results: dict[str, EvaluationResult] = {}
            if retrieval:
                relevance_pass, max_depth = self._build_relevance_pass(
                    case, retrieval
                )
                try:
                    await relevance_pass.a_run(max_depth, limiter)
                except Exception as exc:
                    new_results.update(
                        self._operational_results(
                            retrieval, exc, on_error=on_error
                        )
                    )
                    relevance_pass = None

            for metric_name in to_run:
                if metric_name in new_results:
                    continue
                evaluator = self._evaluators[metric_name]
                try:
                    if relevance_pass is not None and getattr(
                        evaluator, "uses_retrieval_relevance", False
                    ):
                        new_results[metric_name] = evaluator.evaluate_shared(
                            case, relevance_pass
                        )
                        continue
                    a_evaluate = getattr(evaluator, "a_evaluate", None)
                    if a_evaluate is not None:
                        new_results[metric_name] = await a_evaluate(
                            case, judge_limiter=limiter
                        )
                    else:
                        async with limiter:
                            new_results[metric_name] = await asyncio.to_thread(
                                evaluator.evaluate, case
                            )
                except Exception as exc:
                    new_results.update(
                        self._operational_results(
                            [metric_name], exc, on_error=on_error
                        )
                    )

            for metric_name in to_run:
                result = new_results[metric_name]
                tracing.annotate_current_span(
                    metric=result.metric,
                    score=result.score,
                    label=result.label,
                    explanation=result.explanation,
                    annotator_kind=self._evaluators[metric_name].annotator_kind,
                )
            span_id = handle.span_id
            trace_id = handle.trace_id

        results = {
            name: resumed[name] if name in resumed else new_results[name]
            for name in selected
        }
        records = self._build_records(
            case,
            to_run,
            new_results,
            run_name,
            dataset_name,
            trace_id,
            span_id,
        )
        return _LogicalRun(
            results=results,
            records=records,
            resumed_metrics=len(resumed),
            evaluated_metrics=len(to_run),
            operational_errors=sum(
                result.label == "error" for result in new_results.values()
            ),
        )

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
        record_case = case or EvaluationCase(case_id=resolved_case_id)
        audit = rendered_case_fields(record_case)
        case_hash = case_fingerprint(record_case)
        result_status = (
            "error"
            if result.label == "error"
            and result.details
            and result.details.get("status") == "error"
            else "success"
        )
        with tracing.case_evaluation_span(
            resolved_case_id,
            run_name,
            dataset_name,
            self._descriptive_input(case) if case is not None else None,
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
            status=result_status,
            case_id=resolved_case_id,
            input=audit["input"],
            run_name=run_name,
            dataset_name=dataset_name,
            case_fingerprint=case_hash,
            evaluation_fingerprint=external_evaluation_fingerprint(
                case_hash, result.metric, annotator_kind
            ),
            context=audit["context"],
            output=audit["output"],
            instructions=audit["instructions"],
            evaluation_scope=audit["evaluation_scope"],
            retrieved_documents_json=audit["retrieved_documents_json"],
            retrieved_documents=audit["retrieved_documents"],
            metadata=(
                dict(record_case.metadata)
                if record_case.metadata is not None
                else None
            ),
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
        if not self._writers or not records:
            return
        try:
            for writer in self._writers:
                writer.write(records)
        except Exception as exc:  # noqa: BLE001 - re-raised with results attached
            raise PersistenceError(
                f"Failed to persist evaluation results: {exc}", results=results
            ) from exc
