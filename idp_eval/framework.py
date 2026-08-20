"""The evaluation framework: orchestrates evaluators, tracing, and output.

The framework expands an ``EvaluationCase`` according to its output scope, owns
one trace per resulting logical evaluation, runs selected evaluators (whose real
judge calls become child spans), and publishes each result once to the configured
output(s). Evaluation is never re-run for a second destination.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, TypeAlias

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.output import (
    EvaluationRecord,
    PersistenceError,
    build_writers,
    validate_annotator_kind,
)
from idp_eval.rendering import is_empty_value, render_value

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
    ) -> EvaluationReturn:
        """Evaluates a case using its combined/individual/both output scope.

        Args:
            case: The case to evaluate.
            metrics: Metric names to run. Runs all registered metrics when
                ``None``.
            run_name: Optional benchmark run name, attached to the trace and
                every published row.
            dataset_name: Optional dataset name, attached likewise.

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
        selected = self._select(metrics)
        scope, planned_cases = self._scope_plan(case)
        for planned_case in planned_cases:
            self._validate_case(planned_case, selected)

        results = [
            self._evaluate_one(
                planned_case,
                selected,
                run_name=run_name,
                dataset_name=dataset_name,
            )
            for planned_case in planned_cases
        ]
        return self._shape_scope_results(scope, results)

    def _evaluate_one(
        self,
        case: EvaluationCase,
        selected: list[str],
        *,
        run_name: str | None,
        dataset_name: str | None,
    ) -> EvaluationResults:
        """Runs one already-expanded ordinary case with one root trace."""

        with tracing.case_evaluation_span(
            case.case_id,
            run_name,
            dataset_name,
            self._descriptive_input(case),
        ) as handle:
            # One shared document-relevance pass for all selected retrieval
            # metrics (judged once up to the deepest required rank).
            retrieval = self._retrieval_metrics(selected)
            relevance_pass = None
            if retrieval:
                relevance_pass, max_depth = self._build_relevance_pass(
                    case, retrieval
                )
                relevance_pass.run(max_depth)

            results: dict[str, EvaluationResult] = {}
            for metric_name in selected:
                evaluator = self._evaluators[metric_name]
                if relevance_pass is not None and getattr(
                    evaluator, "uses_retrieval_relevance", False
                ):
                    results[metric_name] = evaluator.evaluate_shared(
                        case, relevance_pass
                    )
                else:
                    results[metric_name] = evaluator.evaluate(case)

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
        records = self._build_records(
            case, selected, results, run_name, dataset_name, trace_id, span_id
        )
        self._publish(records, results)

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
        descriptive_input = self._descriptive_input(case)
        return [
            EvaluationRecord.from_result(
                results[name],
                annotator_kind=self._evaluators[name].annotator_kind,
                case_id=case.case_id,
                input=descriptive_input,
                run_name=run_name,
                dataset_name=dataset_name,
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

    def evaluate_many(
        self,
        cases: "Iterable[EvaluationCase]",
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
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

        Raises:
            KeyError: If a requested metric was not configured.
            ValueError: If any case is missing a required field for a selected
                evaluator; the message names the offending ``case_id`` (or index).
        """
        case_list = list(cases)
        selected = self._select(metrics)

        # Pre-validate the entire batch before any judge work (fail fast).
        for index, case in enumerate(case_list):
            try:
                _, planned_cases = self._scope_plan(case)
                for planned_case in planned_cases:
                    self._validate_case(planned_case, selected)
            except ValueError as exc:
                label = case.case_id if case.case_id is not None else f"index {index}"
                raise ValueError(f"Case {label}: {exc}") from exc

        return [
            self.evaluate(
                case,
                metrics=metrics,
                run_name=run_name,
                dataset_name=dataset_name,
            )
            for case in case_list
        ]

    def evaluate_groups(
        self,
        groups: "Iterable[dict]",
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
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
        )

    async def a_evaluate_groups(
        self,
        groups: "Iterable[dict]",
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
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
    ) -> EvaluationReturn:
        """Async counterpart of :meth:`evaluate`, including output scope.

        Judge calls are bounded by a per-call global concurrency limiter.
        Evaluators without an async path run their synchronous ``evaluate`` in a
        worker thread under the same limiter. Validation still runs before any
        judge call.
        """
        self._validate_max_concurrency(max_concurrency)
        selected = self._select(metrics)
        scope, planned_cases = self._scope_plan(case)
        for planned_case in planned_cases:
            self._validate_case(planned_case, selected)
        limiter = asyncio.Semaphore(max_concurrency)
        pairs = await asyncio.gather(
            *(
                self._a_run_case(
                    planned_case, selected, run_name, dataset_name, limiter
                )
                for planned_case in planned_cases
            )
        )
        results_list: list[EvaluationResults] = []
        for results, records in pairs:
            self._publish(records, results)
            results_list.append(results)
        return self._shape_scope_results(scope, results_list)

    async def a_evaluate_many(
        self,
        cases: "Iterable[EvaluationCase]",
        metrics: list[str] | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> list[EvaluationReturn]:
        """Async counterpart of :meth:`evaluate_many`: cases run concurrently.

        Each expanded logical evaluation is independent (own validation, root
        ``idp_eval.evaluate`` trace, and results/rows/annotations). A single shared
        ``asyncio.Semaphore(max_concurrency)`` caps total simultaneous judge calls
        across every case so a large batch never floods the configured backend.

        Failure behavior mirrors :meth:`evaluate_many`: the whole batch is
        validated up front (fail fast, case-aware error). Results are returned in
        input order; persistence happens serially after evaluation to avoid
        concurrent writer mutation.
        """
        self._validate_max_concurrency(max_concurrency)
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
        flat_cases = [case for _, cases in plans for case in cases]
        pairs = await asyncio.gather(
            *(
                self._a_run_case(case, selected, run_name, dataset_name, limiter)
                for case in flat_cases
            )
        )

        # Ordered, serial persistence (writers are not concurrency-safe).
        flat_results: list[EvaluationResults] = []
        for results, records in pairs:
            self._publish(records, results)
            flat_results.append(results)

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
        return results_list

    async def _a_run_case(
        self,
        case: EvaluationCase,
        selected: list[str],
        run_name: str | None,
        dataset_name: str | None,
        limiter: asyncio.Semaphore,
    ) -> tuple[dict[str, EvaluationResult], list[EvaluationRecord]]:
        """Runs one case's selected evaluators within its own root trace.

        Returns the results and the (not-yet-persisted) records so the caller can
        persist serially. Concurrent cases each get an independent root span/trace
        because each is a separate task with its own OpenTelemetry context.
        """
        with tracing.case_evaluation_span(
            case.case_id,
            run_name,
            dataset_name,
            self._descriptive_input(case),
        ) as handle:
            # One shared document-relevance pass for all selected retrieval
            # metrics; documents are judged concurrently under the same limiter.
            retrieval = self._retrieval_metrics(selected)
            relevance_pass = None
            if retrieval:
                relevance_pass, max_depth = self._build_relevance_pass(
                    case, retrieval
                )
                await relevance_pass.a_run(max_depth, limiter)

            results: dict[str, EvaluationResult] = {}
            for metric_name in selected:
                evaluator = self._evaluators[metric_name]
                if relevance_pass is not None and getattr(
                    evaluator, "uses_retrieval_relevance", False
                ):
                    # Deterministic: reads the shared judgments, no judge calls.
                    results[metric_name] = evaluator.evaluate_shared(
                        case, relevance_pass
                    )
                    continue
                a_evaluate = getattr(evaluator, "a_evaluate", None)
                if a_evaluate is not None:
                    results[metric_name] = await a_evaluate(
                        case, judge_limiter=limiter
                    )
                else:
                    # No async path: run sync evaluate in a worker thread under
                    # the shared limiter (its judge calls stay bounded).
                    async with limiter:
                        results[metric_name] = await asyncio.to_thread(
                            evaluator.evaluate, case
                        )

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

        records = self._build_records(
            case, selected, results, run_name, dataset_name, trace_id, span_id
        )
        return results, records

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
            case_id=resolved_case_id,
            input=(
                self._descriptive_input(case) if case is not None else None
            ),
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
