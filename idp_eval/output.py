"""Evaluation result publishing: normalized records and output writers.

Evaluation happens once and produces :class:`~idp_eval.models.EvaluationResult`
objects. Those are normalized into :class:`EvaluationRecord` rows and handed to
one or more :class:`EvaluationWriter` implementations. The canonical Phoenix
persistence mechanism is **native span annotations** (not span attributes);
Excel is an independent flat export. Writers never trigger re-evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from idp_eval.models import EvaluationResult

# Phoenix's accepted annotator kinds.
ANNOTATOR_KINDS = ("LLM", "CODE", "HUMAN")


def validate_annotator_kind(kind: str) -> str:
    """Returns ``kind`` if it is a supported annotator kind, else raises."""
    if kind not in ANNOTATOR_KINDS:
        raise ValueError(
            f"Unsupported annotator_kind {kind!r}; expected one of "
            f"{ANNOTATOR_KINDS}."
        )
    return kind


class PersistenceError(RuntimeError):
    """Raised when writing results fails after evaluation already succeeded.

    The computed results are preserved on :attr:`results` so callers never lose
    them just because persistence failed.
    """

    def __init__(self, message: str, results: dict[str, EvaluationResult]):
        super().__init__(message)
        self.results = results


@dataclass(frozen=True)
class EvaluationRecord:
    """One normalized (case + metric result) row for publishing.

    ``span_id`` targets the root ``idp_eval.evaluate`` span for native Phoenix
    annotations; it is required for Phoenix output and ``None`` when tracing is
    inactive. ``identifier`` is an optional upsert key for the Phoenix annotation
    API and is never required.
    """

    metric: str
    score: float | None
    label: str | None
    explanation: str | None
    annotator_kind: str
    case_id: str | None = None
    run_name: str | None = None
    dataset_name: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    identifier: str | None = None
    details: dict[str, Any] | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_result(
        cls,
        result: EvaluationResult,
        *,
        annotator_kind: str,
        case_id: str | None = None,
        run_name: str | None = None,
        dataset_name: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        identifier: str | None = None,
    ) -> "EvaluationRecord":
        """Builds a record from an :class:`EvaluationResult`."""
        return cls(
            metric=result.metric,
            score=result.score,
            label=result.label,
            explanation=result.explanation,
            annotator_kind=validate_annotator_kind(annotator_kind),
            case_id=case_id,
            run_name=run_name,
            dataset_name=dataset_name,
            trace_id=trace_id,
            span_id=span_id,
            identifier=identifier,
            details=result.details,
        )


@runtime_checkable
class EvaluationWriter(Protocol):
    """Destination for normalized evaluation records."""

    def write(self, records: list[EvaluationRecord]) -> None:
        """Persists a batch of records."""


def _safe_metadata(details: dict[str, Any] | None) -> dict[str, Any] | None:
    """Returns compact, JSON-safe annotation metadata (or ``None``).

    Guards against dumping oversized content into annotation metadata by
    round-tripping through JSON with a string fallback.
    """
    if not details:
        return None
    return json.loads(json.dumps(details, default=str))


def _make_span_annotation(payload: dict[str, Any]):
    """Builds a Phoenix ``SpanAnnotationData`` from a normalized payload.

    Imported lazily so Excel-only usage never imports Phoenix. Patchable in
    tests to avoid requiring the Phoenix package.
    """
    from phoenix.client.resources.spans import SpanAnnotationData

    return SpanAnnotationData(**payload)


def _flush_tracer_provider() -> None:
    """Best-effort flush of pending spans so Phoenix can ingest the target span.

    A span annotation can only attach to a span Phoenix has already received, so
    we flush before logging. No-op when OpenTelemetry is unavailable or the
    active provider has no ``force_flush``.
    """
    try:
        from opentelemetry import trace

        flush = getattr(trace.get_tracer_provider(), "force_flush", None)
        if callable(flush):
            flush()
    except Exception:  # noqa: BLE001 - flushing is best-effort
        pass


class PhoenixEvaluationWriter:
    """Writes each result as a native Phoenix span annotation on the case span.

    Annotations for one case are sent in a single batched
    ``client.spans.log_span_annotations`` call. Requires an active trace (a
    ``span_id`` on every record); if Phoenix output is requested without tracing,
    this raises rather than silently claiming persistence.

    Because the framework logs annotations right after the case span closes, the
    target span may not be ingested by Phoenix yet — Phoenix returns 404 in that
    window and drops the annotation. So the writer flushes pending spans, then
    retries the (idempotent-per-span) write on 404 up to ``ingest_timeout``
    seconds before surfacing the error.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        ingest_timeout: float = 10.0,
        poll_interval: float = 0.5,
    ):
        """Args:
            client: An injected Phoenix client (built lazily if ``None``).
            ingest_timeout: Max seconds to wait for the target span to be
                ingested before failing.
            poll_interval: Delay between write retries while waiting for ingest.
        """
        self._client = client
        self._ingest_timeout = ingest_timeout
        self._poll_interval = poll_interval

    def _get_client(self) -> Any:
        if self._client is None:
            from phoenix.client import Client

            self._client = Client()
        return self._client

    @staticmethod
    def _payload(record: EvaluationRecord) -> dict[str, Any]:
        """Maps a record to Phoenix ``SpanAnnotationData`` kwargs.

        ``score`` / ``label`` / ``explanation`` are omitted when ``None`` (a
        not-applicable result is never coerced into a fake ``0.0``).
        """
        result: dict[str, Any] = {}
        if record.score is not None:
            result["score"] = record.score
        if record.label is not None:
            result["label"] = record.label
        if record.explanation is not None:
            result["explanation"] = record.explanation

        payload: dict[str, Any] = {
            "name": record.metric,
            "span_id": record.span_id,
            "annotator_kind": record.annotator_kind,
            "result": result,
        }
        if record.identifier is not None:
            payload["identifier"] = record.identifier
        metadata = _safe_metadata(record.details)
        if metadata is not None:
            payload["metadata"] = metadata
        return payload

    def write(self, records: list[EvaluationRecord]) -> None:
        if not records:
            return
        if any(record.span_id is None for record in records):
            raise RuntimeError(
                "Phoenix output requires an active trace: no target span id. "
                "Call register_tracing() before evaluating."
            )
        annotations = [_make_span_annotation(self._payload(r)) for r in records]
        _flush_tracer_provider()
        self._log_with_retry(annotations)

    def _log_with_retry(self, annotations: list[Any]) -> None:
        """Logs the batch, retrying on 404 until the target span is ingested."""
        import time

        import httpx

        client = self._get_client()
        deadline = time.monotonic() + self._ingest_timeout
        while True:
            try:
                # sync=True validates persistence (and raises 404 if the target
                # span is not ingested yet); the async default would silently
                # drop the annotation.
                client.spans.log_span_annotations(
                    span_annotations=annotations, sync=True
                )
                return
            except httpx.HTTPStatusError as exc:
                # 404 => target span not yet ingested by Phoenix; retry briefly.
                if (
                    exc.response.status_code != 404
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(self._poll_interval)


# Main-sheet columns, in order.
_EXCEL_COLUMNS = (
    "run_name",
    "dataset_name",
    "case_id",
    "trace_id",
    "metric",
    "score",
    "label",
    "explanation",
    "annotator_kind",
    "details_json",
    "timestamp",
)


class ExcelEvaluationWriter:
    """Appends one row per (case + metric) to an ``.xlsx`` workbook.

    Rows accumulate in memory and the whole file is rewritten on each
    :meth:`write`, so the file is always current for single-case and batch runs.
    Nested evaluator ``details`` are serialized into a single ``details_json``
    column so arbitrary custom metrics need no dynamic columns. Independent of
    Phoenix.
    """

    def __init__(self, path: str):
        self._path = path
        self._rows: list[list[Any]] = []

    def write(self, records: list[EvaluationRecord]) -> None:
        for record in records:
            self._rows.append(
                [
                    record.run_name,
                    record.dataset_name,
                    record.case_id,
                    record.trace_id,
                    record.metric,
                    record.score,
                    record.label,
                    record.explanation,
                    record.annotator_kind,
                    json.dumps(record.details, default=str)
                    if record.details is not None
                    else None,
                    record.timestamp,
                ]
            )
        self._flush()

    def _flush(self) -> None:
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "evaluations"
        sheet.append(list(_EXCEL_COLUMNS))
        for row in self._rows:
            sheet.append(row)
        workbook.save(self._path)


def build_writers(
    output: str | None, excel_path: str | None
) -> list[EvaluationWriter]:
    """Builds the configured writers for an output mode.

    Args:
        output: One of ``None``, ``"phoenix"``, ``"excel"``, or ``"both"``.
        excel_path: Required when ``output`` includes Excel.

    Returns:
        The list of writers (possibly empty).

    Raises:
        ValueError: For an unknown mode or a missing Excel path.
    """
    if output is None:
        return []
    if output not in ("phoenix", "excel", "both"):
        raise ValueError(
            f"Unknown output {output!r}; expected 'phoenix', 'excel', 'both', "
            "or None."
        )

    writers: list[EvaluationWriter] = []
    if output in ("phoenix", "both"):
        writers.append(PhoenixEvaluationWriter())
    if output in ("excel", "both"):
        if not excel_path:
            raise ValueError("excel_path is required when output includes Excel.")
        writers.append(ExcelEvaluationWriter(excel_path))
    return writers
