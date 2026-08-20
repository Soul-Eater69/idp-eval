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

    ``input`` is the optional rendered task/query used for summary reporting.
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
    input: str | None = None

    @classmethod
    def from_result(
        cls,
        result: EvaluationResult,
        *,
        annotator_kind: str,
        case_id: str | None = None,
        input: str | None = None,
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
            input=input,
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
        base_url: str | None = None,
        api_key: str | None = None,
        ingest_timeout: float = 10.0,
        poll_interval: float = 0.5,
    ):
        """Args:
            client: An injected Phoenix client. When provided it is used as-is
                (``base_url`` / ``api_key`` are ignored), which keeps dependency
                injection for tests intact.
            base_url: Phoenix REST base URL. ``None`` → Phoenix reads
                ``PHOENIX_BASE_URL`` (default ``http://localhost:6006``).
            api_key: API key for authenticated Phoenix. ``None`` → Phoenix reads
                ``PHOENIX_API_KEY``. Sent by the Phoenix client as a bearer token;
                never logged or written into annotations.
            ingest_timeout: Max seconds to wait for the target span to be
                ingested before failing.
            poll_interval: Delay between write retries while waiting for ingest.
        """
        self._client = client
        self._base_url = base_url
        self._api_key = api_key
        self._ingest_timeout = ingest_timeout
        self._poll_interval = poll_interval

    def _get_client(self) -> Any:
        """Returns the injected client, or lazily builds a Phoenix ``Client``.

        With no explicit ``base_url`` / ``api_key``, ``Client()`` is used so
        Phoenix resolves ``PHOENIX_BASE_URL`` / ``PHOENIX_API_KEY`` from the
        environment. Explicit values are forwarded to the native client kwargs;
        no Authorization header is constructed by hand.
        """
        if self._client is None:
            from phoenix.client import Client

            kwargs: dict[str, Any] = {}
            if self._base_url is not None:
                kwargs["base_url"] = self._base_url
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            self._client = Client(**kwargs)
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


# ``evaluations`` summary-sheet columns, in order. ``raw_details_json`` is an
# optional lossless trailing column (structured sheets are the primary view).
_SUMMARY_COLUMNS = (
    "run_name",
    "dataset_name",
    "case_id",
    "input",
    "trace_id",
    "metric",
    "score",
    "label",
    "explanation",
    "annotator_kind",
    "timestamp",
    "raw_details_json",
)

# Identity columns prefixed onto every metric-detail sheet row.
_ITEM_IDENTITY_COLUMNS = (
    "run_name",
    "dataset_name",
    "case_id",
    "trace_id",
    "metric",
)


@dataclass(frozen=True)
class _ItemSheet:
    """Declarative mapping from a metric's ``details`` to a flat detail sheet.

    ``columns`` pairs each output column name with the field to read from each
    item dict, so one generic loop can flatten any registered metric — no
    per-metric branching in the writer.
    """

    sheet: str
    list_key: str                          # details key holding the item list
    columns: tuple[tuple[str, str], ...]   # (output column, item field)


# Small declarative registry (not per-metric if/else): each entry flattens one
# metric's item list into a dedicated sheet. Arbitrary custom metrics appear
# only in the summary sheet unless they define a repository-supported layout.
_ITEM_SHEETS: dict[str, _ItemSheet] = {
    # Whole-source coverage. ``details["items"]`` is only present in verbose mode
    # (compact results keep details lean), so this sheet fills for verbose runs.
    "coverage": _ItemSheet(
        "coverage_items",
        "items",
        (
            ("item_id", "id"),
            ("source_item", "source_item"),
            ("meaningfully_present", "meaningfully_present"),
            ("fully_present", "fully_present"),
            ("status", "status"),
            ("item_score", "item_score"),
            ("reason", "reason"),
        ),
    ),
    "instruction_adherence": _ItemSheet(
        "instruction_adherence_items",
        "instructions",
        (
            ("instruction_id", "id"),
            ("instruction", "instruction"),
            ("status", "status"),
            ("item_score", "score"),
            ("reason", "reason"),
        ),
    ),
    "faithfulness": _ItemSheet(
        "faithfulness_items",
        "claims",
        (
            ("claim_id", "id"),
            ("claim", "claim"),
            ("status", "status"),
            ("item_score", "item_score"),
            ("reason", "reason"),
        ),
    ),
}

# Retrieval metrics (relevance_at_k / ndcg_at_k) reuse the SAME per-document
# relevance judgments, so their documents are written to one shared sheet, once
# per case, rather than duplicated per metric. Detected structurally by a
# ``documents`` list in details (a key no other metric uses).
_RETRIEVAL_DOCUMENTS_SHEET = "retrieval_documents"
_RETRIEVAL_IDENTITY_COLUMNS = ("run_name", "dataset_name", "case_id", "trace_id")
_RETRIEVAL_DOCUMENT_COLUMNS = (
    ("rank", "rank"),
    ("document_id", "document_id"),
    ("relevance_score", "relevance_score"),
    ("relevance_label", "relevance_label"),
    ("explanation", "explanation"),
    ("retrieval_score", "retrieval_score"),
)

# Columns given extra width because they hold free text.
_WIDE_COLUMNS = frozenset(
    {
        "explanation",
        "reason",
        "requirement",
        "source_item",
        "instruction",
        "claim",
        "input",
        "raw_details_json",
    }
)


class ExcelEvaluationWriter:
    """Writes evaluation results to a multi-sheet ``.xlsx`` workbook.

    - ``evaluations``: one row per (case + metric) — the human-readable summary.
    - ``<metric>_items``: one row per structured detail item for metrics with a
      registered item layout (coverage requirements / source items, adherence
      instructions), so results are inspectable without reading JSON.

    The workbook is created once and held in memory; each :meth:`write` appends
    rows to the open sheets and saves the file, so it stays current for
    single-case and batch runs (evaluators are never re-run). Metrics without a
    registered layout (custom code metrics, for example) appear only in the
    summary; their full ``details`` remain available in ``raw_details_json``.
    Independent of Phoenix.
    """

    def __init__(self, path: str):
        self._path = path
        self._workbook: Any | None = None
        self._summary: Any | None = None
        self._item_sheets: dict[str, Any] = {}

    def write(self, records: list[EvaluationRecord]) -> None:
        self._ensure_workbook()
        # ``write`` receives one case's records, so retrieval metrics that share
        # the same document-relevance judgments write their document rows only
        # once (the first retrieval record in the batch).
        retrieval_written = False
        for record in records:
            self._summary.append(
                [
                    record.run_name,
                    record.dataset_name,
                    record.case_id,
                    record.input,
                    record.trace_id,
                    record.metric,
                    record.score,
                    record.label,
                    record.explanation,
                    record.annotator_kind,
                    record.timestamp,
                    json.dumps(record.details, default=str)
                    if record.details is not None
                    else None,
                ]
            )
            self._append_item_rows(record)
            if not retrieval_written and self._append_retrieval_rows(record):
                retrieval_written = True
        self._workbook.save(self._path)

    def _ensure_workbook(self) -> None:
        """Builds the workbook and summary sheet once, on the first write."""
        if self._workbook is not None:
            return
        from openpyxl import Workbook

        self._workbook = Workbook()
        self._summary = self._workbook.active
        self._summary.title = "evaluations"
        self._init_sheet(self._summary, _SUMMARY_COLUMNS)

    def _append_item_rows(self, record: EvaluationRecord) -> None:
        """Flattens one record's structured items onto its metric-detail sheet."""
        spec = _ITEM_SHEETS.get(record.metric)
        if spec is None or not record.details:
            return
        items = record.details.get(spec.list_key)
        if not isinstance(items, list) or not items:
            return
        sheet = self._item_sheet(spec)
        for item in items:
            if not isinstance(item, dict):
                continue
            row = [
                record.run_name,
                record.dataset_name,
                record.case_id,
                record.trace_id,
                record.metric,
            ]
            row.extend(item.get(field) for _, field in spec.columns)
            sheet.append(row)

    def _append_retrieval_rows(self, record: EvaluationRecord) -> bool:
        """Writes shared retrieved-document rows once per case; returns whether it did.

        Retrieval records are recognized structurally by a non-empty ``documents``
        list of dicts with a ``rank`` (a shape unique to retrieval metrics), so
        both ``relevance_at_k`` and ``ndcg_at_k`` map to one shared sheet without
        duplicate rows or a metric column.
        """
        details = record.details
        if not isinstance(details, dict):
            return False
        documents = details.get("documents")
        if not isinstance(documents, list) or not documents:
            return False
        if not all(isinstance(doc, dict) and "rank" in doc for doc in documents):
            return False
        sheet = self._item_sheets.get(_RETRIEVAL_DOCUMENTS_SHEET)
        if sheet is None:
            headers = list(_RETRIEVAL_IDENTITY_COLUMNS) + [
                column for column, _ in _RETRIEVAL_DOCUMENT_COLUMNS
            ]
            sheet = self._workbook.create_sheet(_RETRIEVAL_DOCUMENTS_SHEET)
            self._init_sheet(sheet, headers)
            self._item_sheets[_RETRIEVAL_DOCUMENTS_SHEET] = sheet
        for doc in documents:
            row = [
                record.run_name,
                record.dataset_name,
                record.case_id,
                record.trace_id,
            ]
            row.extend(doc.get(field) for _, field in _RETRIEVAL_DOCUMENT_COLUMNS)
            sheet.append(row)
        return True

    def _item_sheet(self, spec: _ItemSheet) -> Any:
        """Returns the detail sheet for ``spec``, creating it on first use."""
        sheet = self._item_sheets.get(spec.sheet)
        if sheet is None:
            sheet = self._workbook.create_sheet(spec.sheet)
            headers = list(_ITEM_IDENTITY_COLUMNS) + [
                column for column, _ in spec.columns
            ]
            self._init_sheet(sheet, headers)
            self._item_sheets[spec.sheet] = sheet
        return sheet

    @staticmethod
    def _init_sheet(sheet: Any, headers) -> None:
        """Writes a styled header row: bold, frozen, auto-filtered, sized."""
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter

        headers = list(headers)
        sheet.append(headers)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"
        for index, header in enumerate(headers, start=1):
            width = 48 if header in _WIDE_COLUMNS else max(len(header) + 2, 12)
            sheet.column_dimensions[get_column_letter(index)].width = width


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
