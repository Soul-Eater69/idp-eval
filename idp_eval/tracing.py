"""Eval-only OpenTelemetry tracing helpers.

The tracing model is intentionally narrow (v1):

    one EvaluationCase   -> one trace  (root span ``idp_eval.evaluate``)
    one real judge call  -> one child span (e.g. ``coverage.extract``)

Everything here degrades to a safe no-op when OpenTelemetry is not installed or
no tracer provider has been registered (``register_tracing`` not called): spans
are non-recording and attribute writes do nothing, so ``evaluate`` behaves
exactly as before. This module never traces arbitrary application code — only
the framework's own evaluation work.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

_TRACER_NAME = "idp_eval"


def _trace_api():
    """Returns the ``opentelemetry.trace`` module, or ``None`` if unavailable."""
    try:
        from opentelemetry import trace
    except Exception:  # noqa: BLE001 - OTel is optional
        return None
    return trace


class CaseSpanHandle:
    """Handle to the case-level root span, exposing the trace id."""

    def __init__(self, span: Any | None):
        self._span = span

    @property
    def trace_id(self) -> str | None:
        """Returns the 32-hex trace id, or ``None`` when not tracing."""
        if self._span is None:
            return None
        context = self._span.get_span_context()
        if context is None or context.trace_id == 0:
            return None
        return format(context.trace_id, "032x")

    @property
    def span_id(self) -> str | None:
        """Returns the 16-hex span id of the root span, or ``None``.

        This is the target for native Phoenix span annotations.
        """
        if self._span is None:
            return None
        context = self._span.get_span_context()
        if context is None or context.span_id == 0:
            return None
        return format(context.span_id, "016x")


@contextmanager
def case_evaluation_span(
    case_id: str | None,
    run_name: str | None = None,
    dataset_name: str | None = None,
) -> Iterator[CaseSpanHandle]:
    """Opens the root span for evaluating one case.

    Child spans created (in the same thread) while this is active nest under it
    automatically via OpenTelemetry context propagation.

    Args:
        case_id: Case identifier, attached as ``idp_eval.case_id`` when set.
        run_name: Optional benchmark run name (``idp_eval.run_name``).
        dataset_name: Optional dataset name (``idp_eval.dataset_name``).

    Yields:
        A :class:`CaseSpanHandle` for the root span.
    """
    trace = _trace_api()
    if trace is None:
        yield CaseSpanHandle(None)
        return

    attributes: dict[str, str] = {}
    if case_id:
        attributes["idp_eval.case_id"] = case_id
    if run_name:
        attributes["idp_eval.run_name"] = run_name
    if dataset_name:
        attributes["idp_eval.dataset_name"] = dataset_name

    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(
        "idp_eval.evaluate", attributes=attributes
    ) as span:
        yield CaseSpanHandle(span)


@contextmanager
def judge_span(
    name: str, attributes: dict[str, str] | None = None
) -> Iterator[None]:
    """Wraps one real judge/model call in a named child span.

    Use only around calls that actually invoke the model, so the trace reflects
    real work. No-op when tracing is unavailable.

    Args:
        name: Stable span name (e.g. ``"coverage.extract"``).
        attributes: Optional namespaced span attributes.
    """
    trace = _trace_api()
    if trace is None:
        yield
        return
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name, attributes=attributes or {}):
        yield


def annotate_current_span(
    metric: str,
    score: float | None,
    label: str | None,
    explanation: str | None,
    annotator_kind: str,
) -> None:
    """Records one metric result as attributes/event on the current span.

    Safe no-op when tracing is off (the current span is non-recording). Large
    input/context/output strings are deliberately not written here; only compact
    result metadata.
    """
    trace = _trace_api()
    if trace is None:
        return
    span = trace.get_current_span()
    if span is None:
        return

    prefix = f"idp_eval.eval.{metric}"
    if score is not None:
        span.set_attribute(f"{prefix}.score", score)
    if label is not None:
        span.set_attribute(f"{prefix}.label", label)
    span.set_attribute(f"{prefix}.annotator_kind", annotator_kind)
    if explanation:
        span.add_event(prefix, {"explanation": explanation})
