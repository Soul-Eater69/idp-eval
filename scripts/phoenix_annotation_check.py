"""Development-only live Phoenix annotation integration check.

Proves the end-to-end native-annotation path against a REAL/local Phoenix
instance, stage by stage:

    1. Phoenix server reachable
    2. root ``idp_eval.evaluate`` span exported to Phoenix
    3. library-written ``coverage`` annotation (output="phoenix") read back
    4. an independent ``integration_check`` annotation written + read back

It uses a fake, no-cost evaluator (no judge/LLM/gateway call). Local Phoenix
started with ``phoenix serve`` needs no API key (auth disabled by default);
Phoenix creates the project automatically on first trace ingest.

Env overrides (self-hosted-with-auth / Phoenix Cloud):
    PHOENIX_HOST / PHOENIX_COLLECTOR_ENDPOINT — server endpoint
    PHOENIX_API_KEY — only when the server has auth enabled

Run it manually (NOT part of the offline unit suite):

    phoenix serve                       # in another terminal
    python3 -m scripts.phoenix_annotation_check
"""

from __future__ import annotations

import os
import time

import httpx

from idp_eval import EvaluationCase, EvaluationFramework, register_tracing
from idp_eval.models import EvaluationResult, Evaluator

PROJECT = "idp-eval-integration-check"
HOST = os.environ.get("PHOENIX_HOST", "http://127.0.0.1:6006").rstrip("/")


class _FakeCoverage(Evaluator):
    """No-cost evaluator standing in for a real metric."""

    name = "task_coverage"
    annotator_kind = "LLM"

    def __init__(self, llm=None):
        pass

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        return EvaluationResult(
            metric=self.name,
            score=0.625,
            label="partial",
            explanation="2 covered, 1 partial, 1 missing.",
            details={"total_requirements": 4},
        )


def _fail(stage: str, detail: str = "") -> None:
    print(f"Integration check: FAIL ({stage}){' — ' + detail if detail else ''}")
    raise SystemExit(1)


def _install_inmemory_capture():
    """Adds an in-memory exporter to capture the root span id.

    Phoenix's ``register(batch=False)`` installs the OTLP exporter as a
    *replaceable default* processor: the first ``add_span_processor`` overwrites
    it. So we explicitly re-add an OTLP processor first (to preserve export to
    Phoenix), then add the capture processor.
    """
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    otlp_endpoint = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317"
    )
    provider = trace.get_tracer_provider()
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
    )
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def _poll_project_exists(timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{HOST}/v1/projects", timeout=5)
            resp.raise_for_status()
            if any(p["name"] == PROJECT for p in resp.json().get("data", [])):
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def _poll_annotation(client, span_id: str, name: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            got = client.spans.get_span_annotations(
                span_ids=[span_id],
                project_identifier=PROJECT,
                include_annotation_names=[name],
            )
            if got:
                return got[0]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:  # 404 => not ingested yet
                raise
        time.sleep(0.5)
    return None


def main() -> None:
    # Stage 1: server reachable.
    try:
        httpx.get(f"{HOST}/v1/projects", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        _fail("server unreachable", f"{HOST}: {exc}")
    print(f"Phoenix server: reachable ({HOST})")
    print(f"Auth: {'API key set' if os.environ.get('PHOENIX_API_KEY') else 'none (local default)'}")
    print(f"Project: {PROJECT}")

    register_tracing(project_name=PROJECT)
    capture = _install_inmemory_capture()

    # Stage 2: run a no-cost evaluation; the library logs the coverage annotation.
    framework = EvaluationFramework(evaluators=[_FakeCoverage()], output="phoenix")
    case = EvaluationCase(
        case_id="integration-001", input="task", context="ctx", output="out"
    )
    framework.evaluate(case, run_name="integration-check")

    root = next(
        s for s in capture.get_finished_spans() if s.name == "idp_eval.evaluate"
    )
    root_span_id = format(root.context.span_id, "016x")
    from opentelemetry import trace

    trace.get_tracer_provider().force_flush()

    if not _poll_project_exists():
        _fail(
            "trace export",
            "Root span was created locally but was not observed in Phoenix.",
        )
    print("Root span exported: yes")
    print(f"Root span id: {root_span_id}")

    from phoenix.client import Client

    client = Client()

    # Stage 3: read back the library-written task_coverage annotation.
    coverage = _poll_annotation(client, root_span_id, "task_coverage")
    if coverage is None:
        _fail("task_coverage annotation read-back", "annotation not found")
    print("Native annotation write: success (task_coverage, via output='phoenix')")

    # Stage 4: independent write + read-back to exercise the write API directly.
    check = _make_check_annotation(root_span_id)
    client.spans.log_span_annotations(span_annotations=[check], sync=True)
    read = _poll_annotation(client, root_span_id, "integration_check")
    if read is None:
        _fail("integration_check read-back", "integration_check not found")

    print("Annotation read-back: success")
    result = coverage["result"]
    print("Annotation:")
    print(f"  name: {coverage['name']}")
    print(f"  score: {result.get('score')}")
    print(f"  label: {result.get('label')}")
    print(f"  annotator_kind: {coverage['annotator_kind']}")
    print(f"  span_id: {coverage['span_id']}")
    assert coverage["span_id"] == root_span_id, "annotation not on root span"
    print("Integration check: PASS")


def _make_check_annotation(span_id: str):
    from idp_eval.output import _make_span_annotation

    return _make_span_annotation(
        {
            "name": "integration_check",
            "span_id": span_id,
            "annotator_kind": "CODE",
            "result": {
                "score": 1.0,
                "label": "pass",
                "explanation": "Native Phoenix annotation integration check.",
            },
        }
    )


if __name__ == "__main__":
    main()
