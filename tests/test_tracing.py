"""Trace-structure tests using an in-memory OpenTelemetry exporter (no Phoenix)."""

import pytest

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    InstructionAdherenceEvaluator,
)
from idp_eval.models import EvaluationResult, Evaluator

opentelemetry = pytest.importorskip("opentelemetry")
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture
def spans():
    """Installs an in-memory tracer provider and yields the captured spans."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The OTel API only allows setting the global provider once per process;
    # override the private slot so each test starts from a clean exporter.
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    yield exporter
    exporter.clear()


class ScriptedJudge:
    def __init__(self, *responses):
        self._responses = list(responses)

    def generate_object(self, prompt, schema):
        return self._responses.pop(0)


def _coverage_judge(*, empty=False):
    if empty:
        return ScriptedJudge({"requirements": []})
    return ScriptedJudge(
        {"requirements": [{"requirement": "a"}, {"requirement": "b"}]},
        {"requirements": [
            {"id": "r1", "meaningfully_present": True, "fully_present": True, "reason": "r"},
            {"id": "r2", "meaningfully_present": False, "fully_present": False, "reason": "r"},
        ]},
    )


CASE = EvaluationCase(case_id="gt-001", input="task", context="ctx", output="out")


def _names(spans):
    return [s.name for s in spans.get_finished_spans()]


def test_one_case_one_trace_with_two_coverage_spans(spans):
    framework = EvaluationFramework(evaluators=[CoverageEvaluator(_coverage_judge())])
    framework.evaluate(CASE)

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert "idp_eval.evaluate" in names
    assert "coverage.extract" in names
    assert "coverage.classify" in names

    root = next(s for s in finished if s.name == "idp_eval.evaluate")
    trace_ids = {s.context.trace_id for s in finished}
    assert len(trace_ids) == 1  # all spans share one trace
    # Child stage spans are parented to the root case span.
    for s in finished:
        if s.name.startswith("coverage."):
            assert s.parent.span_id == root.context.span_id
    assert root.attributes["idp_eval.case_id"] == "gt-001"


def test_empty_extraction_only_extract_span(spans):
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge(empty=True))]
    )
    framework.evaluate(CASE)
    names = _names(spans)
    assert "coverage.extract" in names
    assert "coverage.classify" not in names  # Stage 2 skipped -> no fake span


def test_multiple_metrics_same_trace(spans):
    instr_judge = ScriptedJudge(
        {"instructions": [{"instruction": "Be concise."}]},
        {"answers": [
            {"id": "I1", "status": "followed", "reason": "r"}
        ]},
    )
    framework = EvaluationFramework(
        evaluators=[
            CoverageEvaluator(_coverage_judge()),
            InstructionAdherenceEvaluator(instr_judge),
        ]
    )
    case = EvaluationCase(
        case_id="gt-007", input="task", context="ctx", output="out",
        instructions="Be concise.",
    )
    framework.evaluate(case)

    finished = spans.get_finished_spans()
    assert {s.context.trace_id for s in finished} == {
        next(s for s in finished if s.name == "idp_eval.evaluate").context.trace_id
    }
    names = [s.name for s in finished]
    assert "coverage.extract" in names
    assert "coverage.classify" in names
    assert "instruction_adherence.extract" in names
    assert "instruction_adherence.classify" in names
    root = next(s for s in finished if s.name == "idp_eval.evaluate")
    for span in finished:
        if span.name.startswith("instruction_adherence."):
            assert span.parent.span_id == root.context.span_id


def test_empty_instruction_extraction_only_creates_extract_span(spans):
    framework = EvaluationFramework(
        evaluators=[
            InstructionAdherenceEvaluator(
                ScriptedJudge({"instructions": []})
            )
        ]
    )
    case = EvaluationCase(
        case_id="c", input="t", context="c", output="o", instructions="noise"
    )
    framework.evaluate(case)
    names = _names(spans)
    assert "instruction_adherence.extract" in names
    assert "instruction_adherence.classify" not in names


def test_not_applicable_instruction_makes_no_judge_span(spans):
    framework = EvaluationFramework(
        evaluators=[InstructionAdherenceEvaluator(ScriptedJudge())]
    )
    # No instructions -> not_applicable -> no LLM call -> no judge span.
    case = EvaluationCase(case_id="c", input="t", context="c", output="o")
    framework.evaluate(case)
    assert "instruction_adherence.extract" not in _names(spans)
    assert "instruction_adherence.classify" not in _names(spans)


def test_two_cases_two_traces(spans):
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge())]
    )
    framework.evaluate(EvaluationCase(case_id="gt-001", input="t", context="c", output="o"))
    framework2 = EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge())]
    )
    framework2.evaluate(EvaluationCase(case_id="gt-002", input="t", context="c", output="o"))

    roots = [s for s in spans.get_finished_spans() if s.name == "idp_eval.evaluate"]
    assert len({s.context.trace_id for s in roots}) == 2


def test_fifteen_cases_fifteen_traces_thirty_judge_spans(spans):
    cases = [
        EvaluationCase(case_id=f"gt-{i:03d}", input="t", context="c", output="o")
        for i in range(1, 16)
    ]
    # One framework, fresh scripted judge per case (each judge answers 2 calls).
    for case in cases:
        EvaluationFramework(
            evaluators=[CoverageEvaluator(_coverage_judge())]
        ).evaluate(case)

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "idp_eval.evaluate"]
    extract = [s for s in finished if s.name == "coverage.extract"]
    classify = [s for s in finished if s.name == "coverage.classify"]
    assert len(roots) == 15
    assert len({s.context.trace_id for s in roots}) == 15
    assert len(extract) == 15
    assert len(classify) == 15  # 15 x 2 = 30 judge spans total


def test_supplemental_result_attributes_on_root_span(spans):
    # Compact result attributes are set on the root span whenever tracing is
    # active, independent of output mode (they are supplemental to native
    # Phoenix annotations).
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge())]
    )
    framework.evaluate(CASE, run_name="benchmark-v1", dataset_name="theme-epic-gt")

    root = next(
        s for s in spans.get_finished_spans() if s.name == "idp_eval.evaluate"
    )
    assert root.attributes["idp_eval.run_name"] == "benchmark-v1"
    assert root.attributes["idp_eval.dataset_name"] == "theme-epic-gt"
    assert root.attributes["idp_eval.eval.coverage.score"] == 0.5
    assert root.attributes["idp_eval.eval.coverage.annotator_kind"] == "LLM"


class FakePhoenixClient:
    """Captures native span-annotation batches."""

    def __init__(self):
        self.batches: list[list[dict]] = []
        self.spans = self

    def log_span_annotations(self, span_annotations, sync=False):
        self.batches.append(span_annotations)


def _phoenix_framework(evaluators, monkeypatch):
    """Framework with output='phoenix' wired to a fake client + payload builder."""
    from idp_eval import output

    # Payload builder returns the plain dict so no Phoenix package is needed.
    monkeypatch.setattr(output, "_make_span_annotation", lambda payload: payload)
    framework = EvaluationFramework(evaluators=evaluators, output="phoenix")
    fake = FakePhoenixClient()
    for writer in framework._writers:
        if isinstance(writer, output.PhoenixEvaluationWriter):
            writer._client = fake
    return framework, fake


def test_native_annotations_target_root_span(spans, monkeypatch):
    framework, fake = _phoenix_framework(
        [CoverageEvaluator(_coverage_judge())], monkeypatch
    )
    framework.evaluate(CASE)

    root = next(
        s for s in spans.get_finished_spans() if s.name == "idp_eval.evaluate"
    )
    root_span_id = format(root.context.span_id, "016x")
    # One batched call with one annotation targeting the root span.
    assert len(fake.batches) == 1
    annotation = fake.batches[0][0]
    assert annotation["name"] == "coverage"
    assert annotation["span_id"] == root_span_id
    assert annotation["annotator_kind"] == "LLM"
    assert annotation["result"]["score"] == 0.5


def test_native_annotations_batched_for_multiple_metrics(spans, monkeypatch):
    instr_judge = ScriptedJudge(
        {"instructions": [{"instruction": "Be concise."}]},
        {"answers": [
            {"id": "I1", "status": "followed", "reason": "r"}
        ]},
    )
    framework, fake = _phoenix_framework(
        [CoverageEvaluator(_coverage_judge()), InstructionAdherenceEvaluator(instr_judge)],
        monkeypatch,
    )
    case = EvaluationCase(
        case_id="gt-007", input="t", context="c", output="o",
        instructions="Be concise.",
    )
    framework.evaluate(case)
    # Both metrics in a single batched annotation call.
    assert len(fake.batches) == 1
    names = {a["name"] for a in fake.batches[0]}
    assert names == {"coverage", "instruction_adherence"}


def test_custom_code_evaluation_native_annotation(spans, monkeypatch):
    framework, fake = _phoenix_framework([], monkeypatch)
    framework.log_custom_evaluation(
        name="company_policy", score=1.0, label="pass",
        explanation="All rules passed.", kind="CODE", case_id="gt-001",
    )
    annotation = fake.batches[0][0]
    assert annotation["name"] == "company_policy"
    assert annotation["annotator_kind"] == "CODE"
    assert annotation["result"]["label"] == "pass"
