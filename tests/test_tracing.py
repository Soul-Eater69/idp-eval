"""Eval-only trace structure tests using an in-memory OpenTelemetry exporter."""

import pytest

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    InstructionAdherenceEvaluator,
)


CASE = EvaluationCase(
    context="Source requirement.",
    output="Generated output.",
    instructions="Be concise.",
    case_id="case-1",
)


class Judge:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_object(self, prompt, schema):
        self.calls += 1
        return self.responses.pop(0)


def _coverage_judge(*, empty=False):
    return Judge(
        {
            "items": []
            if empty
            else [
                {
                    "source_item": "Source requirement.",
                    "meaningfully_present": True,
                    "fully_present": True,
                }
            ]
        }
    )


def _instruction_judge():
    return Judge(
        {
            "instructions": [
                {"instruction": "Be concise.", "status": "followed"}
            ]
        },
    )


@pytest.fixture
def spans():
    trace = pytest.importorskip("opentelemetry").trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    yield exporter
    exporter.clear()


def test_one_case_is_one_root_with_one_coverage_stage(spans):
    EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge())]
    ).evaluate(CASE)
    finished = spans.get_finished_spans()
    roots = [span for span in finished if span.name == "idp_eval.evaluate"]
    stages = [span for span in finished if span.name == "coverage.evaluate"]
    assert len(roots) == len(stages) == 1
    assert stages[0].parent.span_id == roots[0].context.span_id
    assert {span.name for span in finished} == {
        "idp_eval.evaluate",
        "coverage.evaluate",
    }


def test_empty_items_still_has_one_real_coverage_call_span(spans):
    judge = _coverage_judge(empty=True)
    result = EvaluationFramework(
        evaluators=[CoverageEvaluator(judge)]
    ).evaluate(CASE)["coverage"]
    names = [span.name for span in spans.get_finished_spans()]
    assert judge.calls == 1
    assert result.label == "not_applicable"
    assert names.count("coverage.evaluate") == 1


def test_multiple_metrics_share_the_same_case_trace(spans):
    EvaluationFramework(
        evaluators=[
            CoverageEvaluator(_coverage_judge()),
            InstructionAdherenceEvaluator(_instruction_judge()),
        ]
    ).evaluate(CASE)
    finished = spans.get_finished_spans()
    root = next(span for span in finished if span.name == "idp_eval.evaluate")
    stages = [span for span in finished if span.name != "idp_eval.evaluate"]
    assert {span.name for span in stages} == {
        "coverage.evaluate",
        "instruction_adherence.evaluate",
    }
    assert all(span.parent.span_id == root.context.span_id for span in stages)


def test_two_cases_create_two_independent_traces(spans):
    framework = EvaluationFramework(
        evaluators=[
            CoverageEvaluator(
                Judge(
                    {
                        "items": [
                            {
                                "source_item": "A",
                                "meaningfully_present": True,
                                "fully_present": True,
                            }
                        ]
                    },
                    {
                        "items": [
                            {
                                "source_item": "A",
                                "meaningfully_present": True,
                                "fully_present": True,
                            }
                        ]
                    },
                )
            )
        ]
    )
    framework.evaluate_many(
        [
            EvaluationCase(context="c", output="o", case_id="a"),
            EvaluationCase(context="c", output="o", case_id="b"),
        ]
    )
    roots = [
        span
        for span in spans.get_finished_spans()
        if span.name == "idp_eval.evaluate"
    ]
    assert len(roots) == 2
    assert len({span.context.trace_id for span in roots}) == 2


def test_root_contains_case_and_compact_result_attributes(spans):
    EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge())]
    ).evaluate(CASE, run_name="run", dataset_name="dataset")
    root = next(
        span
        for span in spans.get_finished_spans()
        if span.name == "idp_eval.evaluate"
    )
    assert root.attributes["idp_eval.case_id"] == "case-1"
    assert root.attributes["idp_eval.run_name"] == "run"
    assert root.attributes["idp_eval.dataset_name"] == "dataset"
    assert root.attributes["idp_eval.eval.coverage.score"] == 1.0
    assert root.attributes["coverage.item_count"] == 1
    assert root.attributes["coverage.judge_call_count"] == 1
    assert root.attributes["coverage.verbose"] is False
    assert "coverage.total_ms" in root.attributes
