"""First-class combined/individual/both EvaluationCase orchestration."""

import asyncio

import pytest

from idp_eval import EvaluationCase, EvaluationFramework, EvaluationResult
from idp_eval.models import Evaluator

openpyxl = pytest.importorskip("openpyxl")


class CapturingEvaluator(Evaluator):
    name = "capture"
    required_fields = ("output",)

    def __init__(self, name="capture"):
        self.name = name
        self.cases = []

    def evaluate(self, case):
        self.cases.append(case)
        return EvaluationResult(
            metric=self.name,
            score=1.0,
            label="ok",
            explanation=repr(case.output),
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


def test_default_scope_is_combined_and_allowed_values_construct():
    assert EvaluationCase().evaluation_scope == "combined"
    for scope in ("combined", "individual", "both"):
        assert EvaluationCase(evaluation_scope=scope).evaluation_scope == scope


@pytest.mark.parametrize("scope", ["unknown", "", None, 1])
def test_unknown_scope_rejected_with_allowed_values(scope):
    with pytest.raises(
        ValueError,
        match="allowed values are: 'combined', 'individual', 'both'",
    ):
        EvaluationCase(evaluation_scope=scope)


def test_default_list_output_remains_one_combined_case_and_flat_result():
    evaluator = CapturingEvaluator()
    case = EvaluationCase(output=[{"id": 1}, {"id": 2}])
    result = EvaluationFramework([evaluator]).evaluate(case)

    assert set(result) == {"capture"}
    assert len(evaluator.cases) == 1
    assert evaluator.cases[0] is case
    assert evaluator.cases[0].output == [{"id": 1}, {"id": 2}]


def test_individual_scope_returns_each_item_in_order():
    evaluator = CapturingEvaluator()
    case = EvaluationCase(
        input="task",
        context={"source": "value"},
        output=[{"id": 1}, {"id": 2}],
        instructions="rules",
        case_id="generation",
        metadata={"tag": "test"},
        retrieved_documents=["doc"],
        evaluation_scope="individual",
    )
    result = EvaluationFramework([evaluator]).evaluate(case)

    assert result["combined"] is None
    assert len(result["individual"]) == 2
    assert [entry["capture"].explanation for entry in result["individual"]] == [
        "{'id': 1}",
        "{'id': 2}",
    ]
    assert [item.output for item in evaluator.cases] == case.output
    assert [item.case_id for item in evaluator.cases] == [
        "generation:0",
        "generation:1",
    ]
    assert all(item.evaluation_scope == "combined" for item in evaluator.cases)
    assert all(item.input == case.input for item in evaluator.cases)
    assert all(item.context == case.context for item in evaluator.cases)
    assert all(item.instructions == case.instructions for item in evaluator.cases)
    assert all(
        item.retrieved_documents == case.retrieved_documents
        for item in evaluator.cases
    )
    assert all(item.metadata == case.metadata for item in evaluator.cases)
    assert evaluator.cases[0].metadata is not case.metadata
    assert case.evaluation_scope == "individual"
    assert case.case_id == "generation"
    assert case.output == [{"id": 1}, {"id": 2}]


def test_both_scope_returns_combined_then_individual_results():
    evaluator = CapturingEvaluator()
    outputs = ["first", "second", "third"]
    result = EvaluationFramework([evaluator]).evaluate(
        EvaluationCase(
            input="Generate three options.",
            output=outputs,
            case_id="case",
            evaluation_scope="both",
        )
    )

    assert result["combined"]["capture"].explanation == repr(outputs)
    assert [entry["capture"].explanation for entry in result["individual"]] == [
        repr(output) for output in outputs
    ]
    assert [case.case_id for case in evaluator.cases] == [
        "case",
        "case:0",
        "case:1",
        "case:2",
    ]
    assert all(
        case.input == "Generate three options." for case in evaluator.cases
    )


@pytest.mark.parametrize("scope", ["individual", "both"])
@pytest.mark.parametrize("output", [None, "scalar", {"a": 1}, []])
def test_fanout_scope_requires_nonempty_top_level_list_before_work(scope, output):
    evaluator = CapturingEvaluator()
    with pytest.raises(
        ValueError,
        match=rf"evaluation_scope='{scope}' requires output to be a non-empty list",
    ):
        EvaluationFramework([evaluator]).evaluate(
            EvaluationCase(output=output, evaluation_scope=scope)
        )
    assert evaluator.cases == []


def test_all_expanded_items_validate_before_any_evaluator_work():
    evaluator = CapturingEvaluator()
    case = EvaluationCase(
        output=["valid", ""], evaluation_scope="individual", case_id="batch"
    )
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        EvaluationFramework([evaluator]).evaluate(case)
    assert evaluator.cases == []


def test_scope_is_metric_agnostic_and_metric_filter_still_applies():
    first = CapturingEvaluator("first")
    second = CapturingEvaluator("second")
    framework = EvaluationFramework([first, second])
    result = framework.evaluate(
        EvaluationCase(output=[1, 2], evaluation_scope="both"),
        metrics=["first"],
    )
    assert set(result["combined"]) == {"first"}
    assert all(set(item) == {"first"} for item in result["individual"])
    assert len(first.cases) == 3
    assert second.cases == []


def test_every_selected_metric_runs_for_every_expanded_case():
    first = CapturingEvaluator("first")
    second = CapturingEvaluator("second")
    result = EvaluationFramework([first, second]).evaluate(
        EvaluationCase(output=["a", "b"], evaluation_scope="individual")
    )
    assert all(set(item) == {"first", "second"} for item in result["individual"])
    assert len(first.cases) == len(second.cases) == 2


def test_both_scope_creates_one_root_trace_per_logical_evaluation(spans):
    EvaluationFramework([CapturingEvaluator()]).evaluate(
        EvaluationCase(
            input="Generate alternatives.",
            output=["a", "b"],
            case_id="trace",
            evaluation_scope="both",
        )
    )
    roots = [
        span
        for span in spans.get_finished_spans()
        if span.name == "idp_eval.evaluate"
    ]
    assert len(roots) == 3
    assert len({span.context.trace_id for span in roots}) == 3
    assert [span.attributes.get("idp_eval.case_id") for span in roots] == [
        "trace",
        "trace:0",
        "trace:1",
    ]
    assert all(
        span.attributes["idp_eval.input"] == "Generate alternatives."
        for span in roots
    )
    assert all(
        span.attributes["idp_eval.input_truncated"] is False for span in roots
    )


def test_trace_input_is_bounded_and_marked_when_truncated(spans):
    EvaluationFramework([CapturingEvaluator()]).evaluate(
        EvaluationCase(input="x" * 400, output="answer")
    )
    root = next(
        span
        for span in spans.get_finished_spans()
        if span.name == "idp_eval.evaluate"
    )
    assert len(root.attributes["idp_eval.input"]) == 256
    assert root.attributes["idp_eval.input_truncated"] is True


def test_both_scope_persists_combined_and_individual_rows_once(tmp_path):
    path = tmp_path / "scoped.xlsx"
    framework = EvaluationFramework(
        [CapturingEvaluator()], output="excel", excel_path=str(path)
    )
    framework.evaluate(
        EvaluationCase(
            input="Generate two options.",
            output=["a", "b"],
            case_id="persist",
            evaluation_scope="both",
        )
    )
    sheet = openpyxl.load_workbook(path)["evaluations"]
    rows = list(sheet.iter_rows(values_only=True))
    header, data = rows[0], rows[1:]
    records = [dict(zip(header, row)) for row in data]
    assert [record["case_id"] for record in records] == [
        "persist",
        "persist:0",
        "persist:1",
    ]
    assert [record["input"] for record in records] == [
        "Generate two options.",
        "Generate two options.",
        "Generate two options.",
    ]


def test_async_evaluate_supports_both_with_same_shape():
    evaluator = CapturingEvaluator()
    result = asyncio.run(
        EvaluationFramework([evaluator]).a_evaluate(
            EvaluationCase(output=["a", "b"], evaluation_scope="both"),
            max_concurrency=2,
        )
    )
    assert result["combined"]["capture"].score == 1.0
    assert len(result["individual"]) == 2
    assert len(evaluator.cases) == 3


def test_evaluate_many_preserves_one_return_entry_per_input_case():
    evaluator = CapturingEvaluator()
    framework = EvaluationFramework([evaluator])
    results = framework.evaluate_many(
        [
            EvaluationCase(input="single task", output="one"),
            EvaluationCase(
                input="multi task",
                output=["a", "b"],
                evaluation_scope="individual",
            ),
        ]
    )
    assert set(results[0]) == {"capture"}
    assert results[1]["combined"] is None
    assert len(results[1]["individual"]) == 2
    assert [(case.output, case.input) for case in evaluator.cases] == [
        ("one", "single task"),
        ("a", "multi task"),
        ("b", "multi task"),
    ]


def test_async_many_preserves_scoped_shape_per_input_case():
    evaluator = CapturingEvaluator()
    framework = EvaluationFramework([evaluator])
    results = asyncio.run(
        framework.a_evaluate_many(
            [
                EvaluationCase(input="single task", output="one"),
                EvaluationCase(
                    input="multi task",
                    output=["a", "b"],
                    evaluation_scope="both",
                ),
            ],
            max_concurrency=2,
        )
    )
    assert set(results[0]) == {"capture"}
    assert len(results[1]["individual"]) == 2
    observed = {
        (
            case.output if isinstance(case.output, str) else repr(case.output),
            case.input,
        )
        for case in evaluator.cases
    }
    assert observed == {
        ("one", "single task"),
        (repr(["a", "b"]), "multi task"),
        ("a", "multi task"),
        ("b", "multi task"),
    }
