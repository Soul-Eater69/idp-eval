"""Bulk/group orchestration and structured values (no live LLM)."""

import copy

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    EvaluationResult,
    CoverageEvaluator,
    InstructionAdherenceEvaluator,
    Evaluator,
)

openpyxl = pytest.importorskip("openpyxl")


@pytest.fixture
def spans():
    """In-memory OTel exporter yielding captured spans (no Phoenix)."""
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


class ReusableSourceJudge:
    """Serves unlimited source-coverage cases; counts calls; score 1.0 each."""

    def __init__(self):
        self.calls = 0

    def generate_object(self, prompt, schema):
        self.calls += 1
        return {
            "items": [
                {
                    "source_item": "item",
                    "meaningfully_present": True,
                    "fully_present": True,
                    "reason": "",
                }
            ],
            "overall_reason": "The source item is fully represented.",
        }


class CapturingEvaluator(Evaluator):
    """Capture fanned-out cases without interpreting their structured data."""

    name = "capture"

    def __init__(self):
        self.cases = []

    def evaluate(self, case):
        self.cases.append(case)
        return EvaluationResult("capture", 1.0, "ok", "captured")


def _read(path, sheet="evaluations"):
    rows = list(openpyxl.load_workbook(path)[sheet].iter_rows(values_only=True))
    header, *data = rows
    return header, [dict(zip(header, row)) for row in data]


# --- structured single case (list output is NOT bulk) -----------------------


def test_list_output_stays_one_case_and_is_rendered():
    judge = ReusableSourceJudge()

    class Capturing(ReusableSourceJudge):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_object(self, prompt, schema):
            self.prompts.append(prompt)
            return super().generate_object(prompt, schema)

    judge = Capturing()
    case = EvaluationCase(context="Source doc.", output=["Step 1", "Step 2"])
    result = EvaluationFramework(
        evaluators=[CoverageEvaluator(judge)]
    ).evaluate(case)

    # One case -> one result mapping (a dict, not a list of results).
    assert isinstance(result, dict)
    assert set(result) == {"coverage"}
    # Exactly one judge call: no fan-out of the list into multiple evaluations.
    assert judge.calls == 1
    # The list output was rendered as one structured bullet block in that call.
    user = judge.prompts[0][1]["content"]
    assert "- Step 1\n- Step 2" in user


# --- evaluate_many independence ---------------------------------------------


def _cases():
    return [
        EvaluationCase(
            context={"description": "Theme 1", "needs": ["a", "b"]},
            output={"title": "Epic 1"},
            case_id="theme-1:epic-1",
        ),
        EvaluationCase(
            context={"description": "Theme 2"},
            output={"title": "Epic 2"},
            case_id="theme-2:epic-2",
        ),
    ]


def test_evaluate_many_independent_results_in_order():
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(ReusableSourceJudge())]
    )
    results = framework.evaluate_many(_cases())
    assert isinstance(results, list) and len(results) == 2
    assert results[0]["coverage"].score == 1.0
    assert results[1]["coverage"].score == 1.0


def test_evaluate_many_one_trace_per_case(spans):
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(ReusableSourceJudge())]
    )
    framework.evaluate_many(_cases())
    roots = [s for s in spans.get_finished_spans() if s.name == "idp_eval.evaluate"]
    assert len(roots) == 2  # one independent root trace per case
    assert len({s.context.trace_id for s in roots}) == 2


def test_evaluate_many_excel_rows_attributable(tmp_path):
    path = tmp_path / "many.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(ReusableSourceJudge())],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate_many(_cases())
    _, rows = _read(path)
    assert [r["key_id"] for r in rows] == ["theme-1:epic-1", "theme-2:epic-2"]
    assert all(r["metric"] == "coverage" for r in rows)


def test_evaluate_many_metric_subset_applied_to_every_case():
    framework = EvaluationFramework(
        evaluators=[
            CoverageEvaluator(ReusableSourceJudge()),
            InstructionAdherenceEvaluator(object()),  # not selected -> never called
        ]
    )
    results = framework.evaluate_many(_cases(), metrics=["coverage"])
    assert all(set(r) == {"coverage"} for r in results)


def test_evaluate_many_fails_fast_with_case_aware_error():
    judge = ReusableSourceJudge()
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(judge)]
    )
    cases = [
        EvaluationCase(context="c", output="o", case_id="ok-1"),
        EvaluationCase(output="o", case_id="bad-2"),  # no context
    ]
    with pytest.raises(ValueError, match="Case bad-2:.*requires non-empty `context`"):
        framework.evaluate_many(cases)
    # Fail fast: no judge calls happened for the valid earlier case either.
    assert judge.calls == 0


# --- grouped convenience ----------------------------------------------------


def _groups():
    return [
        {
            "context": {"description": "Theme 1"},
            "outputs": [{"title": "Epic 1"}, {"title": "Epic 2"}],
            "group_id": "theme-1",
        },
        {
            "context": {"description": "Theme 2"},
            "outputs": [{"title": "Epic 3"}],
            "group_id": "theme-2",
        },
    ]


def test_evaluate_groups_fans_out_to_three_cases(spans):
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(ReusableSourceJudge())]
    )
    results = framework.evaluate_groups(_groups())
    assert len(results) == 3  # 2 + 1 flattened outputs
    roots = [s for s in spans.get_finished_spans() if s.name == "idp_eval.evaluate"]
    assert len(roots) == 3  # one trace per flattened output


def test_evaluate_groups_deterministic_case_ids(tmp_path):
    path = tmp_path / "groups.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(ReusableSourceJudge())],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate_groups(_groups())
    _, rows = _read(path)
    assert [r["key_id"] for r in rows] == [
        "theme-1:0", "theme-1:1", "theme-2:0",
    ]


def test_evaluate_groups_returns_one_result_per_output_in_order():
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(ReusableSourceJudge())]
    )
    results = framework.evaluate_groups(_groups())
    # Fans out to 3 independent outputs (2 + 1), one result mapping each, in order.
    assert len(results) == 3
    assert all(set(r) == {"coverage"} for r in results)
    assert all(r["coverage"].metric == "coverage" for r in results)


def test_group_singular_output_key_is_rejected():
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(ReusableSourceJudge())]
    )
    with pytest.raises(ValueError, match="use 'outputs'|singular 'output'"):
        framework.evaluate_groups([{"context": "c", "output": "epic"}])


def test_group_case_ids_override_is_preserved(tmp_path):
    path = tmp_path / "override.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(ReusableSourceJudge())],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate_groups([
        {
            "context": "c",
            "outputs": ["e1", "e2"],
            "group_id": "g",
            "case_ids": ["custom-a", "custom-b"],
        }
    ])
    _, rows = _read(path)
    assert [r["key_id"] for r in rows] == ["custom-a", "custom-b"]


def test_group_preserves_generic_structured_fields_and_metadata_without_mutation():
    group = {
        "group_id": "request-1",
        "input": {"operation": "compare", "options": [1, 2]},
        "context": {
            "service_limits": {"max_latency": "2 seconds", "region": "US"},
            "requirements": ["99.9% availability", False, None],
        },
        "instructions": ["Use concise sections", {"max_items": 3}],
        "outputs": [
            {"summary": "First", "actions": ["A", "B"]},
            {"summary": "Second", "actions": [{"name": "C"}]},
        ],
        "metadata": {"source": "benchmark", "tags": ["smoke"]},
    }
    original = copy.deepcopy(group)
    evaluator = CapturingEvaluator()

    results = EvaluationFramework([evaluator]).evaluate_groups([group])

    assert len(results) == 2
    assert [case.case_id for case in evaluator.cases] == [
        "request-1:0",
        "request-1:1",
    ]
    assert all(case.input == group["input"] for case in evaluator.cases)
    assert all(case.context == group["context"] for case in evaluator.cases)
    assert all(case.instructions == group["instructions"] for case in evaluator.cases)
    assert [case.output for case in evaluator.cases] == group["outputs"]
    assert all(
        case.metadata
        == {"source": "benchmark", "tags": ["smoke"], "group_id": "request-1"}
        for case in evaluator.cases
    )
    assert evaluator.cases[0].metadata is not evaluator.cases[1].metadata
    assert group == original


def test_group_without_group_id_uses_stable_fallback_case_ids():
    evaluator = CapturingEvaluator()
    groups = [
        {"context": "first", "outputs": ["a", "b"]},
        {"context": "second", "outputs": ["c"]},
    ]

    EvaluationFramework([evaluator]).evaluate_groups(groups)

    assert [case.case_id for case in evaluator.cases] == ["0:0", "0:1", "1:0"]


def test_group_metadata_is_not_rendered_into_coverage_prompt():
    class PromptJudge(ReusableSourceJudge):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def generate_object(self, prompt, schema):
            self.prompts.append(prompt)
            return super().generate_object(prompt, schema)

    judge = PromptJudge()
    EvaluationFramework([CoverageEvaluator(judge)]).evaluate_groups(
        [
            {
                "context": {"requirements": ["Keep response concise"]},
                "outputs": [{"summary": "Concise response"}],
                "metadata": {"private_marker": "DO_NOT_RENDER"},
            }
        ]
    )

    prompt_text = repr(judge.prompts[0])
    assert "DO_NOT_RENDER" not in prompt_text
    assert "private_marker" not in prompt_text


def test_group_rejects_non_mapping_metadata():
    framework = EvaluationFramework([CapturingEvaluator()])
    with pytest.raises(ValueError, match="metadata.*mapping"):
        framework.evaluate_groups(
            [{"context": "c", "outputs": ["o"], "metadata": ["not", "mapping"]}]
        )
