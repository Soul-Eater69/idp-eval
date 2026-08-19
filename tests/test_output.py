"""Evaluation records, Phoenix annotations, and Excel output tests."""

import json

import pytest

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    EvaluationResult,
    InstructionAdherenceEvaluator,
    PersistenceError,
)
from idp_eval.output import (
    ANNOTATOR_KINDS,
    EvaluationRecord,
    ExcelEvaluationWriter,
    PhoenixEvaluationWriter,
    build_writers,
    validate_annotator_kind,
)

openpyxl = pytest.importorskip("openpyxl")


CASE = EvaluationCase(case_id="gt-001", context="ctx", output="out")


class Judge:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_object(self, prompt, schema):
        self.calls += 1
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _coverage_judge(*, reasons=True):
    items = [
        {
            "source_item": "Keep the identity provider.",
            "meaningfully_present": True,
            "fully_present": False,
        },
        {
            "source_item": "Support SSO.",
            "meaningfully_present": False,
            "fully_present": False,
        },
    ]
    if reasons:
        items[0]["reason"] = "Qualifier missing."
        items[1]["reason"] = "SSO absent."
    return Judge({"items": items})


def _instruction_judge():
    return Judge(
        {"instructions": [{"instruction": "Use 3 bullets"}]},
        {
            "answers": [
                {"id": "I1", "status": "followed", "reason": "Three bullets."}
            ]
        },
    )


def _read(path, sheet="evaluations"):
    rows = list(openpyxl.load_workbook(path)[sheet].iter_rows(values_only=True))
    header, *data = rows
    return header, [dict(zip(header, row)) for row in data]


def _sheet_names(path):
    return openpyxl.load_workbook(path).sheetnames


def test_build_writers_modes_and_validation(tmp_path):
    assert build_writers(None, None) == []
    assert len(build_writers("phoenix", None)) == 1
    assert len(build_writers("both", str(tmp_path / "x.xlsx"))) == 2
    assert isinstance(
        build_writers("excel", str(tmp_path / "x.xlsx"))[0],
        ExcelEvaluationWriter,
    )
    with pytest.raises(ValueError, match="Unknown output"):
        build_writers("elsewhere", None)
    with pytest.raises(ValueError, match="excel_path is required"):
        build_writers("excel", None)


def test_annotator_kinds_and_record_validation():
    assert set(ANNOTATOR_KINDS) == {"LLM", "CODE", "HUMAN"}
    for kind in ANNOTATOR_KINDS:
        assert validate_annotator_kind(kind) == kind
    with pytest.raises(ValueError, match="Unsupported annotator_kind"):
        EvaluationRecord.from_result(
            EvaluationResult("m", 1.0, "ok", "why"),
            annotator_kind="ROBOT",
        )


def test_summary_sheet_and_compact_coverage_details(tmp_path):
    path = tmp_path / "evals.xlsx"
    judge = _coverage_judge(reasons=False)
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(judge)],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(CASE, run_name="run", dataset_name="dataset")
    header, rows = _read(path)
    assert header == (
        "run_name",
        "dataset_name",
        "case_id",
        "trace_id",
        "metric",
        "score",
        "label",
        "explanation",
        "annotator_kind",
        "timestamp",
        "raw_details_json",
    )
    row = rows[0]
    assert row["metric"] == "coverage"
    assert row["score"] == 0.25
    assert row["case_id"] == "gt-001"
    details = json.loads(row["raw_details_json"])
    assert details["judge_call_count"] == 1
    assert "items" not in details
    assert _sheet_names(path) == ["evaluations"]


def test_verbose_coverage_items_sheet(tmp_path):
    path = tmp_path / "coverage.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge(), verbose=True)],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(CASE)
    assert _sheet_names(path) == ["evaluations", "coverage_items"]
    header, rows = _read(path, "coverage_items")
    assert header == (
        "run_name",
        "dataset_name",
        "case_id",
        "trace_id",
        "metric",
        "item_id",
        "source_item",
        "meaningfully_present",
        "fully_present",
        "status",
        "item_score",
        "reason",
    )
    assert [row["item_id"] for row in rows] == ["S1", "S2"]
    assert [row["status"] for row in rows] == ["partial", "missing"]
    assert [row["item_score"] for row in rows] == [0.5, 0.0]
    assert rows[0]["reason"] == "Qualifier missing."
    assert all(row["metric"] == "coverage" for row in rows)


def test_only_current_coverage_sheets_are_created(tmp_path):
    path = tmp_path / "coverage.xlsx"
    EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge(), verbose=True)],
        output="excel",
        excel_path=str(path),
    ).evaluate(CASE)
    assert _sheet_names(path) == ["evaluations", "coverage_items"]


def test_instruction_adherence_sheet_remains(tmp_path):
    path = tmp_path / "instruction.xlsx"
    case = EvaluationCase(
        case_id="i1", instructions="Use 3 bullets", output="- a\n- b\n- c"
    )
    EvaluationFramework(
        evaluators=[InstructionAdherenceEvaluator(_instruction_judge())],
        output="excel",
        excel_path=str(path),
    ).evaluate(case)
    assert "instruction_adherence_items" in _sheet_names(path)


def test_multiple_cases_append_in_order(tmp_path):
    path = tmp_path / "many.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge(), verbose=True)],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate_many(
        [
            EvaluationCase(case_id="c1", context="c", output="o"),
            EvaluationCase(case_id="c2", context="c", output="o"),
        ]
    )
    assert [row["case_id"] for row in _read(path)[1]] == ["c1", "c2"]
    assert [row["case_id"] for row in _read(path, "coverage_items")[1]] == [
        "c1",
        "c1",
        "c2",
        "c2",
    ]


class RecordingWriter:
    def __init__(self):
        self.writes = 0

    def write(self, records):
        self.writes += 1


def test_multiple_writers_do_not_rerun_evaluator(tmp_path):
    judge = _coverage_judge(reasons=False)
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(judge)],
        output="excel",
        excel_path=str(tmp_path / "evals.xlsx"),
    )
    second = RecordingWriter()
    framework._writers.append(second)
    framework.evaluate(CASE)
    assert judge.calls == 1
    assert second.writes == 1


def test_custom_evaluations_use_same_excel_path(tmp_path):
    path = tmp_path / "custom.xlsx"
    framework = EvaluationFramework(
        evaluators=[], output="excel", excel_path=str(path)
    )
    framework.log_custom_evaluation(
        name="company_policy",
        score=1.0,
        label="pass",
        explanation="Passed.",
        details={"version": "v1"},
        kind="CODE",
        case_id="gt-001",
    )
    row = _read(path)[1][0]
    assert row["metric"] == "company_policy"
    assert row["annotator_kind"] == "CODE"
    assert json.loads(row["raw_details_json"])["version"] == "v1"


class FailingWriter:
    def write(self, records):
        raise IOError("disk full")


def test_persistence_failure_preserves_computed_result_without_rerun():
    judge = _coverage_judge(reasons=False)
    framework = EvaluationFramework(evaluators=[CoverageEvaluator(judge)])
    framework._writers = [FailingWriter()]
    with pytest.raises(PersistenceError) as exc:
        framework.evaluate(CASE)
    assert judge.calls == 1
    assert exc.value.results["coverage"].score == 0.25


class FakeClient:
    def __init__(self):
        self.batches = []
        self.spans = self

    def log_span_annotations(self, span_annotations, sync=False):
        self.batches.append(span_annotations)


def _record(
    metric="coverage",
    score=0.625,
    label="incomplete",
    kind="LLM",
    span_id="0123456789abcdef",
    details=None,
):
    return EvaluationRecord.from_result(
        EvaluationResult(metric, score, label, "why", details),
        annotator_kind=kind,
        span_id=span_id,
        case_id="gt-001",
    )


def test_phoenix_annotation_mapping_and_metadata(monkeypatch):
    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda payload: payload)
    client = FakeClient()
    PhoenixEvaluationWriter(client=client).write([_record(details={"k": "v"})])
    annotation = client.batches[0][0]
    assert annotation == {
        "name": "coverage",
        "span_id": "0123456789abcdef",
        "annotator_kind": "LLM",
        "result": {
            "score": 0.625,
            "label": "incomplete",
            "explanation": "why",
        },
        "metadata": {"k": "v"},
    }


@pytest.mark.parametrize("kind", ["LLM", "CODE", "HUMAN"])
def test_phoenix_annotation_kinds(monkeypatch, kind):
    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda payload: payload)
    client = FakeClient()
    PhoenixEvaluationWriter(client=client).write([_record(kind=kind)])
    assert client.batches[0][0]["annotator_kind"] == kind


def test_phoenix_score_none_is_omitted_and_annotations_are_batched(monkeypatch):
    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda payload: payload)
    client = FakeClient()
    PhoenixEvaluationWriter(client=client).write(
        [
            _record(score=None, label="not_applicable"),
            _record(metric="faithfulness", score=1.0, label="faithful"),
        ]
    )
    assert len(client.batches) == 1
    assert len(client.batches[0]) == 2
    assert "score" not in client.batches[0][0]["result"]


def test_phoenix_writer_requires_span_id():
    with pytest.raises(RuntimeError, match="requires an active trace"):
        PhoenixEvaluationWriter(client=FakeClient()).write(
            [
                EvaluationRecord.from_result(
                    EvaluationResult("m", 1.0, "ok", "why"),
                    annotator_kind="LLM",
                )
            ]
        )


def test_payload_builds_real_span_annotation_data():
    pytest.importorskip("phoenix.client")
    from idp_eval.output import _make_span_annotation

    annotation = _make_span_annotation(
        PhoenixEvaluationWriter._payload(_record(details={"items": 2}))
    )
    assert annotation["name"] == "coverage"
    assert annotation["result"]["score"] == 0.625
    assert annotation["metadata"] == {"items": 2}
