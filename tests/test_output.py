"""Tests for evaluation output: records, writers, Excel, and custom logging."""

import json

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    EvaluationResult,
    InstructionAdherenceEvaluator,
    PersistenceError,
    CoverageEvaluator,
    TaskCoverageEvaluator,
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


class CountingScriptedJudge:
    """Two-stage coverage judge that counts generate_object calls."""

    def __init__(self):
        self.calls = 0

    def generate_object(self, prompt, schema):
        self.calls += 1
        if self.calls == 1:
            return {"requirements": [{"requirement": "a"}, {"requirement": "b"}]}
        return {"requirements": [
            {"id": "r1", "meaningfully_present": True, "fully_present": True, "reason": "r"},
            {"id": "r2", "meaningfully_present": False, "fully_present": False, "reason": "r"},
        ]}


CASE = EvaluationCase(case_id="gt-001", input="task", context="ctx", output="out")


def _read_xlsx(path, sheet="evaluations"):
    workbook = openpyxl.load_workbook(path)
    rows = list(workbook[sheet].iter_rows(values_only=True))
    header, *data = rows
    return header, data


def _sheet_names(path):
    return openpyxl.load_workbook(path).sheetnames


def _rows_as_dicts(path, sheet):
    header, data = _read_xlsx(path, sheet)
    return [dict(zip(header, row)) for row in data]


# --- writer construction & modes --------------------------------------------


def test_build_writers_modes(tmp_path):
    assert build_writers(None, None) == []
    assert len(build_writers("phoenix", None)) == 1
    assert len(build_writers("both", str(tmp_path / "x.xlsx"))) == 2
    excel_only = build_writers("excel", str(tmp_path / "x.xlsx"))
    assert isinstance(excel_only[0], ExcelEvaluationWriter)


def test_build_writers_invalid_mode():
    with pytest.raises(ValueError, match="Unknown output"):
        build_writers("elsewhere", None)


def test_build_writers_excel_requires_path():
    with pytest.raises(ValueError, match="excel_path is required"):
        build_writers("excel", None)


def test_framework_excel_requires_path():
    with pytest.raises(ValueError, match="excel_path is required"):
        EvaluationFramework(evaluators=[], output="both")


# --- annotator kinds --------------------------------------------------------


def test_annotator_kinds_supported():
    assert set(ANNOTATOR_KINDS) == {"LLM", "CODE", "HUMAN"}
    for kind in ANNOTATOR_KINDS:
        assert validate_annotator_kind(kind) == kind


def test_annotator_kind_rejected():
    with pytest.raises(ValueError, match="Unsupported annotator_kind"):
        validate_annotator_kind("ROBOT")


def test_record_from_result_validates_kind():
    result = EvaluationResult("m", 1.0, "ok", "why", {"k": "v"})
    with pytest.raises(ValueError):
        EvaluationRecord.from_result(result, annotator_kind="ROBOT")


# --- excel output -----------------------------------------------------------


def _source_judge():
    return ScriptedTwoStage(
        {
            "items": [
                {
                    "source_item": "Keep the identity provider.",
                    "meaningfully_present": True,
                    "fully_present": False,
                }
            ]
        }
    )


def _instruction_judge():
    return ScriptedTwoStage(
        {"instructions": [{"instruction": "Use 3 bullets"},
                          {"instruction": "Do not mention pricing"}]},
        {"answers": [
            {"id": "I1", "status": "followed", "reason": "Three bullets."},
            {"id": "I2", "status": "violated", "reason": "Mentions pricing."},
        ]},
    )


class ScriptedTwoStage:
    """Returns queued two-stage responses in order."""

    def __init__(self, *responses):
        self._responses = list(responses)

    def generate_object(self, prompt, schema):
        return self._responses.pop(0)


def test_summary_sheet_columns_and_values(tmp_path):
    path = tmp_path / "evals.xlsx"
    framework = EvaluationFramework(
        evaluators=[TaskCoverageEvaluator(CountingScriptedJudge())],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(CASE, run_name="benchmark-v1", dataset_name="theme-epic-gt")

    header, data = _read_xlsx(path)
    assert header == (
        "run_name", "dataset_name", "case_id", "trace_id", "metric", "score",
        "label", "explanation", "annotator_kind", "timestamp",
        "raw_details_json",
    )
    assert "details_json" not in header  # opaque main column is gone
    assert len(data) == 1
    row = dict(zip(header, data[0]))
    assert row["run_name"] == "benchmark-v1"
    assert row["dataset_name"] == "theme-epic-gt"
    assert row["case_id"] == "gt-001"
    assert row["metric"] == "task_coverage"
    assert row["score"] == 0.5  # stored as a numeric cell
    assert isinstance(row["score"], float)
    assert row["annotator_kind"] == "LLM"
    # Lossless raw JSON is retained in the trailing optional column.
    assert json.loads(row["raw_details_json"])["total_requirements"] == 2


def test_task_coverage_items_sheet(tmp_path):
    path = tmp_path / "tc.xlsx"
    framework = EvaluationFramework(
        evaluators=[TaskCoverageEvaluator(CountingScriptedJudge())],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(CASE, run_name="r1", dataset_name="d1")

    assert "task_coverage_items" in _sheet_names(path)
    header, _ = _read_xlsx(path, "task_coverage_items")
    assert header == (
        "run_name", "dataset_name", "case_id", "trace_id", "metric",
        "item_id", "requirement", "meaningfully_present", "fully_present",
        "status", "item_score", "reason",
    )
    rows = _rows_as_dicts(path, "task_coverage_items")
    assert [r["item_id"] for r in rows] == ["r1", "r2"]
    assert rows[0]["requirement"] == "a"
    assert rows[0]["status"] == "covered"
    assert rows[0]["item_score"] == 1.0
    assert rows[0]["meaningfully_present"] is True
    assert rows[1]["status"] == "missing"
    assert rows[1]["item_score"] == 0.0
    # Identity columns are carried onto every item row.
    assert rows[0]["case_id"] == "gt-001" and rows[0]["metric"] == "task_coverage"


def test_coverage_items_sheet(tmp_path):
    path = tmp_path / "sc.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_source_judge(), mode="g_eval", verbose=True)],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(CASE)

    assert "coverage_items" in _sheet_names(path)
    header, _ = _read_xlsx(path, "coverage_items")
    assert header == (
        "run_name", "dataset_name", "case_id", "trace_id", "metric",
        "item_id", "source_item", "meaningfully_present", "fully_present",
        "status", "item_score", "reason",
    )
    rows = _rows_as_dicts(path, "coverage_items")
    assert rows[0]["source_item"] == "Keep the identity provider."
    assert rows[0]["status"] == "partial"
    assert rows[0]["item_score"] == 0.5
    details = json.loads(_rows_as_dicts(path, "evaluations")[0]["raw_details_json"])
    assert details["judge_call_count"] == 1
    assert details["final_item_count"] == 1
    assert details["mode"] == "g_eval"
    assert "classify_ms" in details and "total_ms" in details


def test_instruction_adherence_items_sheet(tmp_path):
    path = tmp_path / "ia.xlsx"
    case = EvaluationCase(
        case_id="gt-002", input="t", context="c", output="o",
        instructions="Use 3 bullets. Do not mention pricing.",
    )
    framework = EvaluationFramework(
        evaluators=[InstructionAdherenceEvaluator(_instruction_judge())],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(case)

    assert "instruction_adherence_items" in _sheet_names(path)
    header, _ = _read_xlsx(path, "instruction_adherence_items")
    assert header == (
        "run_name", "dataset_name", "case_id", "trace_id", "metric",
        "instruction_id", "instruction", "status", "item_score", "reason",
    )
    rows = _rows_as_dicts(path, "instruction_adherence_items")
    assert [r["instruction_id"] for r in rows] == ["I1", "I2"]
    assert rows[0]["status"] == "followed" and rows[0]["item_score"] == 1.0
    assert rows[1]["status"] == "violated" and rows[1]["item_score"] == 0.0


def test_multiple_metrics_share_one_workbook(tmp_path):
    path = tmp_path / "multi.xlsx"
    case = EvaluationCase(
        case_id="gt-003", input="t", context="c", output="o",
        instructions="Use 3 bullets. Do not mention pricing.",
    )
    framework = EvaluationFramework(
        evaluators=[
            TaskCoverageEvaluator(CountingScriptedJudge()),
            InstructionAdherenceEvaluator(_instruction_judge()),
        ],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(case)

    names = _sheet_names(path)
    assert names[0] == "evaluations"
    assert "task_coverage_items" in names
    assert "instruction_adherence_items" in names
    # Two metric rows in the summary, one workbook.
    _, summary = _read_xlsx(path)
    assert {row[4] for row in summary} == {"task_coverage", "instruction_adherence"}


def test_multiple_cases_append_to_item_sheet(tmp_path):
    path = tmp_path / "append.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_source_judge(), mode="g_eval", verbose=True)],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(EvaluationCase(case_id="c1", input="t", context="c", output="o"))
    # Second case reuses the SAME writer so rows accumulate in one workbook; a
    # fresh scripted judge answers the new case's two stages.
    framework._evaluators["coverage"] = CoverageEvaluator(_source_judge(), mode="g_eval", verbose=True)
    framework.evaluate(EvaluationCase(case_id="c2", input="t", context="c", output="o"))

    rows = _rows_as_dicts(path, "coverage_items")
    assert [r["case_id"] for r in rows] == ["c1", "c2"]


def test_faithfulness_has_no_item_sheet(tmp_path):
    path = tmp_path / "faith.xlsx"
    framework = EvaluationFramework(output="excel", excel_path=str(path), evaluators=[])
    # Faithfulness result carries details=None -> summary only, no item sheet.
    framework.log_evaluation(
        EvaluationResult("faithfulness", 1.0, "faithful", "Grounded.", None),
        case_id="gt-001", annotator_kind="LLM",
    )
    assert _sheet_names(path) == ["evaluations"]


def test_workbook_is_valid_and_reloadable(tmp_path):
    path = tmp_path / "valid.xlsx"
    framework = EvaluationFramework(
        evaluators=[TaskCoverageEvaluator(CountingScriptedJudge())],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(CASE)
    workbook = openpyxl.load_workbook(path)  # raises if corrupt
    summary = workbook["evaluations"]
    assert summary.freeze_panes == "A2"  # header row frozen
    assert summary.auto_filter.ref is not None  # filter applied
    assert summary["A1"].font.bold is True  # bold header


# --- Excel + coverage verbosity / async bulk --------------------------------


def _stateless_source_judge(*, reasons):
    """One call returns item 1 partial and item 2 missing (score 0.25)."""

    class _Judge:
        def generate_object(self, prompt, schema):
            items = [
                {
                    "source_item": "Keep IdP",
                    "meaningfully_present": True,
                    "fully_present": False,
                },
                {
                    "source_item": "Support SSO",
                    "meaningfully_present": False,
                    "fully_present": False,
                },
            ]
            if reasons:
                items[0]["reason"] = "qualifier missing"
                items[1]["reason"] = "SSO absent"
            return {"items": items}

    return _Judge()


def test_excel_reason_blank_in_compact_mode(tmp_path):
    path = tmp_path / "compact.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_stateless_source_judge(reasons=False), mode="g_eval", verbose=True)],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(EvaluationCase(context="c", output="o", case_id="k1"))
    rows = _rows_as_dicts(path, "coverage_items")
    assert [r["status"] for r in rows] == ["partial", "missing"]
    assert all(r["reason"] in (None, "") for r in rows)  # blank, not invented


def test_excel_reason_written_in_verbose_mode(tmp_path):
    path = tmp_path / "verbose.xlsx"
    framework = EvaluationFramework(
        evaluators=[
            CoverageEvaluator(
                _stateless_source_judge(reasons=True), mode="g_eval", verbose=True
            )
        ],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(EvaluationCase(context="c", output="o", case_id="k1"))
    summary = _rows_as_dicts(path, "evaluations")
    assert summary[0]["score"] == 0.25  # same score regardless of verbosity
    rows = _rows_as_dicts(path, "coverage_items")
    assert rows[0]["reason"] == "qualifier missing"
    assert rows[1]["reason"] == "SSO absent"


def test_async_evaluate_many_writes_per_case_rows(tmp_path):
    import asyncio

    class _AllCovered:
        def generate_object(self, prompt, schema):
            return {
                "items": [
                    {
                        "source_item": source_item,
                        "meaningfully_present": True,
                        "fully_present": True,
                    }
                    for source_item in ("A", "B")
                ]
            }

    path = tmp_path / "async.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_AllCovered(), mode="g_eval", verbose=True)],
        output="excel",
        excel_path=str(path),
    )
    cases = [
        EvaluationCase(context="c", output="o", case_id="a1"),
        EvaluationCase(context="c", output="o", case_id="a2"),
    ]
    asyncio.run(framework.a_evaluate_many(cases))

    summary = _rows_as_dicts(path, "evaluations")
    assert [r["case_id"] for r in summary] == ["a1", "a2"]  # order preserved
    item_rows = _rows_as_dicts(path, "coverage_items")
    assert [r["case_id"] for r in item_rows] == ["a1", "a1", "a2", "a2"]


class _RecordingWriter:
    def __init__(self):
        self.writes = 0

    def write(self, records):
        self.writes += 1


def test_multiple_writers_evaluate_once(tmp_path):
    # Two writers must not cause the evaluators to run twice.
    judge = CountingScriptedJudge()
    framework = EvaluationFramework(
        evaluators=[TaskCoverageEvaluator(judge)],
        output="excel",
        excel_path=str(tmp_path / "e.xlsx"),
    )
    second = _RecordingWriter()
    framework._writers.append(second)
    framework.evaluate(CASE)
    # Two-stage coverage = exactly 2 judge calls, not 4 (no re-eval per sink).
    assert judge.calls == 2
    assert second.writes == 1  # each writer received the records once


def test_no_output_writes_no_file(tmp_path):
    path = tmp_path / "should_not_exist.xlsx"
    framework = EvaluationFramework(
        evaluators=[TaskCoverageEvaluator(CountingScriptedJudge())]
    )
    framework.evaluate(CASE)
    assert not path.exists()


def test_batch_writes_one_row_per_case_metric(tmp_path):
    path = tmp_path / "batch.xlsx"
    framework = EvaluationFramework(
        evaluators=[TaskCoverageEvaluator(CountingScriptedJudge())],
        output="excel",
        excel_path=str(path),
    )
    # Fresh judge per case via separate frameworks is simplest; here reuse one
    # framework but note each evaluate needs its own two-response judge.
    cases = [
        EvaluationCase(case_id="c1", input="t", context="c", output="o"),
    ]
    framework.evaluate_batch(cases)
    _, data = _read_xlsx(path)
    assert len(data) == 1


# --- custom evaluation logging ----------------------------------------------


def test_log_custom_evaluation_to_excel(tmp_path):
    path = tmp_path / "custom.xlsx"
    framework = EvaluationFramework(
        evaluators=[], output="excel", excel_path=str(path)
    )
    framework.log_custom_evaluation(
        name="company_policy",
        score=1.0,
        label="pass",
        explanation="All required company rules were satisfied.",
        details={"rule_version": "v3"},
        kind="CODE",
        case_id="gt-001",
    )
    header, data = _read_xlsx(path)
    row = dict(zip(header, data[0]))
    assert row["metric"] == "company_policy"
    assert row["annotator_kind"] == "CODE"
    assert row["case_id"] == "gt-001"
    # Unknown metric: no structured sheet, but details survive in raw JSON.
    assert _sheet_names(path) == ["evaluations"]
    assert json.loads(row["raw_details_json"])["rule_version"] == "v3"


def test_log_evaluation_human_kind(tmp_path):
    path = tmp_path / "human.xlsx"
    framework = EvaluationFramework(
        evaluators=[], output="excel", excel_path=str(path)
    )
    result = EvaluationResult("manual_review", 0.0, "fail", "Reviewer rejected.")
    framework.log_evaluation(result, case_id="gt-002", annotator_kind="HUMAN")
    _, data = _read_xlsx(path)
    assert dict(zip(_read_xlsx(path)[0], data[0]))["annotator_kind"] == "HUMAN"


def test_log_custom_evaluation_invalid_kind(tmp_path):
    framework = EvaluationFramework(
        evaluators=[], output="excel", excel_path=str(tmp_path / "x.xlsx")
    )
    with pytest.raises(ValueError, match="Unsupported annotator_kind"):
        framework.log_custom_evaluation(name="x", score=1.0, kind="BOT")


# --- error handling ---------------------------------------------------------


class _FailingWriter:
    def write(self, records):
        raise IOError("disk full")


def test_persistence_error_preserves_results():
    judge = CountingScriptedJudge()
    framework = EvaluationFramework(evaluators=[TaskCoverageEvaluator(judge)])
    framework._writers = [_FailingWriter()]  # inject a failing sink

    with pytest.raises(PersistenceError) as exc:
        framework.evaluate(CASE)

    # Results computed once and preserved on the exception; no re-evaluation.
    assert judge.calls == 2
    assert exc.value.results["task_coverage"].score == 0.5


# --- native Phoenix annotation mapping (offline, fake client) ---------------


class _FakeClient:
    def __init__(self):
        self.batches = []
        self.spans = self

    def log_span_annotations(self, span_annotations, sync=False):
        self.batches.append(span_annotations)


def _record(metric="task_coverage", score=0.625, label="partial", kind="LLM",
            span_id="0123456789abcdef", details=None):
    return EvaluationRecord.from_result(
        EvaluationResult(metric, score, label, "why", details),
        annotator_kind=kind,
        span_id=span_id,
        case_id="gt-001",
    )


def test_phoenix_annotation_mapping(monkeypatch):
    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda payload: payload)
    client = _FakeClient()
    PhoenixEvaluationWriter(client=client).write([_record(details={"k": "v"})])

    annotation = client.batches[0][0]
    assert annotation["name"] == "task_coverage"
    assert annotation["span_id"] == "0123456789abcdef"
    assert annotation["annotator_kind"] == "LLM"
    assert annotation["result"] == {
        "score": 0.625, "label": "partial", "explanation": "why",
    }
    assert annotation["metadata"] == {"k": "v"}


@pytest.mark.parametrize("kind", ["LLM", "CODE", "HUMAN"])
def test_phoenix_annotation_kinds(monkeypatch, kind):
    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda payload: payload)
    client = _FakeClient()
    PhoenixEvaluationWriter(client=client).write([_record(kind=kind)])
    assert client.batches[0][0]["annotator_kind"] == kind


def test_phoenix_annotation_score_none_not_coerced(monkeypatch):
    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda payload: payload)
    client = _FakeClient()
    PhoenixEvaluationWriter(client=client).write(
        [_record(score=None, label="not_applicable")]
    )
    result = client.batches[0][0]["result"]
    assert "score" not in result  # None is omitted, never coerced to 0.0
    assert result["label"] == "not_applicable"


def test_phoenix_annotations_batched(monkeypatch):
    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda payload: payload)
    client = _FakeClient()
    records = [
        _record(metric="task_coverage"),
        _record(metric="faithfulness", score=1.0, label="faithful"),
    ]
    PhoenixEvaluationWriter(client=client).write(records)
    assert len(client.batches) == 1  # one batched call
    assert len(client.batches[0]) == 2


def test_phoenix_writer_requires_span_id():
    record = EvaluationRecord.from_result(
        EvaluationResult("m", 1.0, "ok", "why"), annotator_kind="LLM"
    )  # no span_id
    with pytest.raises(RuntimeError, match="requires an active trace"):
        PhoenixEvaluationWriter(client=_FakeClient()).write([record])


def _http_404():
    import httpx

    request = httpx.Request("POST", "http://x/v1/span_annotations")
    return httpx.HTTPStatusError(
        "404", request=request, response=httpx.Response(404, request=request)
    )


def _http_500():
    import httpx

    request = httpx.Request("POST", "http://x/v1/span_annotations")
    return httpx.HTTPStatusError(
        "500", request=request, response=httpx.Response(500, request=request)
    )


class _FlakyClient:
    """Raises 404 (span-not-ingested) a few times, then succeeds."""

    def __init__(self, fails):
        self.calls = 0
        self.fails = fails
        self.spans = self

    def log_span_annotations(self, span_annotations, sync=False):
        self.calls += 1
        if self.calls <= self.fails:
            raise _http_404()
        return None


def test_phoenix_writer_retries_on_404_then_succeeds(monkeypatch):
    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda p: p)
    client = _FlakyClient(fails=2)
    writer = output.PhoenixEvaluationWriter(
        client=client, ingest_timeout=5, poll_interval=0.01
    )
    writer.write([_record()])
    assert client.calls == 3  # 2 x 404 then success


def test_phoenix_writer_gives_up_after_timeout(monkeypatch):
    import httpx

    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda p: p)
    writer = output.PhoenixEvaluationWriter(
        client=_FlakyClient(fails=1000), ingest_timeout=0.05, poll_interval=0.01
    )
    with pytest.raises(httpx.HTTPStatusError):
        writer.write([_record()])


def test_phoenix_writer_non_404_raises_immediately(monkeypatch):
    import httpx

    from idp_eval import output

    monkeypatch.setattr(output, "_make_span_annotation", lambda p: p)

    class _Boom:
        def __init__(self):
            self.calls = 0
            self.spans = self

        def log_span_annotations(self, span_annotations, sync=False):
            self.calls += 1
            raise _http_500()

    client = _Boom()
    writer = output.PhoenixEvaluationWriter(
        client=client, ingest_timeout=5, poll_interval=0.01
    )
    with pytest.raises(httpx.HTTPStatusError):
        writer.write([_record()])
    assert client.calls == 1  # non-404 is not retried


def test_payload_builds_real_span_annotation_data():
    """Our payload constructs a valid real ``SpanAnnotationData`` (if installed).

    Verifies compatibility with the actual Phoenix client types, not just the
    fake. Skips when ``arize-phoenix-client`` is not installed.
    """
    pytest.importorskip("phoenix.client")
    from idp_eval.output import _make_span_annotation

    record = _record(details={"total_requirements": 4})
    annotation = _make_span_annotation(PhoenixEvaluationWriter._payload(record))
    assert annotation["name"] == "task_coverage"
    assert annotation["span_id"] == "0123456789abcdef"
    assert annotation["annotator_kind"] == "LLM"
    assert annotation["result"] == {
        "score": 0.625, "label": "partial", "explanation": "why",
    }
    assert annotation["metadata"] == {"total_requirements": 4}
