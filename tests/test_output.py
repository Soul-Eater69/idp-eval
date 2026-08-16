"""Tests for evaluation output: records, writers, Excel, and custom logging."""

import json

import pytest

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    EvaluationResult,
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


def _read_xlsx(path):
    workbook = openpyxl.load_workbook(path)
    sheet = workbook["evaluations"]
    rows = list(sheet.iter_rows(values_only=True))
    header, *data = rows
    return header, data


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


def test_excel_only_writes_expected_columns(tmp_path):
    path = tmp_path / "evals.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(CountingScriptedJudge())],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(CASE, run_name="benchmark-v1", dataset_name="theme-epic-gt")

    header, data = _read_xlsx(path)
    assert header == (
        "run_name", "dataset_name", "case_id", "trace_id", "metric", "score",
        "label", "explanation", "annotator_kind", "details_json", "timestamp",
    )
    assert len(data) == 1
    row = dict(zip(header, data[0]))
    assert row["run_name"] == "benchmark-v1"
    assert row["dataset_name"] == "theme-epic-gt"
    assert row["case_id"] == "gt-001"
    assert row["metric"] == "coverage"
    assert row["score"] == 0.5
    assert row["annotator_kind"] == "LLM"
    # Nested details land in a single JSON column.
    details = json.loads(row["details_json"])
    assert details["total_requirements"] == 2


class _RecordingWriter:
    def __init__(self):
        self.writes = 0

    def write(self, records):
        self.writes += 1


def test_multiple_writers_evaluate_once(tmp_path):
    # Two writers must not cause the evaluators to run twice.
    judge = CountingScriptedJudge()
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(judge)],
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
        evaluators=[CoverageEvaluator(CountingScriptedJudge())]
    )
    framework.evaluate(CASE)
    assert not path.exists()


def test_batch_writes_one_row_per_case_metric(tmp_path):
    path = tmp_path / "batch.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(CountingScriptedJudge())],
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
    assert json.loads(row["details_json"])["rule_version"] == "v3"


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
    framework = EvaluationFramework(evaluators=[CoverageEvaluator(judge)])
    framework._writers = [_FailingWriter()]  # inject a failing sink

    with pytest.raises(PersistenceError) as exc:
        framework.evaluate(CASE)

    # Results computed once and preserved on the exception; no re-evaluation.
    assert judge.calls == 2
    assert exc.value.results["coverage"].score == 0.5


# --- native Phoenix annotation mapping (offline, fake client) ---------------


class _FakeClient:
    def __init__(self):
        self.batches = []
        self.spans = self

    def log_span_annotations(self, span_annotations, sync=False):
        self.batches.append(span_annotations)


def _record(metric="coverage", score=0.625, label="partial", kind="LLM",
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
    assert annotation["name"] == "coverage"
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
        _record(metric="coverage"),
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
    assert annotation["name"] == "coverage"
    assert annotation["span_id"] == "0123456789abcdef"
    assert annotation["annotator_kind"] == "LLM"
    assert annotation["result"] == {
        "score": 0.625, "label": "partial", "explanation": "why",
    }
    assert annotation["metadata"] == {"total_requirements": 4}
