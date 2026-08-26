"""Evaluation records, Phoenix annotations, and Excel output tests."""

import json

import pytest

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    EvaluationResult,
    Evaluator,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
    PersistenceError,
    RelevanceAtKEvaluator,
)
from idp_eval.output import (
    ANNOTATOR_KINDS,
    EvaluationRecord,
    ExcelEvaluationWriter,
    PhoenixEvaluationWriter,
    _json_for_excel,
    _truncate_excel_text,
    _truncate_nested_strings,
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
    # Default overall mode requires internal failure diagnostics even when the
    # test exercises compact (non-verbose) result details.
    items[0]["reason"] = "Qualifier missing."
    items[1]["reason"] = "SSO absent."
    return Judge(
        {
            "items": items,
            "overall_reason": "The identity-provider qualifier and SSO are absent.",
        }
    )


def _instruction_judge():
    return Judge(
        {
            "instructions": [
                {
                    "instruction": "Use 3 bullets",
                    "status": "followed",
                    "reason": "",
                }
            ]
        },
    )


def _faithfulness_judge():
    return Judge(
        {
            "claims": [
                {
                    "claim": "Cancellation takes 24 hours.",
                    "status": "supported",
                    "reason": "",
                },
                {
                    "claim": "Refunds are instant.",
                    "status": "unsupported",
                    "reason": "Context says five days.",
                },
            ],
            "overall_reason": "The instant-refund claim conflicts with context.",
        }
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
    assert rows[0]["explanation"] == (
        "The identity-provider qualifier and SSO are absent."
    )
    assert "2 of" not in rows[0]["explanation"]
    assert header == (
        "run_name",
        "dataset_name",
        "key_id",
        "input",
        "context",
        "output",
        "instructions",
        "metric",
        "status",
        "score",
        "label",
        "explanation",
        "timestamp",
        "raw_details_json",
    )
    row = rows[0]
    assert row["metric"] == "coverage"
    assert row["score"] == 0.25
    assert row["key_id"] == "gt-001"
    assert row["input"] is None
    assert row["context"] == "ctx"
    assert row["output"] == "out"
    assert row["status"] == "success"
    details = json.loads(row["raw_details_json"])
    assert details["judge_call_count"] == 1
    assert "items" not in details
    assert _sheet_names(path) == ["evaluations", "_idp_eval_checkpoint"]
    checkpoint = openpyxl.load_workbook(path)["_idp_eval_checkpoint"]
    assert checkpoint.sheet_state == "veryHidden"
    checkpoint_row = _read(path, "_idp_eval_checkpoint")[1][0]
    assert len(checkpoint_row["case_fingerprint"]) == 64
    assert len(checkpoint_row["evaluation_fingerprint"]) == 64


def test_summary_persists_rendered_descriptive_input_only_at_case_level(tmp_path):
    path = tmp_path / "input.xlsx"
    case = EvaluationCase(
        case_id="input-1",
        input={"task": "Generate recommendations", "count": 3},
        context="ctx",
        output="out",
    )
    EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge(), verbose=True)],
        output="excel",
        excel_path=str(path),
    ).evaluate(case)
    _, rows = _read(path)
    assert rows[0]["input"] == "Task: Generate recommendations\n\nCount: 3"
    assert "input" not in _read(path, "coverage_items")[0]


def test_summary_persists_all_rendered_case_audit_fields(tmp_path):
    path = tmp_path / "audit.xlsx"
    case = EvaluationCase(
        case_id="audit-1",
        input={"task": "Answer"},
        context={"facts": ["A", "B"]},
        output={"answer": "A"},
        instructions=["Be concise"],
        retrieved_documents=[{"text": "A", "score": 0.9}],
    )
    EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge(), verbose=True)],
        output="excel",
        excel_path=str(path),
        report_fields=[
            "input",
            "context",
            "output",
            "instructions",
            "retrieved_documents",
        ],
    ).evaluate(case)
    row = _read(path)[1][0]
    assert row["input"] == "Task: Answer"
    assert row["context"] == "Facts:\n- A\n- B"
    assert row["output"] == "Answer: A"
    assert row["instructions"] == "- Be concise"
    assert row["retrieved_documents"] == "- Text: A\n\n  Score: 0.9"


def test_report_fields_select_order_and_metadata_columns(tmp_path):
    path = tmp_path / "selected.xlsx"
    case = EvaluationCase(
        case_id="epic-1",
        context={"facts": ["A"]},
        output={"answer": "A"},
        metadata={"theme_id": "theme-7"},
    )
    EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge())],
        output="excel",
        excel_path=str(path),
        report_fields=["output", "context", "metadata.theme_id"],
    ).evaluate(case)
    header, rows = _read(path)
    assert header[:6] == (
        "run_name",
        "dataset_name",
        "key_id",
        "output",
        "context",
        "theme_id",
    )
    assert rows[0]["output"] == "Answer: A"
    assert rows[0]["context"] == "Facts:\n- A"
    assert rows[0]["theme_id"] == "theme-7"


def test_missing_selected_metadata_writes_empty_cell(tmp_path):
    path = tmp_path / "missing-metadata.xlsx"
    EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge())],
        output="excel",
        excel_path=str(path),
        report_fields=["metadata.theme_id"],
    ).evaluate(CASE)
    assert _read(path)[1][0]["theme_id"] is None


@pytest.mark.parametrize(
    "fields,match",
    [
        (["foo"], "Unknown report field"),
        (["case_id"], "Unknown report field"),
        (["context", "context"], "Duplicate report field"),
        (["metadata."], "one non-empty key"),
        (["metadata.a.b"], "one non-empty key"),
        (["metadata.metric"], "conflicts with visible Excel column"),
        (["input", "metadata.input"], "conflicts with visible Excel column"),
    ],
)
def test_report_field_validation(fields, match):
    with pytest.raises(ValueError, match=match):
        EvaluationFramework(evaluators=[], report_fields=fields)


def test_visible_sheets_hide_technical_identity_columns(tmp_path):
    path = tmp_path / "clean.xlsx"
    EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge(), verbose=True)],
        output="excel",
        excel_path=str(path),
    ).evaluate(CASE)
    technical = {
        "case_id",
        "trace_id",
        "case_fingerprint",
        "evaluation_fingerprint",
        "annotator_kind",
    }
    assert technical.isdisjoint(_read(path)[0])
    assert technical.isdisjoint(_read(path, "coverage_items")[0])
    checkpoint = openpyxl.load_workbook(path)["_idp_eval_checkpoint"]
    assert checkpoint.sheet_state == "veryHidden"
    assert technical.issubset(set(_read(path, "_idp_eval_checkpoint")[0]))


def test_resume_rejects_visible_report_schema_change_before_judge(tmp_path):
    path = tmp_path / "schema.xlsx"
    first_judge = _coverage_judge()
    EvaluationFramework(
        [CoverageEvaluator(first_judge)],
        output="excel",
        excel_path=str(path),
        resume=True,
        report_fields=["context", "output"],
    ).evaluate(CASE)
    second_judge = _coverage_judge()
    with pytest.raises(ValueError, match="does not match"):
        EvaluationFramework(
            [CoverageEvaluator(second_judge)],
            output="excel",
            excel_path=str(path),
            resume=True,
            report_fields=["output"],
        )
    assert second_judge.calls == 0


def test_verbose_coverage_items_sheet(tmp_path):
    path = tmp_path / "coverage.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge(), verbose=True)],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(CASE)
    assert _sheet_names(path) == [
        "evaluations",
        "_idp_eval_checkpoint",
        "coverage_items",
    ]
    header, rows = _read(path, "coverage_items")
    assert header == (
        "run_name",
        "dataset_name",
        "key_id",
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
    assert _sheet_names(path) == [
        "evaluations",
        "_idp_eval_checkpoint",
        "coverage_items",
    ]


def test_instruction_adherence_sheet_remains(tmp_path):
    path = tmp_path / "instruction.xlsx"
    case = EvaluationCase(
        case_id="i1", instructions="Use 3 bullets", output="- a\n- b\n- c"
    )
    EvaluationFramework(
        evaluators=[
            InstructionAdherenceEvaluator(_instruction_judge(), verbose=True)
        ],
        output="excel",
        excel_path=str(path),
    ).evaluate(case)
    assert "instruction_adherence_items" in _sheet_names(path)


def test_verbose_faithfulness_items_sheet(tmp_path):
    path = tmp_path / "faithfulness.xlsx"
    EvaluationFramework(
        evaluators=[FaithfulnessEvaluator(_faithfulness_judge(), verbose=True)],
        output="excel",
        excel_path=str(path),
    ).evaluate(CASE)
    assert _sheet_names(path) == [
        "evaluations",
        "_idp_eval_checkpoint",
        "faithfulness_items",
    ]
    header, rows = _read(path, "faithfulness_items")
    assert header == (
        "run_name",
        "dataset_name",
        "key_id",
        "metric",
        "claim_id",
        "claim",
        "status",
        "item_score",
        "reason",
    )
    assert [row["claim_id"] for row in rows] == ["F1", "F2"]
    assert [row["item_score"] for row in rows] == [1.0, 0.0]


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
    assert [row["key_id"] for row in _read(path)[1]] == ["c1", "c2"]
    assert [row["key_id"] for row in _read(path, "coverage_items")[1]] == [
        "c1",
        "c1",
        "c2",
        "c2",
    ]


def test_default_non_resume_mode_preserves_append_behavior(tmp_path):
    path = tmp_path / "append.xlsx"
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(_coverage_judge(), verbose=True)],
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(CASE)
    framework.evaluate(CASE)
    assert len(_read(path)[1]) == 2
    assert len(_read(path, "coverage_items")[1]) == 4


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


def test_excel_safe_helpers_preserve_small_values_and_do_not_mutate_inputs():
    short = "short value"
    long_text = "x" * 40000
    original = {
        "short": short,
        "documents": [{"text": long_text}],
        "tuple": (long_text, 7, True, None),
    }

    assert _truncate_excel_text(None) is None
    assert _truncate_excel_text(short) == short
    truncated = _truncate_excel_text(long_text)
    assert len(truncated) <= 30000
    assert "[truncated " in truncated

    copied = _truncate_nested_strings(original, 1000)
    assert copied["short"] == short
    assert len(copied["documents"][0]["text"]) <= 1000
    assert len(copied["tuple"][0]) <= 1000
    assert copied["tuple"][1:] == (7, True, None)
    assert original["documents"][0]["text"] == long_text

    small_json = _json_for_excel({"value": short})
    assert small_json == json.dumps({"value": short})


def test_json_for_excel_uses_valid_fallback_instead_of_slicing():
    structurally_large = {f"key-{index}": index for index in range(10000)}
    serialized = _json_for_excel(structurally_large)

    assert len(serialized) <= 30000
    parsed = json.loads(serialized)
    assert parsed["_excel_truncated"] is True
    assert parsed["_original_json_chars"] > 32767
    assert "Evaluation used full data" in parsed["_note"]


class LargeDetailsEvaluator(Evaluator):
    name = "large_details"
    required_fields = ("context", "output")

    def __init__(self, document_text):
        self.document_text = document_text
        self.calls = 0
        self.seen = None

    def resume_signature(self):
        return {"contract_version": 1}

    def evaluate(self, case):
        self.calls += 1
        self.seen = (case.context, case.output, case.retrieved_documents)
        return EvaluationResult(
            metric=self.name,
            score=1.0,
            label="pass",
            explanation=self.document_text,
            details={
                "documents": [
                    {
                        "rank": 1,
                        "document_id": "doc-1",
                        "text": self.document_text,
                        "relevant": True,
                        "relevance_score": 1.0,
                        "reason": "Relevant document.",
                        "retrieval_score": 0.9,
                    }
                ]
            },
        )


def test_large_excel_values_remain_valid_visible_and_unmodified(tmp_path):
    path = tmp_path / "large.xlsx"
    document_text = "document-content-" * 4000
    context = "context-content-" * 3000
    output = "output-content-" * 3000
    retrieved_documents = [{"text": document_text, "score": 0.9}]
    case = EvaluationCase(
        case_id="large-1",
        context=context,
        output=output,
        retrieved_documents=retrieved_documents,
    )
    evaluator = LargeDetailsEvaluator(document_text)

    result = EvaluationFramework(
        [evaluator],
        output="excel",
        excel_path=str(path),
        report_fields=["context", "output", "retrieved_documents"],
    ).evaluate(case)["large_details"]

    assert evaluator.seen == (context, output, retrieved_documents)
    assert result.details["documents"][0]["text"] == document_text
    assert len(result.details["documents"][0]["text"]) == len(document_text)

    summary = _read(path)[1][0]
    checkpoint = _read(path, "_idp_eval_checkpoint")[1][0]
    for name in ("context", "output", "retrieved_documents", "explanation"):
        assert len(summary[name]) <= 30000
        assert "[truncated " in summary[name]
    parsed_summary_details = json.loads(summary["raw_details_json"])
    parsed_checkpoint_details = json.loads(checkpoint["raw_details_json"])
    parsed_retrieved = json.loads(checkpoint["retrieved_documents_json"])
    assert parsed_summary_details == parsed_checkpoint_details
    assert len(parsed_checkpoint_details["documents"][0]["text"]) < len(
        document_text
    )
    assert len(parsed_retrieved[0]["text"]) < len(document_text)

    header, rows = _read(path, "retrieval_documents")
    assert header == (
        "run_name",
        "dataset_name",
        "key_id",
        "rank",
        "document_id",
        "text",
        "relevant",
        "relevance_score",
        "reason",
        "retrieval_score",
    )
    retrieval_row = rows[0]
    assert retrieval_row["text"] != document_text
    assert "[truncated " in retrieval_row["text"]
    assert retrieval_row["relevant"] is True
    assert retrieval_row["relevance_score"] == 1.0
    assert retrieval_row["retrieval_score"] == 0.9


def test_resume_after_large_details_skips_evaluator_and_parses_json(tmp_path):
    path = tmp_path / "large-resume.xlsx"
    document_text = "large-document-" * 4000
    case = EvaluationCase(
        case_id="resume-large",
        context="authoritative context",
        output="generated output",
        retrieved_documents=[{"text": document_text}],
    )
    first = LargeDetailsEvaluator(document_text)
    first_result = EvaluationFramework(
        [first], output="excel", excel_path=str(path)
    ).evaluate(case)["large_details"]
    assert first_result.details["documents"][0]["text"] == document_text

    second = LargeDetailsEvaluator(document_text)
    resumed_result = EvaluationFramework(
        [second], output="excel", excel_path=str(path), resume=True
    ).evaluate(case)["large_details"]

    assert first.calls == 1
    assert second.calls == 0
    assert resumed_result.label == "pass"
    assert json.loads(
        _read(path, "_idp_eval_checkpoint")[1][0]["raw_details_json"]
    )
    assert len(resumed_result.details["documents"][0]["text"]) < len(
        document_text
    )


def test_excel_safety_does_not_truncate_retrieval_judge_input(tmp_path):
    class CapturingJudge:
        def __init__(self):
            self.prompt = None

        def generate_object(self, prompt, schema):
            self.prompt = prompt
            return {
                "documents": [
                    {"rank": 1, "relevant": True, "reason": "Relevant."}
                ]
            }

    document_text = "complete-judge-document-" * 2000
    judge = CapturingJudge()
    result = EvaluationFramework(
        [RelevanceAtKEvaluator(1, judge, verbose=True)],
        output="excel",
        excel_path=str(tmp_path / "judge-input.xlsx"),
    ).evaluate(
        EvaluationCase(
            case_id="judge-input",
            input="Find the relevant document.",
            retrieved_documents=[{"text": document_text}],
        )
    )["relevance_at_1"]

    assert document_text in str(judge.prompt)
    assert result.details["documents"][0]["text"] == document_text


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
    checkpoint_row = _read(path, "_idp_eval_checkpoint")[1][0]
    assert checkpoint_row["annotator_kind"] == "CODE"
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


def test_descriptive_input_is_not_duplicated_into_phoenix_annotation():
    record = EvaluationRecord.from_result(
        EvaluationResult("coverage", 1.0, "complete", "why"),
        annotator_kind="LLM",
        span_id="0123456789abcdef",
        input="A potentially long generation task.",
    )
    assert "input" not in PhoenixEvaluationWriter._payload(record)


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
