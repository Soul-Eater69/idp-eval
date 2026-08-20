"""Offline tests for instruction-adherence benchmark helpers."""

from scripts.instruction_adherence_benchmark import (
    diagnostic_metrics,
    estimate_calls,
    instruction_text,
    load_cases,
    macro_f1,
    majority_status,
    summarize_evaluation_runs,
)


def test_call_estimate_separates_evaluator_and_diagnostic_calls():
    assert estimate_calls(3, 3, end_to_end=True, diagnostics=True) == {
        "evaluation": 9,
        "end_to_end": 9,
        "evaluator": 18,
        "diagnostic": 9,
        "total": 27,
    }


def test_instruction_text_preserves_fixture_strings():
    assert instruction_text(["Return JSON.", "Be concise."]) == (
        "Return JSON.\nBe concise."
    )
    assert instruction_text("Return JSON.") == "Return JSON."


def test_gt_fixture_loads_without_modification():
    cases = load_cases("instructions_gt.json")
    assert len(cases) == 12
    assert cases[0]["case_id"] == "instr_001"
    assert cases[0]["gold_score"] == 1.0


def _diagnostic():
    return {
        "gold_instructions": [
            {
                "gold_id": "I1",
                "represented": True,
                "evaluated_ids": ["I1"],
                "qualifier_preserved": True,
                "reason": "matched",
            },
            {
                "gold_id": "I2",
                "represented": False,
                "evaluated_ids": [],
                "qualifier_preserved": False,
                "reason": "missed",
            },
        ],
        "evaluated_instructions": [
            {
                "id": "I1",
                "grounded": True,
                "gold_ids": ["I1"],
                "duplicate_of": "",
                "reason": "matched",
            },
            {
                "id": "I2",
                "grounded": False,
                "gold_ids": [],
                "duplicate_of": "",
                "reason": "invented",
            },
        ],
    }


def test_diagnostic_metrics_report_recall_precision_qualifiers_and_extras():
    result = diagnostic_metrics(_diagnostic())
    assert result["recall"] == 0.5
    assert result["precision"] == 0.5
    assert result["qualifier_preservation"] == 0.5
    assert result["invented_count"] == 1
    assert result["missed_gold_ids"] == ["I2"]


def test_evaluation_summary_aggregates_each_run():
    runs = [
        {
            "instructions": [{"id": "I1", "instruction": "Return JSON."}],
            "score": 1.0,
            "diagnostic_metrics": {
                "recall": 1.0,
                "precision": 1.0,
                "qualifier_preservation": 1.0,
                "duplicate_count": 0,
                "invented_count": 0,
                "missed_gold_ids": [],
                "invented_extracted_ids": [],
            },
        },
        {
            "instructions": [{"id": "I1", "instruction": "Return JSON."}],
            "score": 0.0,
            "diagnostic_metrics": {
                "recall": 0.5,
                "precision": 1.0,
                "qualifier_preservation": 0.5,
                "duplicate_count": 0,
                "invented_count": 0,
                "missed_gold_ids": ["I2"],
                "invented_extracted_ids": [],
            },
        },
    ]
    summary = summarize_evaluation_runs(runs)
    assert summary["recall"]["mean"] == 0.75
    assert summary["precision"]["stddev"] == 0.0
    assert summary["exact"]["mean_exact_jaccard"] == 1.0
    assert summary["score"]["mean"] == 0.5


def test_majority_status_and_macro_f1():
    assert majority_status(["followed", "violated", "followed"]) == (
        "followed",
        2 / 3,
    )
    assert macro_f1(
        ["followed", "violated"], ["followed", "violated"]
    ) == 1.0
