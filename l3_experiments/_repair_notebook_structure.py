from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    ROOT / "01_theme_stage.ipynb",
    ROOT / "02_full_context.ipynb",
    ROOT / "03_no_theme_description.ipynb",
    ROOT / "04_no_theme.ipynb",
    ROOT / "05_full_with_hierarchy.ipynb",
]

EVALUATION_MARKDOWN = '''## Batch execution and evaluation

Predictions are produced before ground truth is loaded.

Only **valid evaluation Epics** are scored. An Epic is valid only when:

1. it has non-empty Jira L3 ground truth;
2. prediction/retrieval completed successfully;
3. it has a Value Stream Stage and candidate L3s; and
4. **every ground-truth L3 is present in the candidate set supplied to the LLM**.

Rows that fail any of these checks are kept only as diagnostics and do not contribute to exact match, precision, recall, or F1.
'''

EVALUATION_SOURCE = '''def ground_truth_by_epic():
    ground_truth = read_table(GROUND_TRUTH_PATH).copy()
    ground_truth["l3_capability_id"] = (
        ground_truth["l3_capability_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Blank/no-L3 rows are unlabeled, not negative examples.
    ground_truth = ground_truth.loc[
        ground_truth["l3_capability_id"].ne("")
    ]

    return {
        epic_key: set(group["l3_capability_id"])
        for epic_key, group in ground_truth.groupby(
            "epic_key",
            sort=False,
        )
    }


def empty_metrics(predicted_count, truth_count):
    return {
        "exact_match": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "predicted_count": predicted_count,
        "truth_count": truth_count,
    }


def evaluate_predictions(prediction_frame):
    truth_by_epic = ground_truth_by_epic()
    result_rows = []

    for row in prediction_frame.to_dict(orient="records"):
        predicted_ids = set(json.loads(row["predicted_l3_ids"]))
        available_candidate_ids = set(
            json.loads(row["available_candidate_l3_ids"])
        )
        truth_ids = truth_by_epic.get(row["epic_key"])

        if truth_ids is None:
            available_truth_ids = None
            availability_fraction = None
            exclusion_reason = "missing_ground_truth"
        else:
            available_truth_ids = truth_ids & available_candidate_ids
            availability_fraction = (
                len(available_truth_ids) / len(truth_ids)
            )

            if row["status"] == "error":
                exclusion_reason = "error"
            elif row["status"] == "no_stage":
                exclusion_reason = "no_stage"
            elif row["status"] == "no_candidates" or not available_candidate_ids:
                exclusion_reason = "no_candidates"
            elif not truth_ids.issubset(available_candidate_ids):
                # GT and Stage→L3 candidate mapping are inconsistent.
                # Do not use this Epic to judge the LLM selector.
                exclusion_reason = "gt_not_fully_retrievable"
            else:
                exclusion_reason = ""

        evaluation_eligible = exclusion_reason == ""

        if evaluation_eligible:
            metrics = score_sets(predicted_ids, truth_ids)
        else:
            metrics = empty_metrics(
                predicted_count=len(predicted_ids),
                truth_count=(len(truth_ids) if truth_ids is not None else None),
            )

        row["ground_truth_l3_ids"] = (
            json.dumps(sorted(truth_ids))
            if truth_ids is not None
            else None
        )
        row["gt_available_candidate_l3_ids"] = (
            json.dumps(sorted(available_truth_ids))
            if available_truth_ids is not None
            else None
        )
        row["gt_candidate_available_count"] = (
            len(available_truth_ids)
            if available_truth_ids is not None
            else None
        )
        row["gt_candidate_availability_fraction"] = availability_fraction
        row["evaluation_eligible"] = evaluation_eligible
        row["evaluation_exclusion_reason"] = exclusion_reason
        row.update(metrics)
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def evaluation_summary(results):
    valid = results.loc[results["evaluation_eligible"]].copy()

    summary = pd.DataFrame([
        {
            "scope": "valid_evaluation_population",
            "evaluated_epics": len(valid),
            "exact_match_accuracy": (
                valid["exact_match"].mean() if len(valid) else 0.0
            ),
            "mean_precision": (
                valid["precision"].mean() if len(valid) else 0.0
            ),
            "mean_recall": (
                valid["recall"].mean() if len(valid) else 0.0
            ),
            "mean_f1": (
                valid["f1"].mean() if len(valid) else 0.0
            ),
        }
    ])

    diagnostics = pd.DataFrame([
        {
            "prediction_rows": len(results),
            "valid_evaluation_epics": len(valid),
            "excluded_from_evaluation": int((~results["evaluation_eligible"]).sum()),
            "missing_ground_truth": int(
                (results["evaluation_exclusion_reason"] == "missing_ground_truth").sum()
            ),
            "no_stage": int(
                (results["evaluation_exclusion_reason"] == "no_stage").sum()
            ),
            "no_candidates": int(
                (results["evaluation_exclusion_reason"] == "no_candidates").sum()
            ),
            "gt_not_fully_retrievable": int(
                (
                    results["evaluation_exclusion_reason"]
                    == "gt_not_fully_retrievable"
                ).sum()
            ),
            "errors": int(
                (results["evaluation_exclusion_reason"] == "error").sum()
            ),
        }
    ])

    return summary, diagnostics


predictions, llm_calls = run_predictions()
results = evaluate_predictions(predictions)
summary, diagnostics = evaluation_summary(results)
llm_call_summary = summarize_llm_calls(llm_calls)

print("\\nEvaluation summary — valid Epics only")
display(summary)

print("\\nExcluded-row diagnostics")
display(diagnostics)

print("\\nLLM latency / token summary")
display(llm_call_summary)

print("\\nPer-call LLM metrics")
display(llm_calls.head(50))

display(results.head(20))

output_path = save_results_excel(
    results,
    EXPERIMENT_NAME,
    "results",
    extra_sheets={
        "evaluation_summary": summary,
        "diagnostics": diagnostics,
        "llm_calls": llm_calls,
        "llm_call_summary": llm_call_summary,
    },
)
print(f"Saved {output_path}")
'''


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def system_prompt_value(notebook):
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = source(cell)
        if "SYSTEM_PROMPT" not in text or "def build_user_prompt" not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT"
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError("SYSTEM_PROMPT missing")


def patch_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))

    eval_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
        and "def ground_truth_by_epic" in source(cell)
    )
    notebook["cells"][eval_index]["source"] = EVALUATION_SOURCE

    if eval_index > 0 and notebook["cells"][eval_index - 1].get("cell_type") == "markdown":
        notebook["cells"][eval_index - 1]["source"] = EVALUATION_MARKDOWN

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return notebook


def validate_notebook(path, notebook):
    all_code = "\n".join(
        source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "end_to_end_labeled" not in all_code
    assert "selector_only_fully_retrievable" not in all_code
    assert "selector_eligible" not in all_code
    assert "selector_exclusion_reason" not in all_code
    assert "evaluation_eligible" in all_code
    assert "evaluation_exclusion_reason" in all_code
    assert "valid_evaluation_population" in all_code
    assert "truth_ids.issubset(available_candidate_ids)" in all_code

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{path.name}:cell-{index}", "exec")


def main():
    prompts = []

    for path in NOTEBOOKS:
        notebook = patch_notebook(path)
        validate_notebook(path, notebook)
        prompts.append(system_prompt_value(notebook))

    assert len(set(prompts)) == 1, "SYSTEM_PROMPT differs across experiments"
    print("Updated all five notebooks to score only valid evaluation Epics.")


if __name__ == "__main__":
    main()
