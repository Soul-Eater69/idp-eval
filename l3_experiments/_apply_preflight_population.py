from __future__ import annotations

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


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def cell_after_heading(notebook, heading):
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "markdown" and heading in source(cell):
            return notebook["cells"][index + 1]
    raise AssertionError(f"Missing heading: {heading}")


def markdown_with_heading(notebook, heading):
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "markdown" and heading in source(cell):
            return cell
    raise AssertionError(f"Missing heading: {heading}")


PREDICTION_SOURCE = '''def predict_for_stage(gateway, theme, epic, stage_id):
    stage = stage_context(stage_id)
    candidates = candidate_rows_for_stage(stage_id)
    if not candidates:
        return {
            "stage": stage,
            "candidates": [],
            "user_prompt": build_user_prompt(theme, epic, stage, []),
            "raw_response": None,
            "selections": [],
            "metrics": None,
        }

    user_prompt = build_user_prompt(theme, epic, stage, candidates)
    raw_response, metrics = call_llm_with_metrics(
        gateway,
        SYSTEM_PROMPT,
        user_prompt,
    )
    selections = validate_l3_response(
        parse_json_response(raw_response),
        [candidate["capability_id"] for candidate in candidates],
        allow_empty=True,
        max_selected=3,
    )
    return {
        "stage": stage,
        "candidates": candidates,
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "selections": selections,
        "metrics": metrics,
    }


def metric_text(value):
    return "n/a" if value is None else str(value)


def summarize_llm_calls(call_metrics):
    successful = call_metrics.loc[call_metrics["status"] == "ok"].copy()

    def numeric(column):
        return pd.to_numeric(successful[column], errors="coerce").dropna()

    latency = numeric("latency_seconds")
    input_tokens = numeric("input_tokens")
    output_tokens = numeric("output_tokens")
    total_tokens = numeric("total_tokens")

    return pd.DataFrame([{
        "successful_calls": len(successful),
        "failed_calls": int((call_metrics["status"] == "error").sum()),
        "usage_reported_calls": len(total_tokens),
        "avg_latency_seconds": float(latency.mean()) if len(latency) else None,
        "p50_latency_seconds": float(latency.quantile(0.50)) if len(latency) else None,
        "p95_latency_seconds": float(latency.quantile(0.95)) if len(latency) else None,
        "avg_input_tokens": float(input_tokens.mean()) if len(input_tokens) else None,
        "avg_output_tokens": float(output_tokens.mean()) if len(output_tokens) else None,
        "avg_total_tokens": float(total_tokens.mean()) if len(total_tokens) else None,
        "total_input_tokens": int(input_tokens.sum()) if len(input_tokens) else None,
        "total_output_tokens": int(output_tokens.sum()) if len(output_tokens) else None,
        "total_tokens": int(total_tokens.sum()) if len(total_tokens) else None,
    }])


def run_predictions(preflight):
    eligible_rows = preflight.loc[preflight["evaluation_eligible"]].copy()
    prediction_rows = []
    call_rows = []

    call_columns = [
        "experiment",
        "theme_id",
        "epic_key",
        "stage_id",
        "candidate_count",
        "status",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "selected_count",
        "error",
    ]

    print(
        f"\\nRunning {EXPERIMENT_NAME}: "
        f"{len(eligible_rows)} preflight-valid Epics only"
    )

    if eligible_rows.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=call_columns),
        )

    gateway = load_gateway()

    for epic_index, row in enumerate(
        eligible_rows.to_dict(orient="records"),
        start=1,
    ):
        theme_id = row["theme_id"]
        epic_key = row["epic_key"]
        stage_ids = json.loads(row["stage_ids"])
        theme = themes[theme_id]
        epic = next(
            item for item in theme["epics"] if item["key"] == epic_key
        )

        stage_predictions = []
        predicted_ids = set()
        reasons = []
        status = "ok"
        error = None

        print(
            f"\\n[LLM {epic_index}/{len(eligible_rows)}] "
            f"{theme_id} | {epic_key}"
        )

        for stage_id in stage_ids:
            started = perf_counter()
            try:
                result = predict_for_stage(
                    gateway,
                    theme,
                    epic,
                    stage_id,
                )
                candidates = result["candidates"]

                if not candidates:
                    print(f"  {stage_id} SKIP | no candidates")
                    continue

                metrics = result["metrics"]
                selected_ids = [
                    selection["capability_id"]
                    for selection in result["selections"]
                ]
                print(
                    f"  {stage_id} OK"
                    f" | candidates={len(candidates)}"
                    f" | latency={metrics['latency_seconds']:.3f}s"
                    f" | input_tokens={metric_text(metrics['input_tokens'])}"
                    f" | output_tokens={metric_text(metrics['output_tokens'])}"
                    f" | total_tokens={metric_text(metrics['total_tokens'])}"
                    f" | selected={selected_ids}"
                )

                call_rows.append({
                    "experiment": EXPERIMENT_NAME,
                    "theme_id": theme_id,
                    "epic_key": epic_key,
                    "stage_id": stage_id,
                    "candidate_count": len(candidates),
                    "status": "ok",
                    "latency_seconds": metrics["latency_seconds"],
                    "input_tokens": metrics["input_tokens"],
                    "output_tokens": metrics["output_tokens"],
                    "total_tokens": metrics["total_tokens"],
                    "selected_count": len(selected_ids),
                    "error": None,
                })
                stage_predictions.append({
                    "stage_id": stage_id,
                    "selections": result["selections"],
                })
                for selection in result["selections"]:
                    predicted_ids.add(selection["capability_id"])
                    reasons.append({"stage_id": stage_id, **selection})
            except Exception as exc:
                latency = perf_counter() - started
                status = "error"
                error = str(exc)
                print(
                    f"  {stage_id} ERROR"
                    f" | latency={latency:.3f}s"
                    f" | {error}"
                )
                call_rows.append({
                    "experiment": EXPERIMENT_NAME,
                    "theme_id": theme_id,
                    "epic_key": epic_key,
                    "stage_id": stage_id,
                    "candidate_count": None,
                    "status": "error",
                    "latency_seconds": latency,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "selected_count": None,
                    "error": error,
                })
                break

        prediction_rows.append({
            "experiment": EXPERIMENT_NAME,
            "theme_id": theme_id,
            "epic_key": epic_key,
            "stage_ids": row["stage_ids"],
            "ground_truth_l3_ids": row["ground_truth_l3_ids"],
            "available_candidate_l3_ids": row["available_candidate_l3_ids"],
            "gt_found_in_candidates": row["gt_found_in_candidates"],
            "gt_missing_from_candidates": row["gt_missing_from_candidates"],
            "predicted_l3_ids": json.dumps(sorted(predicted_ids)),
            "model_reasons": json.dumps(reasons, ensure_ascii=False),
            "stage_predictions": json.dumps(
                stage_predictions,
                ensure_ascii=False,
            ),
            "status": status,
            "error": error,
        })

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(call_rows, columns=call_columns),
    )
'''


BATCH_SOURCE = '''def ground_truth_by_epic():
    gt = read_table(GROUND_TRUTH_PATH).copy()
    gt["l3_capability_id"] = (
        gt["l3_capability_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    gt = gt.loc[gt["l3_capability_id"].ne("")]
    return {
        key: set(group["l3_capability_id"])
        for key, group in gt.groupby("epic_key", sort=False)
    }


def preflight_population():
    truth = ground_truth_by_epic()
    rows = []
    total_epics = sum(len(theme["epics"]) for theme in themes.values())
    check_index = 0

    print("\\n================ PRECHECK ================")
    print(f"Themes selected: {len(themes)}")
    print(f"Total Epics: {total_epics}")

    for theme_id, theme in themes.items():
        for epic in theme["epics"]:
            check_index += 1
            epic_key = epic["key"]
            gt_ids = truth.get(epic_key)
            stage_ids = []
            candidate_ids = set()
            reason = ""
            error = None

            print(
                f"\\n[GT CHECK {check_index}/{total_epics}] "
                f"{theme_id} | {epic_key}"
            )

            if gt_ids is None:
                reason = "missing_ground_truth"
            else:
                try:
                    stage_ids = epic_stage_ids(epic_key)
                except Exception as exc:
                    reason = "error"
                    error = str(exc)

                if not reason and not stage_ids:
                    reason = "no_stage"

                if not reason:
                    try:
                        for stage_id in stage_ids:
                            # Validate stage metadata now so invalid rows never
                            # reach the LLM phase.
                            stage_context(stage_id)
                            candidates = candidate_rows_for_stage(stage_id)
                            candidate_ids.update(
                                candidate["capability_id"]
                                for candidate in candidates
                            )
                    except Exception as exc:
                        reason = "error"
                        error = str(exc)

                if not reason and not candidate_ids:
                    reason = "no_candidates"

            if gt_ids is None:
                gt_found = set()
                gt_missing = set()
            else:
                gt_found = gt_ids & candidate_ids
                gt_missing = gt_ids - candidate_ids

            if not reason and gt_missing:
                reason = "gt_not_fully_retrievable"

            eligible = gt_ids is not None and not reason

            print(f"Stages: {stage_ids}")
            print(f"GT L3s: {sorted(gt_ids) if gt_ids is not None else []}")
            print(f"Candidate L3s: {sorted(candidate_ids)}")
            print(f"GT found in candidates: {sorted(gt_found)}")
            print(f"GT missing from candidates: {sorted(gt_missing)}")

            if eligible:
                print("STATUS: VALID")
            else:
                detail = f" | {error}" if error else ""
                print(f"STATUS: INVALID - {reason}{detail}")

            rows.append({
                "experiment": EXPERIMENT_NAME,
                "theme_id": theme_id,
                "epic_key": epic_key,
                "stage_ids": json.dumps(stage_ids),
                "ground_truth_l3_ids": (
                    json.dumps(sorted(gt_ids))
                    if gt_ids is not None
                    else None
                ),
                "available_candidate_l3_ids": json.dumps(
                    sorted(candidate_ids)
                ),
                "gt_found_in_candidates": json.dumps(sorted(gt_found)),
                "gt_missing_from_candidates": json.dumps(
                    sorted(gt_missing)
                ),
                "evaluation_eligible": eligible,
                "evaluation_exclusion_reason": reason,
                "preflight_error": error,
            })

    preflight = pd.DataFrame(rows)
    valid_count = int(preflight["evaluation_eligible"].sum())
    invalid_count = len(preflight) - valid_count

    print("\\n================ PRECHECK SUMMARY ================")
    print(f"Themes selected: {len(themes)}")
    print(f"Total Epics: {len(preflight)}")
    print(f"VALID Epics: {valid_count}")
    print(f"INVALID Epics: {invalid_count}")
    print("Invalid breakdown:")
    for reason in (
        "missing_ground_truth",
        "no_stage",
        "no_candidates",
        "gt_not_fully_retrievable",
        "error",
    ):
        count = int(
            (preflight["evaluation_exclusion_reason"] == reason).sum()
        )
        print(f"  {reason}: {count}")
    print(f"LLM calls will run ONLY for: {valid_count} Epics")
    print("==================================================")

    return preflight


def evaluate_predictions(prediction_frame):
    out = []
    for row in prediction_frame.to_dict(orient="records"):
        pred = set(json.loads(row["predicted_l3_ids"]))
        gt = set(json.loads(row["ground_truth_l3_ids"]))

        if row["status"] == "error":
            metrics = {
                "exact_match": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "predicted_count": len(pred),
                "truth_count": len(gt),
            }
        else:
            metrics = score_sets(pred, gt)

        row.update(metrics)
        out.append(row)

    return pd.DataFrame(out)


def evaluation_summary(results, preflight):
    scored = results.loc[results["exact_match"].notna()]
    valid_preflight = int(preflight["evaluation_eligible"].sum())

    summary = pd.DataFrame([{
        "scope": "valid_evaluation_population",
        "evaluated_epics": len(scored),
        "exact_match_accuracy": (
            scored["exact_match"].mean() if len(scored) else 0.0
        ),
        "mean_precision": (
            scored["precision"].mean() if len(scored) else 0.0
        ),
        "mean_recall": (
            scored["recall"].mean() if len(scored) else 0.0
        ),
        "mean_f1": scored["f1"].mean() if len(scored) else 0.0,
    }])

    diagnostics = pd.DataFrame([{
        "themes_selected": len(themes),
        "total_epics": len(preflight),
        "preflight_valid_epics": valid_preflight,
        "preflight_invalid_epics": len(preflight) - valid_preflight,
        "missing_ground_truth": int((
            preflight["evaluation_exclusion_reason"]
            == "missing_ground_truth"
        ).sum()),
        "no_stage": int((
            preflight["evaluation_exclusion_reason"] == "no_stage"
        ).sum()),
        "no_candidates": int((
            preflight["evaluation_exclusion_reason"] == "no_candidates"
        ).sum()),
        "gt_not_fully_retrievable": int((
            preflight["evaluation_exclusion_reason"]
            == "gt_not_fully_retrievable"
        ).sum()),
        "preflight_errors": int((
            preflight["evaluation_exclusion_reason"] == "error"
        ).sum()),
        "llm_prediction_errors": int((
            results["status"] == "error"
        ).sum()) if len(results) else 0,
        "scored_epics": len(scored),
    }])

    return summary, diagnostics


# Ground truth is used here only to validate the experiment population.
# It is never included in the LLM prompt.
preflight = preflight_population()
predictions, llm_calls = run_predictions(preflight)
results = evaluate_predictions(predictions)
summary, diagnostics = evaluation_summary(results, preflight)
llm_call_summary = summarize_llm_calls(llm_calls)

print("\\nEvaluation summary")
display(summary)
print("\\nPreflight diagnostics")
display(diagnostics)
print("\\nLLM latency / token summary")
display(llm_call_summary)
print("\\nPer-call LLM metrics")
display(llm_calls.head(50))
if len(results):
    display(results.head(20))

output_path = save_results_excel(
    results,
    EXPERIMENT_NAME,
    "results",
    extra_sheets={
        "evaluation_summary": summary,
        "preflight": preflight,
        "diagnostics": diagnostics,
        "llm_calls": llm_calls,
        "llm_call_summary": llm_call_summary,
    },
)
print(f"Saved {output_path}")
'''


BATCH_MARKDOWN = '''## Batch preflight, execution, and evaluation

Ground truth is loaded **before any LLM call** only to validate the experiment population. An Epic is sent to the LLM only when GT exists, a Stage exists, candidates exist, and every GT L3 is present in the union of Stage candidates. GT is never included in the model prompt.
'''


def main():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))

        cell_after_heading(notebook, "## Prediction")["source"] = PREDICTION_SOURCE

        batch_md = markdown_with_heading(
            notebook,
            "## Batch execution and evaluation",
        )
        batch_md["source"] = BATCH_MARKDOWN
        next_index = notebook["cells"].index(batch_md) + 1
        notebook["cells"][next_index]["source"] = BATCH_SOURCE

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile(
                    source(cell),
                    f"{path.name}:cell-{index}",
                    "exec",
                )

        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("Applied preflight validation to all five notebooks.")


if __name__ == "__main__":
    main()
