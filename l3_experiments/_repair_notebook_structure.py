from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK_NAMES = [
    "01_theme_stage.ipynb",
    "02_full_context.ipynb",
    "03_no_theme_description.ipynb",
    "04_no_theme.ipynb",
    "05_full_with_hierarchy.ipynb",
]

EXPERIMENT_NAMES = {
    "01_theme_stage.ipynb": "E1_THEME_STAGE",
    "02_full_context.ipynb": "E2_FULL_CONTEXT",
    "03_no_theme_description.ipynb": "E3_NO_THEME_DESCRIPTION",
    "04_no_theme.ipynb": "E4_NO_THEME",
    "05_full_with_hierarchy.ipynb": "E5_FULL_WITH_HIERARCHY",
}


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def config_source(experiment_name: str) -> str:
    return f'''from pathlib import Path
from time import perf_counter
import ast
import io
import json
import os
import re
import tokenize

import httpx
import pandas as pd
from IPython.display import display

from common import (
    call_llm_with_metrics,
    load_gateway,
    parse_json_response,
    save_results_excel,
    score_sets,
    validate_l3_response,
)

NOTEBOOK_DIR = Path.cwd()
WORKSPACE_DIR = (
    NOTEBOOK_DIR.parent
    if (NOTEBOOK_DIR.parent / "epic_gen.csv").exists()
    else NOTEBOOK_DIR
)
DATA_DIR = Path(os.getenv("L3_EXPERIMENT_DATA_DIR", WORKSPACE_DIR))

THEME_PATH = Path(os.getenv("L3_THEME_PATH", DATA_DIR / "epic_gen.csv"))
STAGE_PATH = Path(os.getenv("L3_STAGE_PATH", DATA_DIR / "VSSrv.csv"))
STAGE_CAPABILITY_MAP_PATH = Path(
    os.getenv(
        "L3_STAGE_CAPABILITY_MAP_PATH",
        DATA_DIR / "VSSCaprv (1).csv",
    )
)
GROUND_TRUTH_PATH = Path(
    os.getenv(
        "L3_GROUND_TRUTH_PATH",
        NOTEBOOK_DIR / "epic_l3_ground_truth_all_themes.xlsx",
    )
)

# Run every Theme in epic_gen.csv. Ground truth is not consulted here.
THEME_IDS = (
    pd.read_csv(
        THEME_PATH,
        usecols=["key"],
        encoding="cp1252",
        encoding_errors="replace",
        dtype=str,
    )["key"]
    .dropna()
    .str.strip()
    .loc[lambda values: values.ne("")]
    .drop_duplicates()
    .tolist()
)

VALUE_STREAM_STAGE_FIELD_ID = "customfield_18700"

# Optional single-example inspection. Leave None for batch execution only.
INSPECTION_THEME_ID = None
INSPECTION_EPIC_KEY = None

EXPERIMENT_NAME = "{experiment_name}"
'''


BASE_CANDIDATE_SOURCE = '''def candidate_rows_for_stage(stage_id):
    rows = stage_capability_map.loc[
        stage_capability_map["Value Stream Stage ID"].astype(str).str.strip()
        == stage_id
    ].copy()
    rows = (
        rows.drop_duplicates(subset=["Capability ID"], keep="first")
        .sort_values(["Capability Name", "Capability ID"], kind="stable")
    )

    return [
        {
            "capability_id": clean_text(row["Capability ID"]),
            "capability_name": clean_text(row["Capability Name"]),
            "capability_description": clean_text(row["Capability Description"]),
            "capability_tier": clean_text(row["Capability Tier"]),
        }
        for _, row in rows.iterrows()
    ]
'''


HIERARCHY_CANDIDATE_SOURCE = '''def candidate_rows_for_stage(stage_id):
    rows = stage_capability_map.loc[
        stage_capability_map["Value Stream Stage ID"].astype(str).str.strip()
        == stage_id
    ].copy()
    rows = (
        rows.drop_duplicates(subset=["Capability ID"], keep="first")
        .sort_values(["Capability Name", "Capability ID"], kind="stable")
    )

    return [
        {
            "capability_id": clean_text(row["Capability ID"]),
            "capability_name": clean_text(row["Capability Name"]),
            "capability_description": clean_text(row["Capability Description"]),
            "capability_tier": clean_text(row["Capability Tier"]),
            "level_1_name": clean_text(row["Level 1 Name"]),
            "level_2_name": clean_text(row["Level 2 Name"]),
        }
        for _, row in rows.iterrows()
    ]
'''


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


def run_predictions():
    gateway = load_gateway()
    prediction_rows = []
    call_rows = []
    total_epics = sum(len(theme["epics"]) for theme in themes.values())
    epic_index = 0

    print(f"Running {EXPERIMENT_NAME}: {len(themes)} themes / {total_epics} epics")

    for theme_id, theme in themes.items():
        for epic in theme["epics"]:
            epic_index += 1
            epic_key = epic["key"]
            stage_ids = []
            stage_predictions = []
            available_ids = set()
            predicted_ids = set()
            reasons = []
            status = "ok"
            error = None

            print(f"\\n[{epic_index}/{total_epics}] {theme_id} | {epic_key}")

            try:
                stage_ids = epic_stage_ids(epic_key)
            except Exception as exc:
                status = "error"
                error = str(exc)
                print(f"  JIRA ERROR | {error}")

            if status == "ok" and not stage_ids:
                status = "no_stage"
                print("  SKIP | no Value Stream Stage")

            if status == "ok":
                for stage_id in stage_ids:
                    started = perf_counter()
                    try:
                        result = predict_for_stage(gateway, theme, epic, stage_id)
                        candidates = result["candidates"]
                        available_ids.update(
                            candidate["capability_id"] for candidate in candidates
                        )

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
                        print(f"  {stage_id} ERROR | latency={latency:.3f}s | {error}")
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

            if status == "ok" and stage_ids and not available_ids:
                status = "no_candidates"

            prediction_rows.append({
                "experiment": EXPERIMENT_NAME,
                "theme_id": theme_id,
                "epic_key": epic_key,
                "stage_ids": json.dumps(stage_ids),
                "available_candidate_l3_ids": json.dumps(sorted(available_ids)),
                "predicted_l3_ids": json.dumps(sorted(predicted_ids)),
                "model_reasons": json.dumps(reasons, ensure_ascii=False),
                "stage_predictions": json.dumps(stage_predictions, ensure_ascii=False),
                "status": status,
                "error": error,
            })

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
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(call_rows, columns=call_columns),
    )
'''


INSPECTION_SOURCE = '''if INSPECTION_THEME_ID and INSPECTION_EPIC_KEY:
    theme = themes[INSPECTION_THEME_ID]
    epic = next(
        item
        for item in theme["epics"]
        if item["key"] == INSPECTION_EPIC_KEY
    )
    stage_id = epic_stage_ids(epic["key"])[0]
    stage = stage_context(stage_id)
    candidates = candidate_rows_for_stage(stage_id)
    user_prompt = build_user_prompt(theme, epic, stage, candidates)

    print("SYSTEM PROMPT")
    print(SYSTEM_PROMPT)
    print("\\nUSER PROMPT")
    print(user_prompt)
    print("\\nCANDIDATES")
    display(pd.DataFrame(candidates))

    if candidates:
        result = predict_for_stage(
            load_gateway(),
            theme,
            epic,
            stage_id,
        )
        print("\\nMODEL RESPONSE")
        print(result["raw_response"])
        print("\\nCALL METRICS")
        display(pd.DataFrame([result["metrics"]]))
    else:
        print("\\nNo candidates for this Stage; LLM call skipped.")
else:
    print(
        "Set INSPECTION_THEME_ID and INSPECTION_EPIC_KEY "
        "to inspect one example."
    )
'''


def code_after_heading(notebook, heading: str):
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown" and heading in source(cell):
            if index + 1 >= len(notebook["cells"]):
                raise AssertionError(f"Missing cell after {heading}")
            next_cell = notebook["cells"][index + 1]
            if next_cell["cell_type"] != "code":
                raise AssertionError(f"Expected code cell after {heading}")
            return next_cell
    raise AssertionError(f"Heading not found: {heading}")


def prompt_value(notebook):
    prompt_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "SYSTEM_PROMPT" in source(cell)
        and "def build_user_prompt" in source(cell)
    )
    tree = ast.parse(source(prompt_cell))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("SYSTEM_PROMPT assignment missing")


def clean_retrieval_comments(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = source(cell)
        if "def load_themes" not in text:
            continue
        text = text.replace(
            "# The stage-capability export already contains description/tier, so keep only the join key\n"
            "# on the right side. This avoids pandas creating Capability Description_x/_y columns.\n",
            "",
        )
        cell["source"] = text


def validate_notebook(path: Path, notebook):
    config = code_after_heading(notebook, "## Configuration and imports")
    config_text = source(config)
    assert config_text.startswith("from pathlib import Path\n")
    assert "from time import perf_counter" in config_text
    assert "call_llm_with_metrics" in config_text
    assert "EXPERIMENT_NAME =" in config_text

    candidate = code_after_heading(notebook, "## Candidate construction")
    candidate_text = source(candidate)
    assert "drop_duplicates(subset=[\"Capability ID\"]" in candidate_text
    assert "sort_values" in candidate_text

    inspection = code_after_heading(notebook, "## Single-example inspection")
    inspection_text = source(inspection)
    assert "CALL METRICS" in inspection_text

    all_code = "\n".join(
        source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert all_code.count("INSPECTION_THEME_ID = None") == 1
    assert "CANDIDATE_FIELDS" not in all_code
    assert "CAPABILITY_MASTER_PATH" not in all_code
    assert "capability_master" not in all_code
    assert "rows.merge(" not in all_code
    assert "call_llm_with_metrics" in all_code
    assert "avg_latency_seconds" in all_code
    assert "p50_latency_seconds" in all_code
    assert "p95_latency_seconds" in all_code
    assert "avg_input_tokens" in all_code
    assert "avg_output_tokens" in all_code
    assert "llm_calls" in all_code
    assert "llm_call_summary" in all_code

    llm_view = next(
        source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
        and "## What the LLM sees" in source(cell)
    )
    assert "value_stream_stage" in llm_view
    assert "candidate_l3_capabilities[]" in llm_view

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(source(cell), f"{path.name}:cell-{index}", "exec")


def main():
    prompts = []

    for name in NOTEBOOK_NAMES:
        path = ROOT / name
        notebook = json.loads(path.read_text(encoding="utf-8"))

        code_after_heading(notebook, "## Configuration and imports")["source"] = (
            config_source(EXPERIMENT_NAMES[name])
        )
        code_after_heading(notebook, "## Candidate construction")["source"] = (
            HIERARCHY_CANDIDATE_SOURCE
            if name == "05_full_with_hierarchy.ipynb"
            else BASE_CANDIDATE_SOURCE
        )
        code_after_heading(notebook, "## Prediction")["source"] = PREDICTION_SOURCE
        code_after_heading(notebook, "## Single-example inspection")["source"] = (
            INSPECTION_SOURCE
        )
        clean_retrieval_comments(notebook)

        validate_notebook(path, notebook)
        prompts.append(prompt_value(notebook))

        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    assert len(set(prompts)) == 1, "SYSTEM_PROMPT differs across experiments"
    print("Structural verification passed for all five L3 notebooks.")


if __name__ == "__main__":
    main()
