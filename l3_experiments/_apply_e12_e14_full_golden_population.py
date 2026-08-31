import json
from pathlib import Path

GT_PATH = Path("l3_experiments/00_fetch_full_golden_ground_truth.ipynb")
EXPERIMENT_PATHS = [
    Path("l3_experiments/12_business_needs_stage_custom_prompt.ipynb"),
    Path("l3_experiments/13_theme_description_stage_custom_prompt.ipynb"),
    Path("l3_experiments/14_theme_needs_description_stage_custom_prompt.ipynb"),
]

EXPERIMENT_NAMES = {
    "12_business_needs_stage_custom_prompt.ipynb": "E12_BUSINESS_NEEDS_STAGE_CUSTOM_PROMPT",
    "13_theme_description_stage_custom_prompt.ipynb": "E13_THEME_DESCRIPTION_STAGE_CUSTOM_PROMPT",
    "14_theme_needs_description_stage_custom_prompt.ipynb": "E14_THEME_NEEDS_DESCRIPTION_STAGE_CUSTOM_PROMPT",
}


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook: dict) -> None:
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def next_code_cell(notebook: dict, heading: str) -> dict:
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown" and heading in cell.get("source", ""):
            for following in cells[index + 1 :]:
                if following.get("cell_type") == "code":
                    return following
    raise ValueError(f"Could not find code cell after heading: {heading}")


def markdown_cell(notebook: dict, heading: str) -> dict:
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "markdown" and heading in cell.get("source", ""):
            return cell
    raise ValueError(f"Could not find markdown cell: {heading}")


def compile_notebook(path: Path, notebook: dict) -> None:
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") == "code":
            compile(cell.get("source", ""), f"{path.name}:cell_{index}", "exec")


def update_gt_notebook() -> None:
    notebook = load_notebook(GT_PATH)
    marker = "## Persist valid evaluation population"

    if not any(
        cell.get("cell_type") == "markdown" and marker in cell.get("source", "")
        for cell in notebook["cells"]
    ):
        notebook["cells"].append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": (
                    "## Persist valid evaluation population\n\n"
                    "Save one row per valid Theme/Epic pair into the same GT workbook so "
                    "E12-E14 can use an identical prevalidated population without repeating "
                    "Jira Stage lookup or GT/candidate coverage checks."
                ),
            }
        )

        notebook["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": '''if "key" not in df.columns:
    raise KeyError("full_golden.parquet does not contain a 'key' Theme column.")

validity_by_epic = {
    row["epic_key"]: row
    for row in validity.to_dict(orient="records")
}

population_rows = []
for _, theme_row in df.iterrows():
    theme_key = str(theme_row.get("key") or "").strip()
    if not theme_key:
        continue

    for epic_key in parse_epic_keys(theme_row.get("epic_keys")):
        coverage = validity_by_epic.get(epic_key)
        if not coverage or not bool(coverage["valid"]):
            continue

        population_rows.append(
            {
                "theme_key": theme_key,
                "epic_key": epic_key,
                "stage_ids": json.dumps(coverage["stage_ids"]),
                "gt_l3_ids": json.dumps(coverage["gt_l3_ids"]),
                "candidate_l3_ids": json.dumps(coverage["candidate_l3_ids"]),
                "missing_gt_l3_ids": json.dumps(coverage["missing_gt_l3_ids"]),
                "valid": True,
                "invalid_reason": "",
            }
        )

evaluation_population = (
    pd.DataFrame(population_rows)
    .drop_duplicates(subset=["theme_key", "epic_key"], keep="first")
    .sort_values(["theme_key", "epic_key"], kind="stable")
    .reset_index(drop=True)
)

if evaluation_population["epic_key"].duplicated().any():
    duplicates = sorted(
        evaluation_population.loc[
            evaluation_population["epic_key"].duplicated(keep=False),
            "epic_key",
        ].unique()
    )
    raise ValueError(
        "A valid Epic is linked to multiple Themes; resolve before sampling: "
        f"{duplicates}"
    )

with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl",
    mode="a",
    if_sheet_exists="replace",
) as writer:
    evaluation_population.to_excel(
        writer,
        sheet_name="evaluation_population",
        index=False,
    )

print(f"Saved valid evaluation population to: {OUTPUT_PATH}")
print(f"Valid Theme/Epic rows: {len(evaluation_population)}")
print(f"Valid unique Epics:     {evaluation_population['epic_key'].nunique()}")
print(f"Themes represented:     {evaluation_population['theme_key'].nunique()}")
display(evaluation_population.head(50))
''',
            }
        )

    compile_notebook(GT_PATH, notebook)
    save_notebook(GT_PATH, notebook)


def update_experiment(path: Path) -> None:
    notebook = load_notebook(path)
    experiment_name = EXPERIMENT_NAMES[path.name]

    config = next_code_cell(notebook, "## Configuration and imports")
    config["source"] = f'''from pathlib import Path
from time import perf_counter
import ast
import json
import os

import pandas as pd
from IPython.display import display

from common import (
    call_llm_with_metrics,
    load_gateway,
    parse_json_response,
    save_results_excel,
    score_sets,
)

NOTEBOOK_DIR = Path.cwd()
WORKSPACE_DIR = (
    NOTEBOOK_DIR.parent
    if (NOTEBOOK_DIR.parent / "full_golden.parquet").exists()
    else NOTEBOOK_DIR
)
DATA_DIR = Path(os.getenv("L3_EXPERIMENT_DATA_DIR", WORKSPACE_DIR))

PARQUET_PATH = Path(
    os.getenv("L3_FULL_GOLDEN_PATH", DATA_DIR / "full_golden.parquet")
)
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
        DATA_DIR / "results" / "epic_l3_ground_truth_full_golden.xlsx",
    )
)

SAMPLE_SIZE = 50
SAMPLE_SEED = 42

# Optional single-example inspection. Leave None for batch execution only.
INSPECTION_THEME_ID = None
INSPECTION_EPIC_KEY = None

EXPERIMENT_NAME = "{experiment_name}"
'''

    retrieval = next_code_cell(notebook, "## Retrieval")
    retrieval["source"] = '''def clean_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def parse_list_value(value) -> list[str]:
    if value is None:
        return []

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return [clean_text(item) for item in value if clean_text(item)]

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]

    if isinstance(parsed, (list, tuple, set)):
        return [clean_text(item) for item in parsed if clean_text(item)]
    return [clean_text(parsed)] if clean_text(parsed) else []


def read_table(path, *, sheet_name=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            dtype=str,
            encoding="cp1252",
            encoding_errors="replace",
        )
    return pd.read_excel(path, sheet_name=sheet_name, dtype=str)


def load_evaluation_population():
    population = read_table(
        GROUND_TRUTH_PATH,
        sheet_name="evaluation_population",
    )
    required = {
        "theme_key",
        "epic_key",
        "stage_ids",
        "gt_l3_ids",
        "candidate_l3_ids",
    }
    missing = required.difference(population.columns)
    if missing:
        raise KeyError(
            f"evaluation_population is missing columns: {sorted(missing)}"
        )

    population = (
        population
        .drop_duplicates(subset=["theme_key", "epic_key"], keep="first")
        .sort_values(["theme_key", "epic_key"], kind="stable")
        .reset_index(drop=True)
    )

    if population["epic_key"].duplicated().any():
        raise ValueError("evaluation_population contains duplicate Epic keys.")

    if len(population) < SAMPLE_SIZE:
        raise ValueError(
            f"Need {SAMPLE_SIZE} valid Epics, found only {len(population)}."
        )

    sample = population.sample(
        n=SAMPLE_SIZE,
        random_state=SAMPLE_SEED,
        replace=False,
    )
    return sample.sort_values(
        ["theme_key", "epic_key"],
        kind="stable",
    ).reset_index(drop=True)


evaluation_population = load_evaluation_population()
selected_pairs = set(
    zip(
        evaluation_population["theme_key"],
        evaluation_population["epic_key"],
    )
)
selected_theme_ids = set(evaluation_population["theme_key"])


def load_themes():
    frame = read_table(PARQUET_PATH)
    required = {"key", "description", "businessNeeds", "epic_keys"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"full_golden.parquet is missing columns: {sorted(missing)}"
        )

    themes = {}
    found_pairs = set()

    for _, row in frame.iterrows():
        theme_id = clean_text(row.get("key"))
        if theme_id not in selected_theme_ids:
            continue

        selected_epics = []
        for epic_key in parse_list_value(row.get("epic_keys")):
            if (theme_id, epic_key) in selected_pairs:
                selected_epics.append({"key": epic_key})
                found_pairs.add((theme_id, epic_key))

        if not selected_epics:
            continue

        themes[theme_id] = {
            "theme_description": clean_text(row.get("description")),
            "theme_business_needs": clean_text(row.get("businessNeeds")),
            "epics": selected_epics,
        }

    missing_pairs = selected_pairs - found_pairs
    if missing_pairs:
        raise ValueError(
            "Selected Theme/Epic pairs were not found in full_golden.parquet: "
            f"{sorted(missing_pairs)}"
        )

    return themes


themes = load_themes()
stage_frame = read_table(STAGE_PATH)
stage_capability_map = read_table(STAGE_CAPABILITY_MAP_PATH)


def stage_context(stage_id):
    match = stage_frame.loc[
        stage_frame["Value Stream Stage ID"].astype(str).str.strip() == stage_id
    ]
    if match.empty:
        raise KeyError(f"No stage metadata for {stage_id}")
    row = match.iloc[0]
    return {
        "stage_id": stage_id,
        "stage_name": clean_text(row["Value Stream Stage Name"]),
        "stage_description": clean_text(row["Value Stream Stage Description"]),
        "entrance_criteria": clean_text(row["Value Stream Stage Entrance Criteria"]),
        "exit_criteria": clean_text(row["Value Stream Stage Exit Criteria"]),
    }


print(
    f"Selected {len(evaluation_population)} valid Epics "
    f"with seed={SAMPLE_SEED} across {len(themes)} Themes."
)
display(evaluation_population.head(50))
'''

    inspection = next_code_cell(notebook, "## Single-example inspection")
    inspection["source"] = '''if INSPECTION_THEME_ID and INSPECTION_EPIC_KEY:
    match = evaluation_population.loc[
        (evaluation_population["theme_key"] == INSPECTION_THEME_ID)
        & (evaluation_population["epic_key"] == INSPECTION_EPIC_KEY)
    ]
    if match.empty:
        raise ValueError(
            "Inspection Theme/Epic is not in the selected 50-Epic population."
        )

    theme = themes[INSPECTION_THEME_ID]
    epic = next(
        item
        for item in theme["epics"]
        if item["key"] == INSPECTION_EPIC_KEY
    )
    stage_id = json.loads(match.iloc[0]["stage_ids"])[0]
    stage = stage_context(stage_id)
    candidates = candidate_rows_for_stage(stage_id)
    user_prompt = build_user_prompt(theme, epic, stage, candidates)

    print("SYSTEM PROMPT")
    print(SYSTEM_PROMPT)
    print()
    print("USER PROMPT")
    print(user_prompt)
    print()
    print("CANDIDATES")
    display(pd.DataFrame(candidates))

    if candidates:
        result = predict_for_stage(
            load_gateway(),
            theme,
            epic,
            stage_id,
        )
        print()
        print("MODEL RESPONSE")
        print(result["raw_response"])
        print()
        print("CALL METRICS")
        display(pd.DataFrame([result["metrics"]]))
    else:
        print()
        print("No candidates for this Stage; LLM call skipped.")
else:
    print(
        "Set INSPECTION_THEME_ID and INSPECTION_EPIC_KEY "
        "to inspect one of the selected 50 Epics."
    )
'''

    batch_md = markdown_cell(notebook, "## Batch preflight, execution, and evaluation")
    batch_md["source"] = (
        "## Batch population, execution, and evaluation\n\n"
        "The GT workbook already contains the prevalidated valid population. "
        "This experiment takes the same fixed random sample of 50 valid Epics "
        "(`SAMPLE_SEED = 42`) and never sends GT to the LLM."
    )

    batch = next_code_cell(notebook, "## Batch population, execution, and evaluation")
    batch["source"] = '''def preflight_population():
    rows = []

    for record in evaluation_population.to_dict(orient="records"):
        theme_id = clean_text(record["theme_key"])
        epic_key = clean_text(record["epic_key"])
        stage_ids = json.loads(record["stage_ids"])
        gt_ids = set(json.loads(record["gt_l3_ids"]))
        candidate_ids = set(json.loads(record["candidate_l3_ids"]))

        rows.append(
            {
                "experiment": EXPERIMENT_NAME,
                "theme_id": theme_id,
                "epic_key": epic_key,
                "stage_ids": json.dumps(stage_ids),
                "ground_truth_l3_ids": json.dumps(sorted(gt_ids)),
                "available_candidate_l3_ids": json.dumps(sorted(candidate_ids)),
                "gt_found_in_candidates": json.dumps(sorted(gt_ids)),
                "gt_missing_from_candidates": json.dumps([]),
                "evaluation_eligible": True,
                "evaluation_exclusion_reason": "",
                "preflight_error": None,
            }
        )

    preflight = pd.DataFrame(rows)
    print()
    print("================ POPULATION SUMMARY ================")
    print(f"Sample seed: {SAMPLE_SEED}")
    print(f"Epics selected: {len(preflight)}")
    print(f"Themes represented: {preflight['theme_id'].nunique()}")
    print("All selected Epics were prevalidated in the GT workbook.")
    print("====================================================")
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

    summary = pd.DataFrame([{
        "scope": "fixed_50_valid_epics_seed_42",
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
        "sample_seed": SAMPLE_SEED,
        "sample_size": SAMPLE_SIZE,
        "themes_selected": preflight["theme_id"].nunique(),
        "total_epics": len(preflight),
        "preflight_valid_epics": len(preflight),
        "preflight_invalid_epics": 0,
        "llm_prediction_errors": int((
            results["status"] == "error"
        ).sum()) if len(results) else 0,
        "scored_epics": len(scored),
    }])

    return summary, diagnostics


preflight = preflight_population()
predictions, llm_calls = run_predictions(preflight)
results = evaluate_predictions(predictions)
summary, diagnostics = evaluation_summary(results, preflight)
llm_call_summary = summarize_llm_calls(llm_calls)

print()
print("Evaluation summary")
display(summary)
print()
print("Population diagnostics")
display(diagnostics)
print()
print("LLM latency / token summary")
display(llm_call_summary)
print()
print("Per-call LLM metrics")
display(llm_calls.head(50))
if len(results):
    display(results.head(50))

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
        "evaluation_population": evaluation_population,
    },
)
print(f"Saved {output_path}")
'''

    compile_notebook(path, notebook)
    save_notebook(path, notebook)


update_gt_notebook()
for experiment_path in EXPERIMENT_PATHS:
    update_experiment(experiment_path)

print("Updated GT population persistence and E12-E14 data sources")
