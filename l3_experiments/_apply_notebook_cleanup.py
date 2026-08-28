from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    ROOT / "01_theme_stage.ipynb",
    ROOT / "02_full_context.ipynb",
    ROOT / "03_no_theme_description.ipynb",
    ROOT / "04_no_theme.ipynb",
    ROOT / "05_full_with_hierarchy.ipynb",
]

PROMPT_STRUCTURES = {
    "01_theme_stage.ipynb": """## What the LLM sees

The **user prompt** is structured as:

```text
task
theme
  ├─ business_needs
  └─ description
value_stream_stage
  ├─ stage_id
  ├─ stage_name
  ├─ stage_description
  ├─ entrance_criteria
  └─ exit_criteria
candidate_l3_capabilities[]
  ├─ capability_id
  ├─ capability_name
  ├─ capability_description
  └─ capability_tier
selection_instruction
```

**Not sent to the LLM:** Epic context, L1/L2 hierarchy, ground truth.
""",
    "02_full_context.ipynb": """## What the LLM sees

The **user prompt** is structured as:

```text
task
theme
  ├─ business_needs
  └─ description
epic
  ├─ description
  └─ success_criteria
value_stream_stage
  ├─ stage_id
  ├─ stage_name
  ├─ stage_description
  ├─ entrance_criteria
  └─ exit_criteria
candidate_l3_capabilities[]
  ├─ capability_id
  ├─ capability_name
  ├─ capability_description
  └─ capability_tier
selection_instruction
```

**Not sent to the LLM:** L1/L2 hierarchy, ground truth.
""",
    "03_no_theme_description.ipynb": """## What the LLM sees

The **user prompt** is structured as:

```text
task
theme
  └─ business_needs
epic
  ├─ description
  └─ success_criteria
value_stream_stage
  ├─ stage_id
  ├─ stage_name
  ├─ stage_description
  ├─ entrance_criteria
  └─ exit_criteria
candidate_l3_capabilities[]
  ├─ capability_id
  ├─ capability_name
  ├─ capability_description
  └─ capability_tier
selection_instruction
```

**Not sent to the LLM:** Theme description, L1/L2 hierarchy, ground truth.
""",
    "04_no_theme.ipynb": """## What the LLM sees

The **user prompt** is structured as:

```text
task
epic
  ├─ description
  └─ success_criteria
value_stream_stage
  ├─ stage_id
  ├─ stage_name
  ├─ stage_description
  ├─ entrance_criteria
  └─ exit_criteria
candidate_l3_capabilities[]
  ├─ capability_id
  ├─ capability_name
  ├─ capability_description
  └─ capability_tier
selection_instruction
```

**Not sent to the LLM:** Theme context, L1/L2 hierarchy, ground truth.
""",
    "05_full_with_hierarchy.ipynb": """## What the LLM sees

The **user prompt** is structured as:

```text
task
theme
  ├─ business_needs
  └─ description
epic
  ├─ description
  └─ success_criteria
value_stream_stage
  ├─ stage_id
  ├─ stage_name
  ├─ stage_description
  ├─ entrance_criteria
  └─ exit_criteria
candidate_l3_capabilities[]
  ├─ capability_id
  ├─ capability_name
  ├─ capability_description
  ├─ capability_tier
  ├─ level_1_name
  └─ level_2_name
selection_instruction
```

**Not sent to the LLM:** ground truth.
""",
}

LOAD_THEMES = {
    "01_theme_stage.ipynb": '''def load_themes():
    frame = read_table(THEME_PATH)
    frame = frame.loc[frame["key"].isin(THEME_IDS)]
    themes = {}

    for _, row in frame.iterrows():
        epic_keys = parse_exported_list(row.get("epic_keys"))
        themes[clean_text(row["key"])] = {
            "theme_description": clean_text(row.get("description")),
            "theme_business_needs": clean_text(row.get("businessNeeds")),
            "epics": [{"key": epic_key} for epic_key in epic_keys],
        }

    return themes
''',
    "02_full_context.ipynb": None,
    "03_no_theme_description.ipynb": None,
    "04_no_theme.ipynb": None,
    "05_full_with_hierarchy.ipynb": None,
}

LOAD_THEMES_FULL = '''def load_themes():
    frame = read_table(THEME_PATH)
    frame = frame.loc[frame["key"].isin(THEME_IDS)]
    themes = {}

    for _, row in frame.iterrows():
        epic_keys = parse_exported_list(row.get("epic_keys"))
        epic_descriptions = parse_exported_list(row.get("epic_description"))
        epic_success_criteria = parse_exported_list(row.get("epic_successCriteria"))
        epics = [
            {
                "key": epic_key,
                "description": epic_descriptions[index] if index < len(epic_descriptions) else "",
                "success_criteria": epic_success_criteria[index] if index < len(epic_success_criteria) else "",
            }
            for index, epic_key in enumerate(epic_keys)
        ]
        themes[clean_text(row["key"])] = {
            "theme_description": clean_text(row.get("description")),
            "theme_business_needs": clean_text(row.get("businessNeeds")),
            "epics": epics,
        }

    return themes
'''

LOAD_THEMES["02_full_context.ipynb"] = LOAD_THEMES_FULL
LOAD_THEMES["05_full_with_hierarchy.ipynb"] = LOAD_THEMES_FULL
LOAD_THEMES["03_no_theme_description.ipynb"] = LOAD_THEMES_FULL.replace(
    '            "theme_description": clean_text(row.get("description")),\n',
    '',
)
LOAD_THEMES["04_no_theme.ipynb"] = LOAD_THEMES_FULL.replace(
    '            "theme_description": clean_text(row.get("description")),\n',
    '',
).replace(
    '            "theme_business_needs": clean_text(row.get("businessNeeds")),\n',
    '',
)

PREDICTION_CELL = '''def predict_for_stage(gateway, theme, epic, stage_id):
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
                    candidates = candidate_rows_for_stage(stage_id)
                    available_ids.update(
                        candidate["capability_id"] for candidate in candidates
                    )

                    if not candidates:
                        print(f"  {stage_id} SKIP | no candidates")
                        continue

                    started = perf_counter()
                    try:
                        result = predict_for_stage(gateway, theme, epic, stage_id)
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
                            "candidate_count": len(candidates),
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
        "experiment", "theme_id", "epic_key", "stage_id", "candidate_count",
        "status", "latency_seconds", "input_tokens", "output_tokens",
        "total_tokens", "selected_count", "error",
    ]
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(call_rows, columns=call_columns),
    )
'''

INSPECTION_CELL = '''if INSPECTION_THEME_ID and INSPECTION_EPIC_KEY:
    theme = themes[INSPECTION_THEME_ID]
    epic = next(
        item for item in theme["epics"]
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
        result = predict_for_stage(load_gateway(), theme, epic, stage_id)
        print("\\nMODEL RESPONSE")
        print(result["raw_response"])
        print("\\nCALL METRICS")
        display(pd.DataFrame([result["metrics"]]))
    else:
        print("\\nNo candidates for this Stage; LLM call skipped.")
else:
    print("Set INSPECTION_THEME_ID and INSPECTION_EPIC_KEY to inspect one example.")
'''

FINAL_TAIL = '''predictions, llm_calls = run_predictions()
results = evaluate_predictions(predictions)
summary, diagnostics = evaluation_summary(results)
llm_call_summary = summarize_llm_calls(llm_calls)

print("\\nEvaluation summary")
display(summary)
print("\\nDataset / retrieval diagnostics")
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


def set_source(cell, value):
    cell["source"] = value


def replace_load_themes(text, replacement):
    start = text.index("def load_themes")
    end = text.index("\ndef jira_headers", start)
    return text[:start] + replacement + "\n" + text[end + 1:]


def strip_capability_master(text):
    text = re.sub(
        r'\n?capability_master\s*=\s*\(.*?\n\)\n',
        '\n',
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'\n?capability_master\s*=\s*read_table\(CAPABILITY_MASTER_PATH\).*?\n',
        '\n',
        text,
    )
    return text


def patch_candidate_cell(text):
    text = re.sub(r'^CANDIDATE_FIELDS\s*=.*?\n\n', '', text, flags=re.MULTILINE)
    text = re.sub(
        r'\s*rows\s*=\s*rows\.merge\(capability_master,.*?\)\n',
        '\n',
        text,
        flags=re.DOTALL,
    )
    return text


def patch_config(text):
    text = re.sub(
        r'^CAPABILITY_MASTER_PATH\s*=.*?(?:\n\s*os\.getenv.*?\n\))?\n',
        '',
        text,
        flags=re.MULTILINE,
    )
    if "from time import perf_counter" not in text:
        text = text.replace("from pathlib import Path\n", "from pathlib import Path\nfrom time import perf_counter\n", 1)
    text = text.replace("    call_llm,\n", "    call_llm_with_metrics,\n")
    text = text.replace("from common import call_llm, ", "from common import call_llm_with_metrics, ")
    return text


def patch_final_cell(text):
    marker = "predictions = run_predictions()"
    marker_compact = "predictions=run_predictions()"
    if marker in text:
        start = text.index(marker)
    elif marker_compact in text:
        start = text.index(marker_compact)
    else:
        raise AssertionError("Could not find final execution tail")
    return text[:start] + FINAL_TAIL


def patch_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))

    # Add / replace the top-level prompt structure.
    existing = next(
        (i for i, cell in enumerate(notebook["cells"])
         if cell["cell_type"] == "markdown" and "## What the LLM sees" in source(cell)),
        None,
    )
    prompt_cell = {"cell_type": "markdown", "metadata": {}, "source": PROMPT_STRUCTURES[path.name]}
    if existing is None:
        notebook["cells"].insert(1, prompt_cell)
    else:
        notebook["cells"][existing] = prompt_cell

    config = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "EXPERIMENT_NAME" in source(cell) and "THEME_PATH" in source(cell)
    )
    set_source(config, patch_config(source(config)))

    retrieval = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def load_themes" in source(cell) and "def epic_stage_ids" in source(cell)
    )
    retrieval_text = replace_load_themes(source(retrieval), LOAD_THEMES[path.name])
    retrieval_text = strip_capability_master(retrieval_text)
    set_source(retrieval, retrieval_text)

    candidate = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def candidate_rows_for_stage" in source(cell)
    )
    set_source(candidate, patch_candidate_cell(source(candidate)))

    prediction = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def predict_for_stage" in source(cell) and "def run_predictions" in source(cell)
    )
    set_source(prediction, PREDICTION_CELL)

    inspection = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "INSPECTION_THEME_ID" in source(cell) and "MODEL RESPONSE" not in source(cell)
    )
    set_source(inspection, INSPECTION_CELL)

    final = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def ground_truth_by_epic" in source(cell)
    )
    set_source(final, patch_final_cell(source(final)))

    # The production prompt must stay byte-identical and readable as a multiline string.
    prompt_source = next(
        source(cell) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "SYSTEM_PROMPT" in source(cell)
    )
    assert 'SYSTEM_PROMPT = """' in prompt_source or "SYSTEM_PROMPT=\"\"\"" in prompt_source

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(source(cell), f"{path.name}:cell-{index}", "exec")

    all_code = "\n".join(source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "CANDIDATE_FIELDS" not in all_code
    assert "CAPABILITY_MASTER_PATH" not in all_code
    assert "capability_master" not in all_code
    assert "call_llm_with_metrics" in all_code
    assert "avg_latency_seconds" in all_code
    assert "avg_input_tokens" in all_code
    assert "avg_output_tokens" in all_code

    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_system_prompt(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or "SYSTEM_PROMPT" not in source(cell):
            continue
        tree = ast.parse(source(cell))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT" for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"SYSTEM_PROMPT missing in {path.name}")


def main():
    for path in NOTEBOOKS:
        patch_notebook(path)

    prompts = {extract_system_prompt(path) for path in NOTEBOOKS}
    assert len(prompts) == 1, "SYSTEM_PROMPT differs across experiments"
    print("Updated and validated all five L3 notebooks.")


if __name__ == "__main__":
    main()
