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
    assert "from common import (" in config_text
    assert "call_llm_with_metrics" in config_text
    assert "EXPERIMENT_NAME =" in config_text
    assert "INSPECTION_THEME_ID = None" in config_text

    inspection = code_after_heading(notebook, "## Single-example inspection")
    inspection_text = source(inspection)
    assert "CALL METRICS" in inspection_text
    assert "predict_for_stage" in inspection_text

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
