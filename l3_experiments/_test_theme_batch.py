from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E7_PATH = ROOT / "07_theme_batch.ipynb"
E8_PATH = ROOT / "08_theme_batch_with_hierarchy.ipynb"


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def find_code(nb, needle):
    return next(
        source(cell)
        for cell in nb["cells"]
        if cell["cell_type"] == "code" and needle in source(cell)
    )


def validate_notebook(path, experiment_name, expect_hierarchy):
    assert path.exists(), f"Missing {path.name}"
    nb = json.loads(path.read_text(encoding="utf-8"))
    all_text = "\n".join(source(cell) for cell in nb["cells"])

    assert f'EXPERIMENT_NAME = "{experiment_name}"' in all_text
    assert ".head(20)" in all_text, "Batch experiments must match the 20-theme comparison population"

    builder = find_code(nb, "def build_theme_batch_prompt")
    assert '"epics"' in builder
    assert '"epic_key"' in builder
    assert '"stages"' in builder
    for forbidden in ("epic_description", "success_criteria", "ground_truth"):
        assert forbidden not in builder, f"{forbidden} leaked into grouped LLM prompt builder"

    candidate_cell = find_code(nb, "def candidate_rows_for_stage")
    if expect_hierarchy:
        assert '"level_1_name"' in candidate_cell
        assert '"level_2_name"' in candidate_cell
    else:
        assert '"level_1_name"' not in candidate_cell
        assert '"level_2_name"' not in candidate_cell

    prediction = find_code(nb, "def run_predictions")
    assert 'preflight["evaluation_eligible"]' in prediction
    assert '.groupby("theme_id"' in prediction
    assert "predict_theme_batch" in prediction
    assert "epics_in_call" in prediction

    validator = find_code(nb, "def validate_theme_batch_response")
    assert "expected_epic_keys" in validator
    assert "allowed_ids_by_epic" in validator
    assert "max_selected=3" in validator

    summary_cell = find_code(nb, "def summarize_llm_calls")
    assert "tokens_per_epic" in summary_cell

    prompt_cell = find_code(nb, "SYSTEM_PROMPT")
    assert '"epics"' in prompt_cell
    assert '"epic_key"' in prompt_cell
    assert "Return JSON only" in prompt_cell

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(source(cell))


validate_notebook(E7_PATH, "E7_THEME_BATCH", expect_hierarchy=False)
validate_notebook(E8_PATH, "E8_THEME_BATCH_WITH_HIERARCHY", expect_hierarchy=True)

print("Theme-batch notebook checks passed")
