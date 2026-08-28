from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E9 = ROOT / "09_business_needs_stage.ipynb"
E10 = ROOT / "10_business_needs_theme_batch.ipynb"


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def find_code(nb, needle):
    return next(
        source(cell)
        for cell in nb["cells"]
        if cell["cell_type"] == "code" and needle in source(cell)
    )


def validate_common(path, experiment_name):
    assert path.exists(), f"Missing {path.name}"
    nb = json.loads(path.read_text(encoding="utf-8"))
    all_text = "\n".join(source(cell) for cell in nb["cells"])

    assert f'EXPERIMENT_NAME = "{experiment_name}"' in all_text
    assert ".head(20)" in all_text

    candidates = find_code(nb, "def candidate_rows_for_stage")
    assert '"level_1_name"' not in candidates
    assert '"level_2_name"' not in candidates

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(source(cell))
    return nb, all_text


nb9, text9 = validate_common(E9, "E9_BUSINESS_NEEDS_STAGE")
builder9 = find_code(nb9, "def build_user_prompt")
assert '"business_needs"' in builder9
assert '"description"' not in builder9
assert "epic_description" not in builder9
assert "success_criteria" not in builder9
assert "ground_truth" not in builder9
run9 = find_code(nb9, "def run_predictions")
assert 'preflight["evaluation_eligible"]' in run9
assert "predict_for_stage" in run9
assert '.groupby("theme_id"' not in run9

nb10, text10 = validate_common(E10, "E10_BUSINESS_NEEDS_THEME_BATCH")
builder10 = find_code(nb10, "def build_theme_batch_prompt")
assert '"business_needs"' in builder10
assert '"description"' not in builder10
assert '"epics"' in builder10
assert '"epic_key"' in builder10
assert "epic_description" not in builder10
assert "success_criteria" not in builder10
assert "ground_truth" not in builder10
run10 = find_code(nb10, "def run_predictions")
assert 'preflight["evaluation_eligible"]' in run10
assert '.groupby("theme_id"' in run10
assert "predict_theme_batch" in run10
assert "tokens_per_epic" in find_code(nb10, "def summarize_llm_calls")

print("Business-needs-only experiment checks passed")
