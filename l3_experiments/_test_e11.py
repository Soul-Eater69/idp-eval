from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E11 = ROOT / "11_business_needs_theme_batch_with_hierarchy.ipynb"


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def find_code(nb, needle):
    return next(
        source(cell)
        for cell in nb["cells"]
        if cell["cell_type"] == "code" and needle in source(cell)
    )


assert E11.exists(), "E11 notebook is missing"
nb = json.loads(E11.read_text(encoding="utf-8"))
all_text = "\n".join(source(cell) for cell in nb["cells"])

assert 'EXPERIMENT_NAME = "E11_BUSINESS_NEEDS_THEME_BATCH_WITH_HIERARCHY"' in all_text
assert ".head(20)" in all_text

candidate_cell = find_code(nb, "def candidate_rows_for_stage")
assert '"level_1_name"' in candidate_cell
assert '"level_2_name"' in candidate_cell
assert 'row["Level 1 Name"]' in candidate_cell
assert 'row["Level 2 Name"]' in candidate_cell

prompt_builder = find_code(nb, "def build_theme_batch_prompt")
assert '"business_needs"' in prompt_builder
assert '"description"' not in prompt_builder
assert "epic_description" not in prompt_builder
assert "success_criteria" not in prompt_builder
assert "ground_truth" not in prompt_builder

prediction = find_code(nb, "def run_predictions")
assert 'preflight["evaluation_eligible"]' in prediction
assert '.groupby("theme_id"' in prediction
assert "predict_theme_batch" in prediction

summary = find_code(nb, "def summarize_llm_calls")
assert "tokens_per_epic" in summary

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        ast.parse(source(cell))

print("E11 notebook checks passed")
