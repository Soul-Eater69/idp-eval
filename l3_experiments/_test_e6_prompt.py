from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "05_full_with_hierarchy.ipynb"
ENHANCED = ROOT / "06_enhanced_full_with_hierarchy.ipynb"


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def find_code(nb, needle):
    return next(
        source(cell)
        for cell in nb["cells"]
        if cell["cell_type"] == "code" and needle in source(cell)
    )


def system_prompt(nb):
    text = find_code(nb, "SYSTEM_PROMPT")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("SYSTEM_PROMPT not found")


assert ENHANCED.exists(), "E6 enhanced notebook does not exist yet"

base = json.loads(BASELINE.read_text(encoding="utf-8"))
e6 = json.loads(ENHANCED.read_text(encoding="utf-8"))

all_e6 = "\n".join(source(cell) for cell in e6["cells"])
assert "# Experiment 6 — Enhanced Full Context + Hierarchy" in all_e6
assert 'EXPERIMENT_NAME = "E6_ENHANCED_FULL_WITH_HIERARCHY"' in all_e6
assert ".head(20)" in all_e6, "E6 must match the current 20-theme comparison run"

prompt = system_prompt(e6)
base_prompt = system_prompt(base)
assert prompt != base_prompt
for required in (
    "FUNCTION EXTRACTION",
    "COVERAGE PASS",
    "PRUNING PASS",
    "MISSED-CAPABILITY CHECK",
    "Do not stop after finding the first valid capability",
):
    assert required in prompt, required

base_prompt_cell = find_code(base, "SYSTEM_PROMPT")
e6_prompt_cell = find_code(e6, "SYSTEM_PROMPT")
assert 'SYSTEM_PROMPT = """' in e6_prompt_cell, "Keep E6 prompt readable as a multiline string"
base_build = base_prompt_cell[base_prompt_cell.index("def build_user_prompt"):]
e6_build = e6_prompt_cell[e6_prompt_cell.index("def build_user_prompt"):]
assert e6_build == base_build, "E6 changed the user-prompt payload"
assert find_code(e6, "def candidate_rows_for_stage") == find_code(
    base, "def candidate_rows_for_stage"
), "E6 changed candidate construction"
assert find_code(e6, "def ground_truth_by_epic") == find_code(
    base, "def ground_truth_by_epic"
), "E6 changed preflight/evaluation logic"

for cell in e6["cells"]:
    if cell["cell_type"] == "code":
        ast.parse(source(cell))

print("E6 enhanced-prompt notebook checks passed")
