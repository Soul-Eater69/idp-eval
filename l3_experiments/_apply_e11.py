from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "10_business_needs_theme_batch.ipynb"
OUTPUT = ROOT / "11_business_needs_theme_batch_with_hierarchy.ipynb"


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def replace_code_cell(nb, needle, transform):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = source(cell)
        if needle in text:
            cell["source"] = transform(text)
            return
    raise AssertionError(f"Code cell not found: {needle}")


nb = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))

nb["cells"][0]["source"] = (
    "# Experiment 11 — Theme Business Needs Batch + Hierarchy\n\n"
    "Same Theme-batch design as E10: Theme business needs + each Epic's Stage(s), "
    "with L1/L2 hierarchy added to every L3 candidate."
)
nb["cells"][1]["source"] = (
    "## What the LLM sees\n\n"
    "One LLM call is made per Theme for all preflight-valid Epics under that Theme.\n\n"
    "```text\n"
    "task\n"
    "theme\n"
    "  └─ business_needs\n"
    "epics[]\n"
    "  ├─ epic_key\n"
    "  └─ stages[]\n"
    "      ├─ value_stream_stage\n"
    "      │   ├─ stage_id\n"
    "      │   ├─ stage_name\n"
    "      │   ├─ stage_description\n"
    "      │   ├─ entrance_criteria\n"
    "      │   └─ exit_criteria\n"
    "      └─ candidate_l3_capabilities[]\n"
    "          ├─ capability_id\n"
    "          ├─ capability_name\n"
    "          ├─ capability_description\n"
    "          ├─ capability_tier\n"
    "          ├─ level_1_name\n"
    "          └─ level_2_name\n"
    "selection_instruction\n"
    "```\n\n"
    "**Not sent to the LLM:** Theme description, Epic description, Epic success criteria, ground truth.\n"
)

replace_code_cell(
    nb,
    'EXPERIMENT_NAME = "E10_BUSINESS_NEEDS_THEME_BATCH"',
    lambda text: text.replace(
        'EXPERIMENT_NAME = "E10_BUSINESS_NEEDS_THEME_BATCH"',
        'EXPERIMENT_NAME = "E11_BUSINESS_NEEDS_THEME_BATCH_WITH_HIERARCHY"',
    ),
)


def add_hierarchy(text):
    old = '''            "capability_tier": clean_text(row["Capability Tier"]),\n'''
    new = '''            "capability_tier": clean_text(row["Capability Tier"]),\n            "level_1_name": clean_text(row["Level 1 Name"]),\n            "level_2_name": clean_text(row["Level 2 Name"]),\n'''
    if old not in text:
        raise AssertionError("candidate hierarchy insertion point not found")
    return text.replace(old, new, 1)


replace_code_cell(nb, "def candidate_rows_for_stage", add_hierarchy)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        ast.parse(source(cell))

OUTPUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT.name}")
