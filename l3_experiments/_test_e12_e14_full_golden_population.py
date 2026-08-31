import json
from pathlib import Path

GT_NOTEBOOK = Path("l3_experiments/00_fetch_full_golden_ground_truth.ipynb")
EXPERIMENTS = [
    Path("l3_experiments/12_business_needs_stage_custom_prompt.ipynb"),
    Path("l3_experiments/13_theme_description_stage_custom_prompt.ipynb"),
    Path("l3_experiments/14_theme_needs_description_stage_custom_prompt.ipynb"),
]


def code_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        cell.get("source", "")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(cells, start=1):
        compile(source, f"{path.name}:cell_{index}", "exec")
    return "\n\n".join(cells)


gt_code = code_text(GT_NOTEBOOK)
assert 'sheet_name="evaluation_population"' in gt_code
assert '"theme_key"' in gt_code
assert '"epic_key"' in gt_code
assert '"stage_ids"' in gt_code
assert '"gt_l3_ids"' in gt_code
assert '"candidate_l3_ids"' in gt_code
assert 'if_sheet_exists="replace"' in gt_code

expected_prompt_markers = {
    "12_business_needs_stage_custom_prompt.ipynb": "Use the supplied Theme Business Needs and Value Stream Stage context",
    "13_theme_description_stage_custom_prompt.ipynb": "Use the supplied Theme Description and Value Stream Stage context",
    "14_theme_needs_description_stage_custom_prompt.ipynb": "Use the supplied Theme Business Needs, Theme Description, and Value Stream Stage context",
}

for path in EXPERIMENTS:
    code = code_text(path)
    assert "full_golden.parquet" in code
    assert "epic_l3_ground_truth_full_golden.xlsx" in code
    assert 'sheet_name="evaluation_population"' in code
    assert "SAMPLE_SIZE = 50" in code
    assert "SAMPLE_SEED = 42" in code
    assert "random_state=SAMPLE_SEED" in code
    assert "epic_gen.csv" not in code
    assert "customfield_18700" not in code
    assert "epic_stage_ids" not in code
    assert expected_prompt_markers[path.name] in code
    assert "Do not return reasons, explanations, Markdown, or additional fields." in code

print("E12-E14 full-golden population structure is valid")
