import json
from pathlib import Path

EXPERIMENTS = [
    Path("l3_experiments/12_business_needs_stage_custom_prompt.ipynb"),
    Path("l3_experiments/13_theme_description_stage_custom_prompt.ipynb"),
    Path("l3_experiments/14_theme_needs_description_stage_custom_prompt.ipynb"),
]


def code_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        cell.get("source", "")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(code_cells, start=1):
        compile(source, f"{path.name}:cell_{index}", "exec")
    return "\n\n".join(code_cells)


for path in EXPERIMENTS:
    code = code_text(path)
    assert "def resolve_data_path(" in code, path
    assert '"full_golden.parquet"' in code and '"L3_FULL_GOLDEN_PATH"' in code, path
    assert '"VSSrv.csv"' in code and '"L3_STAGE_PATH"' in code, path
    assert '"VSSCaprv (1).csv"' in code and '"L3_STAGE_CAPABILITY_MAP_PATH"' in code, path
    assert '"results/epic_l3_ground_truth_full_golden.xlsx"' in code, path
    assert '"L3_GROUND_TRUTH_PATH"' in code, path
    assert 'NOTEBOOK_DIR.parent' in code, path

print("Mixed notebook/root data-path resolution is configured")
