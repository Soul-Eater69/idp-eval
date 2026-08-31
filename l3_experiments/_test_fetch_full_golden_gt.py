import json
from pathlib import Path

NOTEBOOK = Path("l3_experiments/00_fetch_full_golden_ground_truth.ipynb")

assert NOTEBOOK.exists(), f"Missing notebook: {NOTEBOOK}"

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4

code_cells = [
    cell["source"]
    for cell in notebook["cells"]
    if cell.get("cell_type") == "code"
]
code = "\n\n".join(code_cells)

assert "full_golden.parquet" in code
assert "pd.read_parquet" in code
assert "epic_keys" in code
assert 'customfield_18603' in code
assert "epic_l3_ground_truth_full_golden.xlsx" in code
assert "gt_not_fully_retrievable" not in code

for index, source in enumerate(code_cells, start=1):
    compile(source, f"cell_{index}", "exec")

print("GT fetch notebook structure is valid")
# workflow trigger
