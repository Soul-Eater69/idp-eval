import json
from pathlib import Path

NOTEBOOK = Path("l3_experiments/00_fetch_full_golden_ground_truth.ipynb")

assert NOTEBOOK.exists(), f"Missing notebook: {NOTEBOOK}"

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
code_cells = [
    cell["source"]
    for cell in notebook["cells"]
    if cell.get("cell_type") == "code"
]
code = "\n\n".join(code_cells)

assert "VSSCaprv (1).csv" in code
assert "epic_vss" in code
assert "gt_not_fully_retrievable" in code
assert "missing_gt_l3_ids" in code
assert "Valid Epics:" in code
assert "Invalid Epics:" in code

for index, source in enumerate(code_cells, start=1):
    compile(source, f"cell_{index}", "exec")

print("Full golden validity section is structurally valid")
