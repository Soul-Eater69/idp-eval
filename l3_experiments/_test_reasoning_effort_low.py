import ast
import json
from pathlib import Path

COMMON = Path("l3_experiments/common.py")
EXPERIMENTS = [
    Path("l3_experiments/12_business_needs_stage_custom_prompt.ipynb"),
    Path("l3_experiments/13_theme_description_stage_custom_prompt.ipynb"),
    Path("l3_experiments/14_theme_needs_description_stage_custom_prompt.ipynb"),
]

common_source = COMMON.read_text(encoding="utf-8")
common_tree = ast.parse(common_source)
call_fn = next(
    node
    for node in common_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "call_llm_with_metrics"
)
assert call_fn.args.kwarg is not None, "call_llm_with_metrics must accept **options"
assert call_fn.args.kwarg.arg == "options"
assert "**options" in common_source, "gateway.complete must receive forwarded options"

for path in EXPERIMENTS:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n\n".join(
        cell.get("source", "")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert 'reasoning_effort="low"' in code, path

print("E12-E14 explicitly request low reasoning effort")
