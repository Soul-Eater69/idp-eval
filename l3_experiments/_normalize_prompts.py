from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    ROOT / "01_theme_stage.ipynb",
    ROOT / "02_full_context.ipynb",
    ROOT / "03_no_theme_description.ipynb",
    ROOT / "04_no_theme.ipynb",
    ROOT / "05_full_with_hierarchy.ipynb",
]


def cell_source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def normalize_prompt_source(text: str) -> str:
    tree = ast.parse(text)
    assignment = None
    prompt_value = None

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT"
            for target in node.targets
        ):
            assignment = node
            prompt_value = ast.literal_eval(node.value)
            break

    if assignment is None or not isinstance(prompt_value, str):
        raise AssertionError("SYSTEM_PROMPT assignment not found")
    if '"""' in prompt_value:
        raise AssertionError("SYSTEM_PROMPT contains triple quotes")

    lines = text.splitlines(keepends=True)
    start = assignment.lineno - 1
    end = assignment.end_lineno
    replacement = 'SYSTEM_PROMPT = """' + prompt_value + '"""\n'
    return "".join(lines[:start]) + replacement + "".join(lines[end:])


for path in NOTEBOOKS:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    prompt_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "SYSTEM_PROMPT" in cell_source(cell)
        and "def build_user_prompt" in cell_source(cell)
    )
    prompt_cell["source"] = normalize_prompt_source(cell_source(prompt_cell))
    compile(prompt_cell["source"], path.name, "exec")
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

print("Normalized SYSTEM_PROMPT formatting in all five notebooks.")
