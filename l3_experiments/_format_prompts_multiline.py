import ast
import json
from pathlib import Path

BASE = Path("l3_experiments")


def prompt_value(source: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT":
                    return ast.literal_eval(node.value)
    raise ValueError("SYSTEM_PROMPT assignment not found")


def format_prompt_cell(source: str) -> str:
    marker = "SYSTEM_PROMPT = "
    start = source.find(marker)
    if start < 0:
        return source

    line_end = source.find("\n", start)
    if line_end < 0:
        line_end = len(source)

    before_value = prompt_value(source)

    if "'''" not in before_value:
        delimiter = "'''"
    elif '"""' not in before_value:
        delimiter = '"""'
    else:
        raise ValueError("Prompt contains both triple-quote delimiters")

    formatted_assignment = f"SYSTEM_PROMPT = {delimiter}{before_value}{delimiter}"
    formatted = source[:start] + formatted_assignment + source[line_end:]

    after_value = prompt_value(formatted)
    if before_value != after_value:
        raise AssertionError("Prompt text changed while formatting")

    return formatted


def notebook_number(path: Path) -> int | None:
    prefix = path.name.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


targets = []
for path in sorted(BASE.glob("*.ipynb")):
    number = notebook_number(path)
    if number is not None and 12 <= number <= 16:
        targets.append(path)

if not targets:
    raise RuntimeError("No notebooks 12-16 found")

changed = []
for path in targets:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found_prompt = False

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")
        if "SYSTEM_PROMPT = " in source:
            found_prompt = True
            new_source = format_prompt_cell(source)
            cell["source"] = new_source

        compile(cell.get("source", ""), f"{path.name}:cell_{index}", "exec")

    if not found_prompt:
        raise RuntimeError(f"No SYSTEM_PROMPT found in {path.name}")

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    changed.append(path.name)

print("Formatted prompts as multiline blocks:")
for name in changed:
    print(f"- {name}")
