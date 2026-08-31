import json
import re
from pathlib import Path

COMMON = Path("l3_experiments/common.py")
EXPERIMENTS = [
    Path("l3_experiments/12_business_needs_stage_custom_prompt.ipynb"),
    Path("l3_experiments/13_theme_description_stage_custom_prompt.ipynb"),
    Path("l3_experiments/14_theme_needs_description_stage_custom_prompt.ipynb"),
]

common = COMMON.read_text(encoding="utf-8")
old_sig = '''def call_llm_with_metrics(
    gateway: Any,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, float | int | None]]:'''
new_sig = '''def call_llm_with_metrics(
    gateway: Any,
    system_prompt: str,
    user_prompt: str,
    **options: Any,
) -> tuple[str, dict[str, float | int | None]]:'''
if old_sig not in common:
    raise RuntimeError("Expected call_llm_with_metrics signature not found")
common = common.replace(old_sig, new_sig, 1)

old_complete = '''    response = gateway.complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )'''
new_complete = '''    response = gateway.complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **options,
    )'''
if old_complete not in common:
    raise RuntimeError("Expected gateway.complete call not found")
common = common.replace(old_complete, new_complete, 1)
compile(common, str(COMMON), "exec")
COMMON.write_text(common, encoding="utf-8")

pattern = re.compile(
    r"call_llm_with_metrics\(\s*gateway,\s*SYSTEM_PROMPT,\s*user_prompt,\s*\)",
    flags=re.MULTILINE,
)
replacement = '''call_llm_with_metrics(
        gateway,
        SYSTEM_PROMPT,
        user_prompt,
        reasoning_effort="low",
    )'''

for path in EXPERIMENTS:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    replacements = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        updated, count = pattern.subn(replacement, source)
        if count:
            cell["source"] = updated
            replacements += count

    if replacements != 1:
        raise RuntimeError(f"Expected one LLM call update in {path}, found {replacements}")

    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") == "code":
            compile(cell.get("source", ""), f"{path.name}:cell_{index}", "exec")

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {path}")

print("Enabled reasoning_effort=low for E12-E14")
