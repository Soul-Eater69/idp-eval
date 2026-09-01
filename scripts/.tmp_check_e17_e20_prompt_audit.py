from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOKS = {
    17: Path("l3_experiments/17_theme_needs_description_stage_individual.ipynb"),
    18: Path("l3_experiments/18_business_needs_stage_individual.ipynb"),
    19: Path("l3_experiments/19_theme_needs_description_stage_batch.ipynb"),
    20: Path("l3_experiments/20_business_needs_stage_batch.ipynb"),
}
COMBINED = {17, 19}
BATCH = {19, 20}
COMPARISON = Path("l3_experiments/e17_e20_results_comparison.md")
COMMON = Path("l3_experiments/common.py")


def source(cell):
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


errors = []

# Shared prompt-audit infrastructure.
common = COMMON.read_text(encoding="utf-8")
try:
    compile(common, str(COMMON), "exec")
except SyntaxError as exc:
    errors.append(f"common.py does not compile: {exc}")

for required in (
    "_prompt_audit_log",
    "def reset_prompt_audit_log",
    "def get_prompt_audit_log",
    '"system_prompt": system_prompt',
    '"user_prompt": user_prompt',
    '"prompt_call_index"',
    'status="error"',
    'status="ok"',
):
    if required not in common:
        errors.append(f"common.py missing {required!r}")

for number, path in NOTEBOOKS.items():
    if not path.exists():
        errors.append(f"missing notebook: {path}")
        continue

    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [source(cell) for cell in nb["cells"] if cell.get("cell_type") == "code"]
    all_code = "\n\n".join(code_cells)

    for index, code in enumerate(code_cells):
        try:
            compile(code, f"{path}:code-cell-{index}", "exec")
        except SyntaxError as exc:
            errors.append(f"{path}: code cell {index} does not compile: {exc}")

    if "def theme_context(" not in all_code:
        errors.append(f"{path}: theme_context() is missing")
    if "reset_prompt_audit_log()" not in all_code:
        errors.append(f"{path}: prompt audit is not reset before run")
    if "get_prompt_audit_log()" not in all_code:
        errors.append(f"{path}: prompt audit is not read after run")
    for text in ("prompt_log", "prompt_sample", '"prompt_log"', '"prompt_sample"'):
        if text not in all_code:
            errors.append(f"{path}: missing prompt workbook artifact {text!r}")

    prompt_cells = [code for code in code_cells if "SYSTEM_PROMPT =" in code and "def build_user_prompt" in code]
    if len(prompt_cells) != 1:
        errors.append(f"{path}: expected exactly one production prompt cell; found {len(prompt_cells)}")
        continue

    prompt_code = prompt_cells[0]
    if "SYSTEM_PROMPT = \"\"\"" not in prompt_code:
        errors.append(f"{path}: SYSTEM_PROMPT is not a multiline triple-quoted block")

    try:
        ns = {}
        exec(prompt_code, ns)
        row = {
            "theme_business_needs": "Need to create and present a governed quote.",
            "theme_description": "Quote lifecycle modernization.",
        }
        context = ns["theme_context"](row)
        if number in COMBINED:
            if context.get("theme_description") != row["theme_description"]:
                errors.append(f"{path}: combined context does not include Theme Description")
        elif "theme_description" in context:
            errors.append(f"{path}: needs-only context unexpectedly includes Theme Description")

        stage = {
            "stage_id": "VSS000123",
            "stage_name": "Generate Quote",
            "stage_description": "Generate a governed customer quote.",
            "entrance_criteria": "Request is qualified.",
            "exit_criteria": "Quote is ready to present.",
        }
        candidate = {
            "capability_id": "CAP000456",
            "capability_name": "Quote Management",
            "capability_description": "Create and manage customer quotes.",
            "capability_tier": "3",
        }
        if number in BATCH:
            payload = [{**stage, "candidate_l3_capabilities": [candidate]}]
            user_prompt = ns["build_user_prompt"](context, payload)
        else:
            user_prompt = ns["build_user_prompt"](context, stage, [candidate])

        for expected in (
            "Theme Business Needs:",
            "Stage ID: VSS000123",
            "Stage Name: Generate Quote",
            "Stage Description: Generate a governed customer quote.",
            "Entrance Criteria: Request is qualified.",
            "Exit Criteria: Quote is ready to present.",
            "Capability ID: CAP000456",
            "Capability Name: Quote Management",
            "Capability Description: Create and manage customer quotes.",
            "Capability Tier: 3",
        ):
            if expected not in user_prompt:
                errors.append(f"{path}: formatted user prompt missing {expected!r}")
        if number in COMBINED and "Theme Description:\nQuote lifecycle modernization." not in user_prompt:
            errors.append(f"{path}: formatted prompt missing Theme Description")
        if number not in COMBINED and "Theme Description:" in user_prompt:
            errors.append(f"{path}: needs-only formatted prompt contains Theme Description")
        if user_prompt.lstrip().startswith("{"):
            errors.append(f"{path}: user prompt is JSON rather than readable formatted text")
    except Exception as exc:
        errors.append(f"{path}: synthetic prompt check failed: {exc}")

# Concise comparison report with the supplied run numbers.
if not COMPARISON.exists():
    errors.append(f"missing comparison report: {COMPARISON}")
else:
    report = COMPARISON.read_text(encoding="utf-8")
    for required in (
        "E17",
        "E18",
        "E19",
        "E20",
        "59.74%",
        "56.69%",
        "54.60%",
        "54.48%",
        "Preferred batch configuration",
        "Highest raw classification quality",
        "prompt_log",
        "prompt_sample",
    ):
        if required not in report:
            errors.append(f"comparison report missing {required!r}")

if errors:
    print("VERIFICATION FAILED")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print("VERIFICATION PASSED")
print("- common.py records exact prompts for successful and failed LLM calls")
for number, path in NOTEBOOKS.items():
    print(f"- E{number}: theme_context + readable prompt + prompt_log/prompt_sample workbook export")
print(f"- comparison report: {COMPARISON}")
