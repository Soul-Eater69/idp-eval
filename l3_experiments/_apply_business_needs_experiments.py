from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E1 = ROOT / "01_theme_stage.ipynb"
E7 = ROOT / "07_theme_batch.ipynb"
E9 = ROOT / "09_business_needs_stage.ipynb"
E10 = ROOT / "10_business_needs_theme_batch.ipynb"


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def set_markdown(nb, index, text):
    assert nb["cells"][index]["cell_type"] == "markdown"
    nb["cells"][index]["source"] = text


def replace_code_cell(nb, needle, transform):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = source(cell)
        if needle in text:
            cell["source"] = transform(text)
            return
    raise AssertionError(f"Code cell not found: {needle}")


def finish(nb, output):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            ast.parse(source(cell))
    output.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output.name}")


def build_e9():
    nb = copy.deepcopy(json.loads(E1.read_text(encoding="utf-8")))

    set_markdown(
        nb,
        0,
        "# Experiment 9 — Theme Business Needs + Stage\n\n"
        "Theme business needs + Stage + base L3 candidates. Theme description, Epic context, and hierarchy are excluded.",
    )
    set_markdown(
        nb,
        1,
        "## What the LLM sees\n\n"
        "```text\n"
        "task\n"
        "theme\n"
        "  └─ business_needs\n"
        "value_stream_stage\n"
        "  ├─ stage_id\n"
        "  ├─ stage_name\n"
        "  ├─ stage_description\n"
        "  ├─ entrance_criteria\n"
        "  └─ exit_criteria\n"
        "candidate_l3_capabilities[]\n"
        "  ├─ capability_id\n"
        "  ├─ capability_name\n"
        "  ├─ capability_description\n"
        "  └─ capability_tier\n"
        "selection_instruction\n"
        "```\n\n"
        "**Not sent to the LLM:** Theme description, Epic description, Epic success criteria, L1/L2 hierarchy, ground truth.\n",
    )

    def config(text):
        text = text.replace('EXPERIMENT_NAME = "E1_THEME_STAGE"', 'EXPERIMENT_NAME = "E9_BUSINESS_NEEDS_STAGE"')
        if ".head(20)" not in text:
            text = text.replace("    .drop_duplicates()\n    .tolist()", "    .drop_duplicates()\n    .head(20)\n    .tolist()")
        return text

    replace_code_cell(nb, 'EXPERIMENT_NAME = "E1_THEME_STAGE"', config)

    def prompt(text):
        system_start = text.index('SYSTEM_PROMPT = """')
        build_start = text.index("def build_user_prompt")
        system = text[system_start:build_start]
        system = system.replace(
            "EVIDENCE PRIORITY\\nUse only fields that are present, in this order when available:\\n1. Epic success criteria\\n2. Epic description\\n3. Value Stream Stage context\\n4. Theme business needs\\n5. Theme description\\n\\nIf Epic context is absent, do not assume or refer to it.",
            "EVIDENCE PRIORITY\\nUse only the supplied fields, in this order:\\n1. Value Stream Stage context\\n2. Theme business needs\\n\\nTheme description, Epic description, and Epic success criteria are not supplied. Do not assume them.",
        )
        builder = '''def build_user_prompt(theme, epic, stage, candidate_rows):
    payload = {
        "task": "Select the materially represented L3 business capabilities from the supplied candidates.",
        "theme": {
            "business_needs": theme["theme_business_needs"],
        },
        "value_stream_stage": stage,
        "candidate_l3_capabilities": candidate_rows,
        "selection_instruction": "Select 0 to 3 candidates; return an empty l3 list when none has direct evidence.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
'''
        return system + builder

    replace_code_cell(nb, "SYSTEM_PROMPT", prompt)
    finish(nb, E9)


def build_e10():
    nb = copy.deepcopy(json.loads(E7.read_text(encoding="utf-8")))

    set_markdown(
        nb,
        0,
        "# Experiment 10 — Theme Business Needs Batch\n\n"
        "Theme business needs + each Epic's Stage(s) + base L3 candidates, with all preflight-valid Epics in a Theme classified together in one LLM call.",
    )
    set_markdown(
        nb,
        1,
        "## What the LLM sees\n\nOne LLM call is made per Theme for all preflight-valid Epics under that Theme.\n\n"
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
        "          └─ capability_tier\n"
        "selection_instruction\n"
        "```\n\n"
        "**Not sent to the LLM:** Theme description, Epic description, Epic success criteria, L1/L2 hierarchy, ground truth.\n",
    )

    replace_code_cell(
        nb,
        'EXPERIMENT_NAME = "E7_THEME_BATCH"',
        lambda text: text.replace(
            'EXPERIMENT_NAME = "E7_THEME_BATCH"',
            'EXPERIMENT_NAME = "E10_BUSINESS_NEEDS_THEME_BATCH"',
        ),
    )

    def prompt(text):
        text = text.replace(
            "1. Value Stream Stage context for that Epic\\n2. Theme business needs\\n3. Theme description",
            "1. Value Stream Stage context for that Epic\\n2. Theme business needs",
        )
        text = text.replace(
            "Epic description and Epic success criteria are not supplied. Do not assume them.",
            "Theme description, Epic description, and Epic success criteria are not supplied. Do not assume them.",
        )
        text = text.replace(
            '        "theme": {\n            "business_needs": theme["theme_business_needs"],\n            "description": theme["theme_description"],\n        },',
            '        "theme": {\n            "business_needs": theme["theme_business_needs"],\n        },',
        )
        return text

    replace_code_cell(nb, "SYSTEM_PROMPT", prompt)
    finish(nb, E10)


build_e9()
build_e10()
