from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "09_business_needs_stage.ipynb"

EXPERIMENTS = [
    {
        "filename": "12_business_needs_stage_custom_prompt.ipynb",
        "name": "E12_BUSINESS_NEEDS_STAGE_CUSTOM_PROMPT",
        "title": "# Experiment 12 — Theme Business Needs + Stage + L3 — Custom Prompt, IDs Only",
        "summary": "Theme Business Needs + Value Stream Stage + base L3 candidates. Theme description, Epic context, hierarchy, ground truth, and model reasons are excluded.",
        "sees": """## What the LLM sees

```text
task
theme
  └─ business_needs
value_stream_stage
  ├─ stage_id
  ├─ stage_name
  ├─ stage_description
  ├─ entrance_criteria
  └─ exit_criteria
candidate_l3_capabilities[]
  ├─ capability_id
  ├─ capability_name
  ├─ capability_description
  └─ capability_tier
```

**Not sent to the LLM:** Theme description, Epic description, Epic success criteria, L1/L2 hierarchy, ground truth, model reasons.
""",
        "theme": '{"business_needs": theme["theme_business_needs"]}',
        "prompt": """You are performing Level 3 business capability classification for one Epic.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use the supplied Theme Business Needs and Value Stream Stage context to select the candidate L3 capabilities materially represented.

EVIDENCE

Theme Business Needs describes the business outcomes and needs the Theme is intended to address.

Value Stream Stage defines the business activity boundary relevant to the Epic.

For each candidate L3:
- capability_id is the exact identifier to return when the capability is selected.
- capability_description is the primary semantic definition of the business function.
- capability_name is the supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer meaning from capability_id.

SELECTION RULES

Compare the Theme Business Needs and Value Stream Stage context against the candidate L3 capability definitions.

Select every candidate L3 capability whose business function is directly supported by the supplied evidence.

Do not select a capability merely because:
- it belongs to the supplied Value Stream Stage,
- it shares similar keywords,
- it is generally related to the Theme,
- it is adjacent, upstream, or downstream,
- it provides data or support to another capability.

Only return capability_id values that exist in the supplied candidate list.

If none are supported, return an empty list.

OUTPUT

Return JSON only:

{"l3":["CAP00000123","CAP00000456"]}

Do not return reasons, explanations, Markdown, or additional fields.""",
    },
    {
        "filename": "13_theme_description_stage_custom_prompt.ipynb",
        "name": "E13_THEME_DESCRIPTION_STAGE_CUSTOM_PROMPT",
        "title": "# Experiment 13 — Theme Description + Stage + L3 — Custom Prompt, IDs Only",
        "summary": "Theme Description + Value Stream Stage + base L3 candidates. Theme Business Needs, Epic context, hierarchy, ground truth, and model reasons are excluded.",
        "sees": """## What the LLM sees

```text
task
theme
  └─ description
value_stream_stage
  ├─ stage_id
  ├─ stage_name
  ├─ stage_description
  ├─ entrance_criteria
  └─ exit_criteria
candidate_l3_capabilities[]
  ├─ capability_id
  ├─ capability_name
  ├─ capability_description
  └─ capability_tier
```

**Not sent to the LLM:** Theme Business Needs, Epic description, Epic success criteria, L1/L2 hierarchy, ground truth, model reasons.
""",
        "theme": '{"description": theme["theme_description"]}',
        "prompt": """You are performing Level 3 business capability classification for one Epic.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use the supplied Theme Description and Value Stream Stage context to select the candidate L3 capabilities materially represented.

EVIDENCE

Theme Description describes the functional scope and intent of the Theme.

Value Stream Stage defines the business activity boundary relevant to the Epic.

For each candidate L3:
- capability_id is the exact identifier to return when the capability is selected.
- capability_description is the primary semantic definition of the business function.
- capability_name is the supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer meaning from capability_id.

SELECTION RULES

Compare the Theme Description and Value Stream Stage context against the candidate L3 capability definitions.

Select every candidate L3 capability whose business function is directly supported by the supplied Theme Description within the supplied Value Stream Stage context.

Do not select a capability merely because:
- it belongs to the supplied Value Stream Stage,
- it shares similar terminology,
- it is broadly related to the Theme,
- it is adjacent, upstream, or downstream,
- it provides supporting functionality to another capability.

Only return capability_id values that exist in the supplied candidate list.

If none are supported, return an empty list.

OUTPUT

Return JSON only:

{"l3":["CAP00000123"]}

Do not return reasons, explanations, Markdown, or additional fields.""",
    },
    {
        "filename": "14_theme_needs_description_stage_custom_prompt.ipynb",
        "name": "E14_THEME_NEEDS_DESCRIPTION_STAGE_CUSTOM_PROMPT",
        "title": "# Experiment 14 — Theme Business Needs + Description + Stage + L3 — Custom Prompt, IDs Only",
        "summary": "Theme Business Needs + Theme Description + Value Stream Stage + base L3 candidates. Epic context, hierarchy, ground truth, and model reasons are excluded.",
        "sees": """## What the LLM sees

```text
task
theme
  ├─ business_needs
  └─ description
value_stream_stage
  ├─ stage_id
  ├─ stage_name
  ├─ stage_description
  ├─ entrance_criteria
  └─ exit_criteria
candidate_l3_capabilities[]
  ├─ capability_id
  ├─ capability_name
  ├─ capability_description
  └─ capability_tier
```

**Not sent to the LLM:** Epic description, Epic success criteria, L1/L2 hierarchy, ground truth, model reasons.
""",
        "theme": '{"business_needs": theme["theme_business_needs"], "description": theme["theme_description"]}',
        "prompt": """You are performing Level 3 business capability classification for one Epic.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use the supplied Theme Business Needs, Theme Description, and Value Stream Stage context to select the candidate L3 capabilities materially represented.

EVIDENCE

Theme Business Needs describes the business outcomes and needs the Theme is intended to address.

Theme Description provides additional functional context about the Theme's scope and intent.

Value Stream Stage defines the business activity boundary relevant to the Epic.

For each candidate L3:
- capability_id is the exact identifier to return when the capability is selected.
- capability_description is the primary semantic definition of the business function.
- capability_name is the supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer meaning from capability_id.

SELECTION RULES

Use Theme Business Needs as the primary business evidence.

Use Theme Description as supporting context to clarify the intended business function and scope.

Use Value Stream Stage to constrain the classification to the relevant business activity boundary.

Compare this evidence against the candidate L3 capability definitions.

Select every candidate L3 capability whose business function is directly supported by the supplied evidence.

Do not use Theme Description to broaden the selection beyond what is supported by the Theme Business Needs.

Do not select a capability merely because:
- it belongs to the supplied Value Stream Stage,
- it shares similar terminology,
- it is broadly related to the Theme,
- it is adjacent, upstream, or downstream,
- it provides supporting functionality to another capability.

Only return capability_id values that exist in the supplied candidate list.

If none are supported, return an empty list.

OUTPUT

Return JSON only:

{"l3":["CAP00000123","CAP00000456"]}

Do not return reasons, explanations, Markdown, or additional fields.""",
    },
]

VALIDATOR = '''def validate_l3_id_response(payload, candidate_ids):
    if set(payload) != {"l3"}:
        raise ValueError("LLM response must contain exactly one top-level field: l3.")
    raw = payload.get("l3")
    if not isinstance(raw, list):
        raise ValueError("LLM response must contain an 'l3' list.")
    allowed = {str(value).strip() for value in candidate_ids if str(value).strip()}
    selected = []
    seen = set()
    for index, capability_id in enumerate(raw, start=1):
        if not isinstance(capability_id, str):
            raise ValueError(f"L3 selection #{index} must be a capability_id string.")
        capability_id = capability_id.strip()
        if not capability_id:
            raise ValueError(f"L3 selection #{index} is empty.")
        if capability_id not in allowed:
            raise ValueError(f"LLM selected {capability_id}, which is not a supplied candidate.")
        if capability_id in seen:
            raise ValueError(f"LLM returned duplicate capability_id {capability_id}.")
        seen.add(capability_id)
        selected.append(capability_id)
    return selected
'''


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def after_heading(nb, heading):
    for i, cell in enumerate(nb["cells"][:-1]):
        if cell.get("cell_type") == "markdown" and source(cell).strip().startswith(heading):
            return nb["cells"][i + 1]
    raise RuntimeError(f"Missing {heading}")


def replace_once(text, old, new):
    if old not in text:
        raise RuntimeError(f"Expected template text not found: {old[:80]!r}")
    return text.replace(old, new, 1)


def prompt_cell(config):
    return f'''SYSTEM_PROMPT = {config["prompt"]!r}\n\ndef build_user_prompt(theme, epic, stage, candidate_rows):\n    payload = {{\n        "task": "Select directly supported L3 business capability IDs from the supplied candidates.",\n        "theme": {config["theme"]},\n        "value_stream_stage": stage,\n        "candidate_l3_capabilities": candidate_rows,\n    }}\n    return json.dumps(payload, ensure_ascii=False, indent=2)\n'''


def make_prediction_ids_only(text):
    text = replace_once(text, "def predict_for_stage", VALIDATOR + "\n\ndef predict_for_stage")
    text = replace_once(
        text,
        '''selections = validate_l3_response(\n        parse_json_response(raw_response),\n        [candidate["capability_id"] for candidate in candidates],\n        allow_empty=True,\n        max_selected=3,\n    )''',
        '''selections = validate_l3_id_response(\n        parse_json_response(raw_response),\n        [candidate["capability_id"] for candidate in candidates],\n    )''',
    )
    text = replace_once(text, "        reasons = []\n", "")
    text = replace_once(
        text,
        '''selected_ids = [\n                    selection["capability_id"]\n                    for selection in result["selections"]\n                ]''',
        '''selected_ids = result["selections"]''',
    )
    text = replace_once(
        text,
        '''stage_predictions.append({\n                    "stage_id": stage_id,\n                    "selections": result["selections"],\n                })\n                for selection in result["selections"]:\n                    predicted_ids.add(selection["capability_id"])\n                    reasons.append({"stage_id": stage_id, **selection})''',
        '''stage_predictions.append({\n                    "stage_id": stage_id,\n                    "selected_l3_ids": selected_ids,\n                })\n                predicted_ids.update(selected_ids)''',
    )
    text = replace_once(
        text,
        '            "model_reasons": json.dumps(reasons, ensure_ascii=False),\n',
        "",
    )
    return text


def build(template, config):
    nb = copy.deepcopy(template)
    nb["cells"][0]["source"] = config["title"] + "\n\n" + config["summary"]
    nb["cells"][1]["source"] = config["sees"]

    config_cell = after_heading(nb, "## Configuration and imports")
    config_source = source(config_cell).replace("    validate_l3_response,\n", "")
    config_source = replace_once(
        config_source,
        'EXPERIMENT_NAME = "E9_BUSINESS_NEEDS_STAGE"',
        f'EXPERIMENT_NAME = "{config["name"]}"',
    )
    config_cell["source"] = config_source

    after_heading(nb, "## Production prompt")["source"] = prompt_cell(config)
    prediction_cell = after_heading(nb, "## Prediction")
    prediction_cell["source"] = make_prediction_ids_only(source(prediction_cell))
    return nb


def verify(path, config):
    nb = json.loads(path.read_text(encoding="utf-8"))
    p = source(after_heading(nb, "## Production prompt"))
    pred = source(after_heading(nb, "## Prediction"))
    cfg = source(after_heading(nb, "## Configuration and imports"))
    assert config["name"] in cfg
    assert "validate_l3_response" not in cfg
    assert "validate_l3_id_response" in pred
    assert "model_reasons" not in pred
    assert "max_selected" not in pred
    assert "Select 0 to 3" not in p
    assert "smallest defensible" not in p.lower()
    assert '"reason"' not in p
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{path.name}:cell", "exec")


def main():
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for config in EXPERIMENTS:
        path = ROOT / config["filename"]
        path.write_text(json.dumps(build(template, config), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        verify(path, config)
        print(f"Generated and verified {path.name}")


if __name__ == "__main__":
    main()
