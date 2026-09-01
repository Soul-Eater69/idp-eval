import json
from pathlib import Path

FILES = {
    17: Path("l3_experiments/17_theme_needs_description_stage_individual.ipynb"),
    18: Path("l3_experiments/18_business_needs_stage_individual.ipynb"),
    19: Path("l3_experiments/19_theme_needs_description_stage_batch.ipynb"),
    20: Path("l3_experiments/20_business_needs_stage_batch.ipynb"),
}


def source(cell):
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def set_source(cell, value):
    cell["source"] = value


def find_code_cell(nb, needle):
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and needle in source(cell):
            return cell
    raise KeyError(f"Could not find code cell containing: {needle}")


STAGE_PAYLOAD_CODE = '''STAGE_PATH = resolve("VSSrv.csv", "L3_STAGE_PATH")
STAGE_CAPABILITY_MAP_PATH = resolve(
    "VSSCaprv (1).csv",
    "L3_STAGE_CAPABILITY_MAP_PATH",
)

stage_frame = pd.read_csv(
    STAGE_PATH,
    dtype=str,
    encoding="cp1252",
    encoding_errors="replace",
)
stage_capability_map = pd.read_csv(
    STAGE_CAPABILITY_MAP_PATH,
    dtype=str,
    encoding="cp1252",
    encoding_errors="replace",
)


def stage_context(stage_id):
    """Return the full governed metadata for one Value Stream Stage."""
    stage_id = clean(stage_id)
    match = stage_frame.loc[
        stage_frame["Value Stream Stage ID"].astype(str).str.strip().eq(stage_id)
    ]

    if match.empty:
        raise KeyError(f"No Stage metadata found for {stage_id}")

    row = match.iloc[0]
    return {
        "stage_id": stage_id,
        "stage_name": clean(row["Value Stream Stage Name"]),
        "stage_description": clean(row["Value Stream Stage Description"]),
        "entrance_criteria": clean(row["Value Stream Stage Entrance Criteria"]),
        "exit_criteria": clean(row["Value Stream Stage Exit Criteria"]),
    }


def candidate_rows_for_stage(stage_id, allowed_candidate_ids):
    """Return full governed L3 metadata for candidates belonging to one Stage."""
    stage_id = clean(stage_id)
    rows = stage_capability_map.loc[
        stage_capability_map["Value Stream Stage ID"]
        .astype(str)
        .str.strip()
        .eq(stage_id)
    ].copy()

    allowed = {clean(value) for value in allowed_candidate_ids if clean(value)}
    if allowed:
        rows = rows.loc[
            rows["Capability ID"].astype(str).str.strip().isin(allowed)
        ]

    rows = (
        rows
        .drop_duplicates(subset=["Capability ID"], keep="first")
        .sort_values(["Capability Name", "Capability ID"], kind="stable")
    )

    return [
        {
            "capability_id": clean(row["Capability ID"]),
            "capability_name": clean(row["Capability Name"]),
            "capability_description": clean(row["Capability Description"]),
            "capability_tier": clean(row["Capability Tier"]),
        }
        for _, row in rows.iterrows()
    ]


def row_stages(row):
    """Build Stage-specific prompt payloads for one evaluation record."""
    stage_ids = as_list(row["stage_ids"])
    allowed_candidate_ids = as_list(row["candidate_l3_ids"])

    return [
        {
            "stage": stage_context(stage_id),
            "candidates": candidate_rows_for_stage(
                stage_id,
                allowed_candidate_ids,
            ),
        }
        for stage_id in stage_ids
    ]


def merged_stages(rows):
    """Build one deduplicated Stage payload per Theme batch."""
    stage_to_allowed = {}

    for row in rows.to_dict("records"):
        allowed_candidate_ids = as_list(row["candidate_l3_ids"])
        for stage_id in as_list(row["stage_ids"]):
            stage_to_allowed.setdefault(stage_id, set()).update(
                allowed_candidate_ids
            )

    return [
        {
            **stage_context(stage_id),
            "candidate_l3_capabilities": candidate_rows_for_stage(
                stage_id,
                stage_to_allowed[stage_id],
            ),
        }
        for stage_id in sorted(stage_to_allowed)
    ]
'''


def individual_prompt_code(include_description):
    description_system = """Use Theme Description as supporting context that can clarify the Business Needs, but do not use it to introduce unsupported business functions.\n""" if include_description else ""
    description_user = '''\n    sections.extend([\n        "",\n        "Theme Description:",\n        _prompt_text(context.get("theme_description")),\n    ])\n''' if include_description else ""

    return f'''SYSTEM_PROMPT = """\\
You are performing Level 3 business capability classification.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use Theme Business Needs as the primary business evidence.
{description_system}Use the supplied Value Stream Stage data to determine the relevant business boundary.
Select every supplied candidate L3 capability whose business function is directly supported by the supplied Theme context within that Stage boundary.

EVIDENCE

Theme Business Needs describes the business outcomes, requirements, and functions that need to be delivered.
{("Theme Description provides supporting scope and intent for those Business Needs.\n" if include_description else "")}Stage data provides the governed business-process boundary for this classification.

For each candidate L3:
- capability_id is the exact identifier to return when selected.
- capability_description is the primary semantic definition of the business function.
- capability_name is a supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer business meaning from capability_id.

CLASSIFICATION

1. Identify the business functions directly supported by the Theme context.
2. Use the Stage name, description, entrance criteria, and exit criteria to constrain those functions to the supplied Stage boundary.
3. Compare that evidence against the supplied candidate L3 capability definitions.
4. Select every candidate whose business function is directly supported.

Do not select a capability merely because:
- it belongs to the supplied Stage,
- it shares terminology with the Theme context,
- it is broadly related,
- it is upstream or downstream,
- it commonly supports another capability.

Only return capability_id values supplied in Candidate L3 Capabilities.
If no candidate is supported, return an empty list.

OUTPUT

Return JSON only:

{{"l3":["CAP00000123","CAP00000456"]}}

Do not return reasons, explanations, Markdown, or additional fields.
"""


def _prompt_text(value):
    return "" if value is None else str(value).strip()


def _format_stage_block(stage, candidate_rows, label="Stage Data:"):
    lines = [
        label,
        f"  Stage ID: {{_prompt_text(stage.get('stage_id'))}}",
        f"  Stage Name: {{_prompt_text(stage.get('stage_name'))}}",
        f"  Stage Description: {{_prompt_text(stage.get('stage_description'))}}",
        f"  Entrance Criteria: {{_prompt_text(stage.get('entrance_criteria'))}}",
        f"  Exit Criteria: {{_prompt_text(stage.get('exit_criteria'))}}",
        "",
        "Candidate L3 Capabilities:",
    ]

    if not candidate_rows:
        lines.append("  None")
        return "\\n".join(lines)

    for index, capability in enumerate(candidate_rows, start=1):
        lines.extend(
            [
                f"  {{index}}.",
                f"    Capability ID: {{_prompt_text(capability.get('capability_id'))}}",
                f"    Capability Name: {{_prompt_text(capability.get('capability_name'))}}",
                f"    Capability Description: {{_prompt_text(capability.get('capability_description'))}}",
                f"    Capability Tier: {{_prompt_text(capability.get('capability_tier'))}}",
            ]
        )

    return "\\n".join(lines)


def build_user_prompt(context, stage, candidate_rows):
    sections = [
        "Theme Business Needs:",
        _prompt_text(context.get("theme_business_needs")),
    ]
{description_user}
    sections.extend(
        [
            "",
            _format_stage_block(stage, candidate_rows),
        ]
    )

    return "\\n".join(sections).strip()
'''


def batch_prompt_code(include_description):
    description_system = """Use Theme Description as supporting context that can clarify the Business Needs, but do not use it to introduce unsupported business functions.\n""" if include_description else ""
    description_user = '''\n    sections.extend([\n        "",\n        "Theme Description:",\n        _prompt_text(context.get("theme_description")),\n    ])\n''' if include_description else ""

    return f'''SYSTEM_PROMPT = """\\
You are performing Level 3 business capability classification for multiple Value Stream Stages that share the same Theme context.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use Theme Business Needs as the primary shared business evidence.
{description_system}Classify each supplied Value Stream Stage independently using only the shared Theme context, that Stage's governed metadata, and that Stage's candidate L3 capabilities.

EVIDENCE

Theme Business Needs describes the shared business outcomes, requirements, and functions that need to be delivered.
{("Theme Description provides supporting scope and intent for those Business Needs.\n" if include_description else "")}Each Stage's name, description, entrance criteria, and exit criteria define that Stage's business-process boundary.

For each candidate L3:
- capability_id is the exact identifier to return when selected.
- capability_description is the primary semantic definition of the business function.
- capability_name is a supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer business meaning from capability_id.

CLASSIFICATION

For each Stage independently:
1. Identify the business functions directly supported by the shared Theme context.
2. Constrain those functions using that Stage's name, description, entrance criteria, and exit criteria.
3. Compare the resulting evidence only against that Stage's supplied candidate L3 capability definitions.
4. Select every candidate whose business function is directly supported.

Do not use one Stage's metadata or candidates as evidence for another Stage.
Do not select a capability merely because it belongs to the Stage, shares terminology, is broadly related, is upstream or downstream, or commonly supports another capability.
Only return capability_id values supplied for that Stage.
If no candidate is supported for a Stage, return an empty list.
Return exactly one result for every supplied stage_id.

OUTPUT

Return JSON only:

{{
  "stages": [
    {{
      "stage_id": "VSS000123",
      "l3": ["CAP00000123", "CAP00000456"]
    }},
    {{
      "stage_id": "VSS000456",
      "l3": []
    }}
  ]
}}

Do not return reasons, explanations, Markdown, or additional fields.
"""


def _prompt_text(value):
    return "" if value is None else str(value).strip()


def _format_stage_block(stage, candidate_rows, index):
    lines = [
        f"Stage {{index}}:",
        f"  Stage ID: {{_prompt_text(stage.get('stage_id'))}}",
        f"  Stage Name: {{_prompt_text(stage.get('stage_name'))}}",
        f"  Stage Description: {{_prompt_text(stage.get('stage_description'))}}",
        f"  Entrance Criteria: {{_prompt_text(stage.get('entrance_criteria'))}}",
        f"  Exit Criteria: {{_prompt_text(stage.get('exit_criteria'))}}",
        "",
        "  Candidate L3 Capabilities:",
    ]

    if not candidate_rows:
        lines.append("    None")
        return "\\n".join(lines)

    for capability_index, capability in enumerate(candidate_rows, start=1):
        lines.extend(
            [
                f"    {{capability_index}}.",
                f"      Capability ID: {{_prompt_text(capability.get('capability_id'))}}",
                f"      Capability Name: {{_prompt_text(capability.get('capability_name'))}}",
                f"      Capability Description: {{_prompt_text(capability.get('capability_description'))}}",
                f"      Capability Tier: {{_prompt_text(capability.get('capability_tier'))}}",
            ]
        )

    return "\\n".join(lines)


def build_user_prompt(context, stages):
    sections = [
        "Theme Business Needs:",
        _prompt_text(context.get("theme_business_needs")),
    ]
{description_user}
    sections.extend(["", "Value Stream Stages:"])

    for index, stage_payload in enumerate(stages, start=1):
        stage = {{
            key: value
            for key, value in stage_payload.items()
            if key != "candidate_l3_capabilities"
        }}
        candidate_rows = stage_payload.get("candidate_l3_capabilities", [])
        sections.extend(
            [
                "",
                _format_stage_block(stage, candidate_rows, index),
            ]
        )

    return "\\n".join(sections).strip()
'''


for number, path in FILES.items():
    nb = json.loads(path.read_text(encoding="utf-8"))

    stage_cell = find_code_cell(nb, "def row_stages")
    set_source(stage_cell, STAGE_PAYLOAD_CODE)

    prompt_cell = find_code_cell(nb, "def build_user_prompt")
    include_description = number in (17, 19)
    is_batch = number in (19, 20)
    set_source(
        prompt_cell,
        batch_prompt_code(include_description)
        if is_batch
        else individual_prompt_code(include_description),
    )

    preview_cell = find_code_cell(nb, "preview_user_prompt")
    preview_source = source(preview_cell)
    preview_source = preview_source.replace(
        "FORMATTED USER PROMPT",
        "USER PROMPT — EXACT TEXT SENT TO LLM",
    )
    preview_source = preview_source.replace(
        "SYSTEM PROMPT",
        "SYSTEM PROMPT — EXACT TEXT SENT TO LLM",
    )
    if "USER PROMPT — EXACT TEXT SENT TO LLM" not in preview_source:
        preview_source = preview_source.replace(
            "USER PROMPT",
            "USER PROMPT — EXACT TEXT SENT TO LLM",
        )
    set_source(preview_cell, preview_source)

    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

print("Updated E17-E20 with governed enrichment and exact-text prompts")
