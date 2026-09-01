import json
from pathlib import Path

FILES = {
    17: Path("l3_experiments/17_theme_needs_description_stage_individual.ipynb"),
    18: Path("l3_experiments/18_business_needs_stage_individual.ipynb"),
    19: Path("l3_experiments/19_theme_needs_description_stage_batch.ipynb"),
    20: Path("l3_experiments/20_business_needs_stage_batch.ipynb"),
}

STAGE = {
    "stage_id": "VSS000123",
    "stage_name": "Generate Quote",
    "stage_description": "Generate a customer quote.",
    "entrance_criteria": "A quote request exists.",
    "exit_criteria": "A completed quote is available.",
}
CANDIDATES = [
    {
        "capability_id": "CAP000111",
        "capability_name": "Quote Management",
        "capability_description": "Manage creation and maintenance of quotes.",
        "capability_tier": "L3",
    }
]
CTX = {
    "theme_business_needs": "Improve quote generation for customers.",
    "theme_description": "Modernize the quote experience.",
}


def source(cell):
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


for number, path in FILES.items():
    nb = json.loads(path.read_text(encoding="utf-8"))
    all_source = "\n".join(source(c) for c in nb["cells"])

    assert "golden_valid_set.parquet" in all_source, (number, "population source")
    assert "epic_l3_ground_truth_full_golden.xlsx" in all_source, (number, "GT source")
    assert "VSSrv.csv" in all_source, (number, "Stage enrichment source")
    assert "VSSCaprv (1).csv" in all_source, (number, "L3 enrichment source")
    assert "USER PROMPT — EXACT TEXT SENT TO LLM" in all_source, (number, "preview label")
    assert "SYSTEM PROMPT — EXACT TEXT SENT TO LLM" in all_source, (number, "system preview label")

    prompt_cells = [
        source(c)
        for c in nb["cells"]
        if c.get("cell_type") == "code"
        and "SYSTEM_PROMPT" in source(c)
        and "def build_user_prompt" in source(c)
    ]
    assert len(prompt_cells) == 1, (number, "prompt cell count", len(prompt_cells))
    prompt_code = prompt_cells[0]

    assert 'SYSTEM_PROMPT = """' in prompt_code or "SYSTEM_PROMPT = '''" in prompt_code, (
        number,
        "multiline system prompt",
    )
    assert "json.dumps" not in prompt_code, (number, "user prompt must be plain formatted text")
    assert "epic_key" not in prompt_code.lower(), (number, "epic_key leaked to prompt")
    assert "item_id" not in prompt_code, (number, "item_id leaked to prompt")
    assert " epic" not in prompt_code.lower(), (number, "Epic wording leaked to prompt")

    namespace = {}
    exec(prompt_code, namespace)
    system_prompt = namespace["SYSTEM_PROMPT"]
    assert system_prompt.count("\n") >= 15, (number, "system prompt formatting")

    if number in (17, 18):
        user_prompt = namespace["build_user_prompt"](CTX, STAGE, CANDIDATES)
    else:
        stage_payload = {**STAGE, "candidate_l3_capabilities": CANDIDATES}
        user_prompt = namespace["build_user_prompt"](CTX, [stage_payload])

    assert not user_prompt.lstrip().startswith("{"), (number, "user prompt is still JSON")
    for expected in (
        "Theme Business Needs:",
        "Stage ID: VSS000123",
        "Stage Name: Generate Quote",
        "Stage Description: Generate a customer quote.",
        "Entrance Criteria: A quote request exists.",
        "Exit Criteria: A completed quote is available.",
        "Capability ID: CAP000111",
        "Capability Name: Quote Management",
        "Capability Description: Manage creation and maintenance of quotes.",
        "Capability Tier: L3",
    ):
        assert expected in user_prompt, (number, expected)

    if number in (17, 19):
        assert "Theme Description:" in user_prompt, (number, "description missing")
        assert CTX["theme_description"] in user_prompt
    else:
        assert "Theme Description:" not in user_prompt, (number, "description should be excluded")

    if number in (19, 20):
        assert "Stage 1:" in user_prompt, (number, "batch stage formatting")
        assert '"stages"' in system_prompt and '"stage_id"' in system_prompt, (
            number,
            "batch output contract",
        )

    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            compile(source(cell), f"{path}:cell{index}", "exec")

print("E17-E20 exact prompt + enrichment verification passed")
