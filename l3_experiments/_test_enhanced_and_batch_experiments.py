import json
from pathlib import Path

EXPECTED = {
    "12_business_needs_stage_enhanced_prompt.ipynb": "E12_BUSINESS_NEEDS_STAGE_ENHANCED_PROMPT",
    "14_theme_needs_description_stage_enhanced_prompt.ipynb": "E14_THEME_NEEDS_DESCRIPTION_STAGE_ENHANCED_PROMPT",
    "15_theme_needs_description_theme_batch_custom_prompt.ipynb": "E15_THEME_NEEDS_DESCRIPTION_THEME_BATCH_CUSTOM_PROMPT",
    "16_business_needs_theme_batch_custom_prompt.ipynb": "E16_BUSINESS_NEEDS_THEME_BATCH_CUSTOM_PROMPT",
}

for name, experiment in EXPECTED.items():
    path = Path("l3_experiments") / name
    assert path.exists(), f"missing experiment notebook: {name}"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n\n".join(
        cell.get("source", "")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") == "code":
            compile(cell.get("source", ""), f"{name}:cell_{index}", "exec")
    assert experiment in code
    assert "SAMPLE_SIZE = 50" in code
    assert "SAMPLE_SEED = 42" in code
    assert 'reasoning_effort="low"' in code
    assert "reason\"" not in code
    assert "Select 0 to 3" not in code

for name in [
    "15_theme_needs_description_theme_batch_custom_prompt.ipynb",
    "16_business_needs_theme_batch_custom_prompt.ipynb",
]:
    code = "\n\n".join(
        cell.get("source", "")
        for cell in json.loads((Path("l3_experiments") / name).read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "groupby(\"theme_key\"" in code
    assert "validate_theme_batch_response" in code
    assert '"epics"' in code

print("enhanced and batch experiment structure verified")
