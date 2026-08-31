import json
from pathlib import Path

BASE = Path("l3_experiments")

SECTIONS = {
    "15_theme_needs_description_theme_batch_custom_prompt.ipynb": """## What the LLM sees

### Shared Theme context

✅ Theme Business Needs  
✅ Theme Description

### Per-Epic context

✅ Epic key  
✅ Epic-specific Value Stream Stage(s):
- stage_id
- stage_name
- stage_description
- entrance_criteria
- exit_criteria

✅ Epic-specific candidate L3 capabilities:
- capability_id
- capability_name
- capability_description
- capability_tier

### Not sent to the LLM

❌ Epic description  
❌ Epic success criteria  
❌ Ground truth L3s  
❌ L1/L2 hierarchy

### Execution

One LLM call per Theme for all sampled valid Epics belonging to that Theme. Shared Theme context is sent once; each Epic carries only its own Stage(s) and candidate L3s. Each Epic is classified independently inside the same Theme-level call.
""",
    "16_business_needs_theme_batch_custom_prompt.ipynb": """## What the LLM sees

### Shared Theme context

✅ Theme Business Needs  
❌ Theme Description

### Per-Epic context

✅ Epic key  
✅ Epic-specific Value Stream Stage(s):
- stage_id
- stage_name
- stage_description
- entrance_criteria
- exit_criteria

✅ Epic-specific candidate L3 capabilities:
- capability_id
- capability_name
- capability_description
- capability_tier

### Not sent to the LLM

❌ Theme Description  
❌ Epic description  
❌ Epic success criteria  
❌ Ground truth L3s  
❌ L1/L2 hierarchy

### Execution

One LLM call per Theme for all sampled valid Epics belonging to that Theme. Shared Theme Business Needs is sent once; each Epic carries only its own Stage(s) and candidate L3s. Each Epic is classified independently inside the same Theme-level call.
""",
}

for filename, section in SECTIONS.items():
    path = BASE / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))

    target = None
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "markdown" and cell.get("source", "").startswith("## What the LLM sees"):
            target = cell
            break

    if target is None:
        raise RuntimeError(f"What the LLM sees section not found in {filename}")

    target["source"] = section

    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") == "code":
            compile(cell.get("source", ""), f"{filename}:cell_{index}", "exec")

    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

print("Expanded What the LLM sees in E15 and E16")
