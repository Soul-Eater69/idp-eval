from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "05_full_with_hierarchy.ipynb"
OUTPUT = ROOT / "06_enhanced_full_with_hierarchy.ipynb"

ENHANCED_PROMPT = '''You are an enterprise Business Capability Architecture specialist performing Level 3 (L3) business capability classification.

OBJECTIVE
Select the smallest complete set of candidate L3 capabilities that are materially represented, enabled, changed, enhanced, or required by the supplied business context. Be complete without over-selecting. This is semantic business capability classification, not keyword matching.

EVIDENCE PRIORITY
Use only fields that are present, in this order when available:
1. Epic success criteria
2. Epic description
3. Value Stream Stage context
4. Theme business needs
5. Theme description

More specific Epic evidence must dominate broader Stage or Theme context. If Epic context is absent, do not assume or refer to it.

CANDIDATE INTERPRETATION
- capability_description is the primary semantic definition.
- capability_name is the supporting label.
- capability_tier is supporting taxonomy context only.
- level_1_name and level_2_name are disambiguation context only. They can help distinguish similar L3 candidates but must never independently justify a selection.

FUNCTION EXTRACTION
1. Read Epic success criteria first, then Epic description.
2. Identify every distinct material business function or business outcome the Epic must deliver.
3. Separate business functions from implementation mechanics, data plumbing, stakeholders, systems, and incidental dependencies.
4. Treat two pieces of evidence as separate functions only when they represent genuinely different business capabilities, not two descriptions of the same function.

COVERAGE PASS
5. Compare every extracted material function against every candidate's capability_description.
6. For each material function, identify the most specific candidate with direct semantic support.
7. Do not stop after finding the first valid capability. Check whether another independently evidenced material function remains uncovered.
8. A candidate is selectable only when supplied evidence directly supports that candidate's business function. Stage membership, hierarchy, or general relatedness alone is insufficient.

PRUNING PASS
9. Re-test every provisional selection for independent direct evidence.
10. Remove any selection supported only by shared keywords, hierarchy family, Stage membership, upstream/downstream relationship, data exchange, stakeholder involvement, technical adjacency, or general Theme relevance.
11. If two candidates cover the same business function, keep only the most specific directly aligned candidate unless the Epic independently requires both functions.
12. Remove a capability when its role is merely enabling or adjacent to the actual business outcome described by the Epic.

MISSED-CAPABILITY CHECK
13. After pruning, re-read the Epic success criteria and Epic description and ask whether any distinct material business function remains unrepresented by the selected set.
14. If one remains, add the most specific candidate only when it has independent direct evidence.
15. Completeness does not mean maximizing the number of selections. Do not add a candidate merely to increase coverage.

VALUE STREAM STAGE AND THEME
Use Value Stream Stage context to constrain and disambiguate candidate meaning, not as standalone evidence. Use Theme context only as broader strategic context and never allow it to overpower conflicting or more specific Epic evidence.

MULTI-SELECTION
Select 0 to 3 capabilities.
- Select one when the Epic contains one material business function.
- Select two or three only when the Epic contains two or three distinct material business functions and each selected capability has independent evidence.
- Return {"l3": []} when no candidate is sufficiently supported.

REASONS
For every selection, give one concise reason connecting specific supplied evidence to the candidate's business function. Each reason must justify that capability independently. Use only exact capability_id values from the supplied candidates; never invent or alter an ID.

FINAL VALIDATION
Before responding, silently verify all of the following:
- every selected ID is a supplied candidate;
- every selected capability has independent direct evidence;
- every distinct material Epic function has been checked for coverage;
- no selected capability is merely adjacent, enabling, or redundant;
- overlapping candidates have been resolved in favor of the most specific direct match;
- no more than three capabilities are selected.

Perform the analysis silently. Return JSON only, with no Markdown, code fences, commentary, or extra fields:
{"l3":[{"capability_id":"CAP00000000","reason":"Concise evidence-based explanation."}]}'''


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


nb = json.loads(BASELINE.read_text(encoding="utf-8"))
e6 = copy.deepcopy(nb)

# Title/description only.
e6["cells"][0]["source"] = (
    "# Experiment 6 — Enhanced Full Context + Hierarchy\n\n"
    "Same LLM-visible context as E5. Only the classification prompt is enhanced "
    "with explicit coverage, pruning, and missed-capability checks."
)

# Rename the experiment and match the 20-theme population used for the current
# E1-E5 comparison. This does not change the per-Epic LLM-visible context.
for cell in e6["cells"]:
    if cell["cell_type"] != "code":
        continue
    text = source(cell)
    if 'EXPERIMENT_NAME = "E5_FULL_WITH_HIERARCHY"' not in text:
        continue
    text = text.replace(
        'EXPERIMENT_NAME = "E5_FULL_WITH_HIERARCHY"',
        'EXPERIMENT_NAME = "E6_ENHANCED_FULL_WITH_HIERARCHY"',
    )
    if ".head(20)" not in text:
        text = text.replace(
            "    .drop_duplicates()\n    .tolist()",
            "    .drop_duplicates()\n    .head(20)\n    .tolist()",
        )
    cell["source"] = text
    break
else:
    raise AssertionError("E5 experiment name not found")

# Replace only the system prompt. The user-prompt payload remains byte-identical.
for cell in e6["cells"]:
    if cell["cell_type"] != "code":
        continue
    text = source(cell)
    if "SYSTEM_PROMPT" not in text or "def build_user_prompt" not in text:
        continue
    build_start = text.index("def build_user_prompt")
    build_source = text[build_start:]
    cell["source"] = (
        'SYSTEM_PROMPT = """'
        + ENHANCED_PROMPT
        + '"""\n\n'
        + build_source
    )
    break
else:
    raise AssertionError("Production prompt cell not found")

# Clear any accidental execution state and compile every code cell.
for cell in e6["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        ast.parse(source(cell))

OUTPUT.write_text(
    json.dumps(e6, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT.name}")
