from __future__ import annotations

import json
from pathlib import Path

E21 = Path("l3_experiments/21_theme_needs_description_stage_batch_recall_tuned.ipynb")
E22 = Path("l3_experiments/22_theme_needs_description_stage_batch_candidate_relevance.ipynb")


def source(cell):
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def check_notebook(path: Path, required: list[str]) -> list[str]:
    errors = []
    if not path.exists():
        return [f"missing notebook: {path}"]

    nb = json.loads(path.read_text(encoding="utf-8"))
    combined = "\n".join(source(cell) for cell in nb["cells"])

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            try:
                compile(source(cell), f"{path}:cell-{i}", "exec")
            except SyntaxError as exc:
                errors.append(f"{path}: code cell {i} does not compile: {exc}")

    for marker in required:
        if marker not in combined:
            errors.append(f"{path}: missing required marker {marker!r}")

    # Prompt input architecture must remain E19-style.
    prompt_cells = [
        source(cell)
        for cell in nb["cells"]
        if cell.get("cell_type") == "code" and "SYSTEM_PROMPT =" in source(cell)
    ]
    if len(prompt_cells) != 1:
        errors.append(f"{path}: expected exactly one production prompt cell")
    else:
        prompt = prompt_cells[0]
        for marker in (
            "Theme Business Needs:",
            "Theme Description:",
            "Value Stream Stages:",
            "Stage Name:",
            "Stage Description:",
            "Entrance Criteria:",
            "Exit Criteria:",
            "Candidate L3 Capabilities:",
            "Capability ID:",
            "Capability Name:",
            "Capability Description:",
            "Capability Tier:",
        ):
            if marker not in prompt:
                errors.append(f"{path}: prompt architecture missing {marker!r}")
        if "record_key" in prompt or "item_id" in prompt:
            errors.append(f"{path}: record/item identifier leaked into prompt cell")

    return errors


def main():
    errors = []

    errors += check_notebook(
        E21,
        [
            "SAMPLE_SIZE=100",
            "SAMPLE_SEED=42",
            "E21_THEME_NEEDS_DESCRIPTION_STAGE_BATCH_RECALL_TUNED",
            "An L3 capability is a Level 3 business capability",
            "Evaluate EVERY supplied candidate L3 independently",
            "strongly semantically implied",
            "Selecting one capability does not exclude another capability",
            "perform an omission check",
            "Prefer complete coverage of genuinely supported business functions",
            "reasoning_effort='low'",
            "id='9zdn8n'",
            "fixed_100_from_golden_valid_set_seed_42_theme_batch",
            "prompt_log",
            "prompt_sample",
        ],
    )

    errors += check_notebook(
        E22,
        [
            "SAMPLE_SIZE=100",
            "SAMPLE_SEED=42",
            "E22_THEME_NEEDS_DESCRIPTION_STAGE_BATCH_CANDIDATE_RELEVANCE",
            "An L3 capability is a Level 3 business capability",
            "independent relevance claim",
            "strongly semantically implied",
            "You MUST evaluate every candidate exactly once",
            '"relevant": true',
            '"relevant": false',
            "set(candidate) != {'capability_id','relevant'}",
            "isinstance(candidate['relevant'], bool)",
            "candidate_relevance",
            '"candidate_relevance": candidate_relevance',
            "reasoning_effort='low'",
            "id='9zdn8n'",
            "fixed_100_from_golden_valid_set_seed_42_theme_batch_candidate_relevance",
            "prompt_log",
            "prompt_sample",
        ],
    )

    if errors:
        print("VERIFICATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("VERIFICATION PASSED: E21 and E22 compile and satisfy the approved experiment design.")


if __name__ == "__main__":
    main()
