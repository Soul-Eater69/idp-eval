from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    ROOT / "01_theme_stage.ipynb",
    ROOT / "02_full_context.ipynb",
    ROOT / "03_no_theme_description.ipynb",
    ROOT / "04_no_theme.ipynb",
    ROOT / "05_full_with_hierarchy.ipynb",
]


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def all_code(notebook):
    return "\n".join(
        source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def main():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = all_code(notebook)

        assert "def preflight_population():" in code, path.name
        assert "ground_truth_by_epic()" in code, path.name
        assert "GT CHECK" in code, path.name
        assert "GT L3s:" in code, path.name
        assert "Candidate L3s:" in code, path.name
        assert "GT found in candidates:" in code, path.name
        assert "GT missing from candidates:" in code, path.name
        assert "STATUS: VALID" in code, path.name
        assert "STATUS: INVALID" in code, path.name
        assert "PRECHECK SUMMARY" in code, path.name
        assert "VALID Epics:" in code, path.name
        assert "INVALID Epics:" in code, path.name
        assert "LLM calls will run ONLY for:" in code, path.name

        assert "def run_predictions(preflight):" in code, path.name
        assert 'preflight.loc[preflight["evaluation_eligible"]]' in code, path.name
        assert "predictions, llm_calls = run_predictions(preflight)" in code, path.name

        assert "gt_found_in_candidates" in code, path.name
        assert "gt_missing_from_candidates" in code, path.name
        assert "evaluation_eligible" in code, path.name
        assert "evaluation_exclusion_reason" in code, path.name

        # Invalid rows must be filtered before prediction; GT is validation-only,
        # never added to the model prompt.
        prompt_cells = [
            source(cell)
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "def build_user_prompt" in source(cell)
        ]
        assert len(prompt_cells) == 1, path.name
        assert "ground_truth" not in prompt_cells[0].lower(), path.name

    print("Preflight population assertions passed for all five notebooks.")


if __name__ == "__main__":
    main()
