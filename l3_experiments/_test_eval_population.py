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


def main():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        all_code = "\n".join(
            source(cell)
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        assert "end_to_end_labeled" not in all_code, path.name
        assert "selector_only_fully_retrievable" not in all_code, path.name
        assert "selector_eligible" not in all_code, path.name
        assert "selector_exclusion_reason" not in all_code, path.name
        assert "evaluation_eligible" in all_code, path.name
        assert "evaluation_exclusion_reason" in all_code, path.name
        assert "valid_evaluation_population" in all_code, path.name
        assert "gt_not_fully_retrievable" in all_code, path.name
        assert "truth_ids.issubset(available_candidate_ids)" in all_code, path.name

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile(source(cell), f"{path.name}:cell-{index}", "exec")

    print("Valid-evaluation population checks passed for all five notebooks.")


if __name__ == "__main__":
    main()
