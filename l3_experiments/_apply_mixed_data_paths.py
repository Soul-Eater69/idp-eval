import json
from pathlib import Path

EXPERIMENTS = [
    Path("l3_experiments/12_business_needs_stage_custom_prompt.ipynb"),
    Path("l3_experiments/13_theme_description_stage_custom_prompt.ipynb"),
    Path("l3_experiments/14_theme_needs_description_stage_custom_prompt.ipynb"),
]

OLD_BLOCK = '''NOTEBOOK_DIR = Path.cwd()
WORKSPACE_DIR = (
    NOTEBOOK_DIR.parent
    if (NOTEBOOK_DIR.parent / "full_golden.parquet").exists()
    else NOTEBOOK_DIR
)
DATA_DIR = Path(os.getenv("L3_EXPERIMENT_DATA_DIR", WORKSPACE_DIR))

PARQUET_PATH = Path(
    os.getenv("L3_FULL_GOLDEN_PATH", DATA_DIR / "full_golden.parquet")
)
STAGE_PATH = Path(os.getenv("L3_STAGE_PATH", DATA_DIR / "VSSrv.csv"))
STAGE_CAPABILITY_MAP_PATH = Path(
    os.getenv(
        "L3_STAGE_CAPABILITY_MAP_PATH",
        DATA_DIR / "VSSCaprv (1).csv",
    )
)
GROUND_TRUTH_PATH = Path(
    os.getenv(
        "L3_GROUND_TRUTH_PATH",
        DATA_DIR / "results" / "epic_l3_ground_truth_full_golden.xlsx",
    )
)
'''

NEW_BLOCK = r'''NOTEBOOK_DIR = Path.cwd()


def resolve_data_path(relative_path: str, env_var: str) -> Path:
    """Resolve a data file from an override, notebook folder, or repo root."""
    override = os.getenv(env_var)
    if override:
        return Path(override).expanduser()

    relative_path = Path(relative_path)
    search_roots = [
        NOTEBOOK_DIR,
        NOTEBOOK_DIR.parent,
        NOTEBOOK_DIR / "l3_experiments",
    ]
    seen = set()
    attempted = []

    for root in search_roots:
        candidate = root / relative_path
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        attempted.append(candidate)
        if candidate.exists():
            return candidate

    tried = "\n  - ".join(str(path) for path in attempted)
    raise FileNotFoundError(
        f"Could not find {relative_path}. Tried:\n  - {tried}"
    )


PARQUET_PATH = resolve_data_path("full_golden.parquet", "L3_FULL_GOLDEN_PATH")
STAGE_PATH = resolve_data_path("VSSrv.csv", "L3_STAGE_PATH")
STAGE_CAPABILITY_MAP_PATH = resolve_data_path(
    "VSSCaprv (1).csv",
    "L3_STAGE_CAPABILITY_MAP_PATH",
)
GROUND_TRUTH_PATH = resolve_data_path(
    "results/epic_l3_ground_truth_full_golden.xlsx",
    "L3_GROUND_TRUTH_PATH",
)
'''


def update_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if OLD_BLOCK in source:
            cell["source"] = source.replace(OLD_BLOCK, NEW_BLOCK, 1)
            changed = True
            break

    if not changed:
        raise RuntimeError(f"Expected path configuration block not found in {path}")

    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") == "code":
            compile(cell.get("source", ""), f"{path.name}:cell_{index}", "exec")

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {path}")


for experiment in EXPERIMENTS:
    update_notebook(experiment)

# Keep this updater temporary; this comment only retriggers CI after test correction.
