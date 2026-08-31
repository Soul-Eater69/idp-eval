import json
from pathlib import Path

NOTEBOOK_PATH = Path("l3_experiments/00_fetch_full_golden_ground_truth.ipynb")
MARKER = "## Validate GT against Stage candidate sets"

notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

if not any(
    cell.get("cell_type") == "markdown" and MARKER in cell.get("source", "")
    for cell in notebook["cells"]
):
    notebook["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "## Validate GT against Stage candidate sets\n\n"
                "An Epic is **VALID** only when every Jira GT L3 capability is present "
                "in the union of candidate L3 capabilities mapped to that Epic's Value "
                "Stream Stage(s). `full_golden.parquet` supplies the aligned `epic_keys` "
                "and `epic_vss`; `VSSCaprv (1).csv` supplies Stage → L3 candidates."
            ),
        }
    )

    validity_code = '''def find_vsscap_file() -> Path:
    """Find the Stage-to-L3 capability mapping export."""
    filename = "VSSCaprv (1).csv"
    for directory in (Path.cwd(), Path.cwd().parent):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} from {Path.cwd()}")


def parse_aligned_values(value) -> list:
    """Return a parquet list-like cell as a Python list without changing order."""
    if value is None:
        return []

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return list(value)

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]

    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def extract_vss_ids(value) -> list[str]:
    """Extract all VSS IDs from one Epic's parquet VSS value."""
    if value is None:
        return []

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        ids = []
        for item in value:
            ids.extend(extract_vss_ids(item))
        return list(dict.fromkeys(ids))

    return list(
        dict.fromkeys(
            match.upper()
            for match in re.findall(r"\\bVSS\\d+\\b", str(value), flags=re.IGNORECASE)
        )
    )


if "epic_vss" not in df.columns:
    raise KeyError("full_golden.parquet does not contain an 'epic_vss' column.")

# epic_keys and epic_vss are aligned by position inside each parquet row.
epic_stage_ids: dict[str, set[str]] = {}
stage_alignment_errors: dict[str, str] = {}

for row_index, row in df.iterrows():
    epic_keys = parse_epic_keys(row["epic_keys"])
    epic_vss_values = parse_aligned_values(row["epic_vss"])

    if len(epic_keys) != len(epic_vss_values):
        for epic_key in epic_keys:
            stage_alignment_errors[epic_key] = (
                f"row {row_index}: epic_keys={len(epic_keys)} "
                f"but epic_vss={len(epic_vss_values)}"
            )
        continue

    for epic_key, epic_vss_value in zip(epic_keys, epic_vss_values):
        epic_stage_ids.setdefault(epic_key, set()).update(
            extract_vss_ids(epic_vss_value)
        )


VSSCAP_PATH = find_vsscap_file()
vsscap = pd.read_csv(
    VSSCAP_PATH,
    dtype=str,
    encoding="cp1252",
    encoding_errors="replace",
)

required_columns = {"Value Stream Stage ID", "Capability ID"}
missing_columns = required_columns.difference(vsscap.columns)
if missing_columns:
    raise KeyError(
        f"{VSSCAP_PATH.name} is missing columns: {sorted(missing_columns)}"
    )

vsscap = vsscap.copy()
vsscap["Value Stream Stage ID"] = (
    vsscap["Value Stream Stage ID"].fillna("").astype(str).str.strip().str.upper()
)
vsscap["Capability ID"] = (
    vsscap["Capability ID"].fillna("").astype(str).str.strip()
)
vsscap = vsscap.loc[
    vsscap["Value Stream Stage ID"].ne("")
    & vsscap["Capability ID"].ne("")
]

candidate_ids_by_stage = {
    stage_id: set(group["Capability ID"])
    for stage_id, group in vsscap.groupby("Value Stream Stage ID", sort=False)
}

gt_ok = gt_l3.loc[gt_l3["status"] == "ok"].copy()
gt_ok["l3_capability_id"] = (
    gt_ok["l3_capability_id"].fillna("").astype(str).str.strip()
)
gt_ok = gt_ok.loc[gt_ok["l3_capability_id"].ne("")]

gt_ids_by_epic = {
    epic_key: set(group["l3_capability_id"])
    for epic_key, group in gt_ok.groupby("epic_key", sort=False)
}

coverage_rows = []
for epic_key in all_epic_keys:
    gt_ids = gt_ids_by_epic.get(epic_key, set())
    stage_ids = epic_stage_ids.get(epic_key, set())

    candidate_ids = set()
    for stage_id in stage_ids:
        candidate_ids.update(candidate_ids_by_stage.get(stage_id, set()))

    missing_gt_ids = gt_ids - candidate_ids

    if epic_key in stage_alignment_errors:
        invalid_reason = "stage_alignment_error"
    elif not gt_ids:
        invalid_reason = "missing_ground_truth"
    elif not stage_ids:
        invalid_reason = "no_stage"
    elif not candidate_ids:
        invalid_reason = "no_candidates"
    elif missing_gt_ids:
        invalid_reason = "gt_not_fully_retrievable"
    else:
        invalid_reason = ""

    coverage_rows.append(
        {
            "epic_key": epic_key,
            "stage_ids": sorted(stage_ids),
            "gt_l3_ids": sorted(gt_ids),
            "candidate_l3_ids": sorted(candidate_ids),
            "missing_gt_l3_ids": sorted(missing_gt_ids),
            "valid": not invalid_reason,
            "invalid_reason": invalid_reason,
            "alignment_error": stage_alignment_errors.get(epic_key),
        }
    )

validity = pd.DataFrame(coverage_rows)
valid_epics = int(validity["valid"].sum())
invalid_epics = len(validity) - valid_epics

print(f"Total Epics:   {len(validity)}")
print(f"Valid Epics:   {valid_epics}")
print(f"Invalid Epics: {invalid_epics}")

print("\\nInvalid breakdown:")
if invalid_epics:
    display(
        validity.loc[~validity["valid"], "invalid_reason"]
        .value_counts()
        .rename_axis("invalid_reason")
        .reset_index(name="epic_count")
    )
else:
    print("None")

print("\\nInvalid Epic details:")
display(validity.loc[~validity["valid"]].reset_index(drop=True))
'''

    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": validity_code,
        }
    )

NOTEBOOK_PATH.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

# Structural verification before the workflow commits the notebook.
for index, cell in enumerate(notebook["cells"], start=1):
    if cell.get("cell_type") == "code":
        compile(cell.get("source", ""), f"cell_{index}", "exec")

print(f"Updated {NOTEBOOK_PATH}")
