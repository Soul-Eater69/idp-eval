from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "01_theme_stage.ipynb"


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


SYSTEM_AND_BUILDER = '''SYSTEM_PROMPT = """You are an enterprise Business Capability Architecture specialist performing Level 3 (L3) business capability classification.

OBJECTIVE
For each Epic in the supplied Theme batch, select only candidate L3 capabilities materially represented by that Epic's supplied Value Stream Stage context and the shared Theme context. Classify each Epic independently. This is capability classification, not keyword matching.

EVIDENCE PRIORITY
Use only fields that are present, in this order:
1. Value Stream Stage context for that Epic
2. Theme business needs
3. Theme description

Epic description and Epic success criteria are not supplied. Do not assume them.

CANDIDATE INTERPRETATION
- capability_description is the primary semantic definition.
- capability_name is the supporting label.
- capability_tier is supporting taxonomy context only.
- level_1_name and level_2_name, when supplied, are for disambiguation only and must never independently justify a selection.

DECISION PROCEDURE
1. For each Epic, determine the business function or outcome supported by its supplied Stage context and the shared Theme context.
2. Compare that evidence semantically against that Epic's candidates.
3. Select a candidate only when the supplied evidence materially supports its business function; relatedness alone is insufficient.
4. Stage membership alone is not enough; use Stage meaning together with Theme context.
5. When candidates overlap, prefer the most specific directly aligned capability.

EPIC ISOLATION
- Use only candidates listed under the Epic being classified.
- Do not use another Epic's Stage or candidates as evidence for the current Epic.
- Shared Theme membership does not mean different Epics should receive the same L3 selection.

DO NOT SELECT
Do not select a capability merely because of shared keywords, hierarchy family, Stage membership, upstream/downstream relationship, data exchange, stakeholder involvement, technical adjacency, or general Theme relevance.

MULTI-SELECTION
Select 0 to 3 capabilities per Epic. Default to one when one capability adequately represents the function. Select multiple only for distinct material business functions with independent evidence. Return an empty l3 list when none is sufficiently supported.

REASONS
For every selection, give a concise reason connecting supplied evidence to the candidate definition. Use only exact capability_id values from that Epic's supplied candidates; never invent or alter an ID.

FINAL VALIDATION
Before responding, verify that every supplied epic_key appears exactly once, every selected ID belongs to that Epic's candidates, every selection has direct evidence, no selection is merely adjacent, and no more than three capabilities are selected per Epic.

OUTPUT CONTRACT
Return JSON only, with no Markdown, code fences, commentary, or extra fields:
{"epics":[{"epic_key":"GROUP-00000","l3":[{"capability_id":"CAP00000000","reason":"Concise evidence-based explanation."}]}]}"""


def build_theme_batch_prompt(theme, epic_payloads):
    payload = {
        "task": "Classify every supplied Epic independently and select 0 to 3 L3 capabilities for each.",
        "theme": {
            "business_needs": theme["theme_business_needs"],
            "description": theme["theme_description"],
        },
        "epics": epic_payloads,
        "selection_instruction": "Return one result for every epic_key. Use only candidates listed under that Epic.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
'''


PREDICTION_CODE = '''def build_batch_epics(theme_rows):
    epic_payloads = []
    allowed_ids_by_epic = {}

    for row in theme_rows.to_dict(orient="records"):
        epic_key = row["epic_key"]
        stages = []
        allowed_ids = set()

        for stage_id in json.loads(row["stage_ids"]):
            candidates = candidate_rows_for_stage(stage_id)
            stages.append({
                "value_stream_stage": stage_context(stage_id),
                "candidate_l3_capabilities": candidates,
            })
            allowed_ids.update(
                candidate["capability_id"] for candidate in candidates
            )

        epic_payloads.append({
            "epic_key": epic_key,
            "stages": stages,
        })
        allowed_ids_by_epic[epic_key] = sorted(allowed_ids)

    return epic_payloads, allowed_ids_by_epic


def validate_theme_batch_response(
    payload,
    expected_epic_keys,
    allowed_ids_by_epic,
):
    if not isinstance(payload, dict) or set(payload) != {"epics"}:
        raise ValueError("Theme-batch response must contain only the 'epics' field")
    if not isinstance(payload["epics"], list):
        raise ValueError("Theme-batch 'epics' must be a list")

    expected_epic_keys = list(expected_epic_keys)
    expected_set = set(expected_epic_keys)
    seen = {}

    for item in payload["epics"]:
        if not isinstance(item, dict):
            raise ValueError("Each Epic result must be an object")
        epic_key = str(item.get("epic_key", "")).strip()
        if epic_key not in expected_set:
            raise ValueError(f"Unexpected epic_key in response: {epic_key}")
        if epic_key in seen:
            raise ValueError(f"Duplicate epic_key in response: {epic_key}")

        selections = validate_l3_response(
            {"l3": item.get("l3", [])},
            allowed_ids_by_epic[epic_key],
            allow_empty=True,
            max_selected=3,
        )
        seen[epic_key] = selections

    missing = [key for key in expected_epic_keys if key not in seen]
    if missing:
        raise ValueError(f"Missing Epic results: {missing}")

    return seen


def predict_theme_batch(
    gateway,
    theme,
    epic_payloads,
    allowed_ids_by_epic,
):
    user_prompt = build_theme_batch_prompt(theme, epic_payloads)
    raw_response, metrics = call_llm_with_metrics(
        gateway,
        SYSTEM_PROMPT,
        user_prompt,
    )
    expected_epic_keys = [item["epic_key"] for item in epic_payloads]
    predictions = validate_theme_batch_response(
        parse_json_response(raw_response),
        expected_epic_keys,
        allowed_ids_by_epic,
    )
    return {
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "predictions": predictions,
        "metrics": metrics,
    }


def metric_text(value):
    return "n/a" if value is None else str(value)


def summarize_llm_calls(call_metrics):
    successful = call_metrics.loc[call_metrics["status"] == "ok"].copy()

    def numeric(column):
        return pd.to_numeric(successful[column], errors="coerce").dropna()

    latency = numeric("latency_seconds")
    input_tokens = numeric("input_tokens")
    output_tokens = numeric("output_tokens")
    total_tokens = numeric("total_tokens")
    epics_in_call = numeric("epics_in_call")
    total_epics_in_calls = int(epics_in_call.sum()) if len(epics_in_call) else 0
    total_token_count = int(total_tokens.sum()) if len(total_tokens) else None

    return pd.DataFrame([{
        "successful_calls": len(successful),
        "failed_calls": int((call_metrics["status"] == "error").sum()),
        "usage_reported_calls": len(total_tokens),
        "total_epics_in_calls": total_epics_in_calls,
        "avg_epics_per_call": float(epics_in_call.mean()) if len(epics_in_call) else None,
        "avg_latency_seconds": float(latency.mean()) if len(latency) else None,
        "p50_latency_seconds": float(latency.quantile(0.50)) if len(latency) else None,
        "p95_latency_seconds": float(latency.quantile(0.95)) if len(latency) else None,
        "avg_input_tokens": float(input_tokens.mean()) if len(input_tokens) else None,
        "avg_output_tokens": float(output_tokens.mean()) if len(output_tokens) else None,
        "avg_total_tokens": float(total_tokens.mean()) if len(total_tokens) else None,
        "total_input_tokens": int(input_tokens.sum()) if len(input_tokens) else None,
        "total_output_tokens": int(output_tokens.sum()) if len(output_tokens) else None,
        "total_tokens": total_token_count,
        "tokens_per_epic": (
            float(total_token_count / total_epics_in_calls)
            if total_token_count is not None and total_epics_in_calls
            else None
        ),
    }])


def run_predictions(preflight):
    eligible_rows = preflight.loc[preflight["evaluation_eligible"]].copy()
    prediction_rows = []
    call_rows = []

    call_columns = [
        "experiment",
        "theme_id",
        "epics_in_call",
        "stage_count",
        "candidate_instances",
        "status",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "selected_count",
        "error",
    ]

    theme_groups = list(eligible_rows.groupby("theme_id", sort=False))
    print(
        f"\\nRunning {EXPERIMENT_NAME}: "
        f"{len(eligible_rows)} valid Epics in {len(theme_groups)} Theme-level calls"
    )

    if eligible_rows.empty:
        return pd.DataFrame(), pd.DataFrame(columns=call_columns)

    gateway = load_gateway()

    for theme_index, (theme_id, theme_rows) in enumerate(theme_groups, start=1):
        theme = themes[theme_id]
        epic_payloads, allowed_ids_by_epic = build_batch_epics(theme_rows)
        stage_count = sum(len(item["stages"]) for item in epic_payloads)
        candidate_instances = sum(
            len(stage["candidate_l3_capabilities"])
            for item in epic_payloads
            for stage in item["stages"]
        )
        started = perf_counter()

        print(
            f"\\n[THEME LLM {theme_index}/{len(theme_groups)}] {theme_id}"
            f" | epics={len(epic_payloads)}"
            f" | stages={stage_count}"
            f" | candidate_instances={candidate_instances}"
        )

        try:
            result = predict_theme_batch(
                gateway,
                theme,
                epic_payloads,
                allowed_ids_by_epic,
            )
            metrics = result["metrics"]
            predictions_by_epic = result["predictions"]
            selected_count = sum(
                len(selections) for selections in predictions_by_epic.values()
            )

            print(
                f"  OK"
                f" | latency={metrics['latency_seconds']:.3f}s"
                f" | input_tokens={metric_text(metrics['input_tokens'])}"
                f" | output_tokens={metric_text(metrics['output_tokens'])}"
                f" | total_tokens={metric_text(metrics['total_tokens'])}"
                f" | selected={selected_count}"
            )

            call_rows.append({
                "experiment": EXPERIMENT_NAME,
                "theme_id": theme_id,
                "epics_in_call": len(epic_payloads),
                "stage_count": stage_count,
                "candidate_instances": candidate_instances,
                "status": "ok",
                "latency_seconds": metrics["latency_seconds"],
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "total_tokens": metrics["total_tokens"],
                "selected_count": selected_count,
                "error": None,
            })

            for row in theme_rows.to_dict(orient="records"):
                epic_key = row["epic_key"]
                selections = predictions_by_epic[epic_key]
                predicted_ids = sorted({
                    selection["capability_id"] for selection in selections
                })
                prediction_rows.append({
                    "experiment": EXPERIMENT_NAME,
                    "theme_id": theme_id,
                    "epic_key": epic_key,
                    "stage_ids": row["stage_ids"],
                    "ground_truth_l3_ids": row["ground_truth_l3_ids"],
                    "available_candidate_l3_ids": row["available_candidate_l3_ids"],
                    "gt_found_in_candidates": row["gt_found_in_candidates"],
                    "gt_missing_from_candidates": row["gt_missing_from_candidates"],
                    "predicted_l3_ids": json.dumps(predicted_ids),
                    "model_reasons": json.dumps(selections, ensure_ascii=False),
                    "status": "ok",
                    "error": None,
                })
        except Exception as exc:
            latency = perf_counter() - started
            error = str(exc)
            print(f"  ERROR | latency={latency:.3f}s | {error}")
            call_rows.append({
                "experiment": EXPERIMENT_NAME,
                "theme_id": theme_id,
                "epics_in_call": len(epic_payloads),
                "stage_count": stage_count,
                "candidate_instances": candidate_instances,
                "status": "error",
                "latency_seconds": latency,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "selected_count": None,
                "error": error,
            })
            for row in theme_rows.to_dict(orient="records"):
                prediction_rows.append({
                    "experiment": EXPERIMENT_NAME,
                    "theme_id": theme_id,
                    "epic_key": row["epic_key"],
                    "stage_ids": row["stage_ids"],
                    "ground_truth_l3_ids": row["ground_truth_l3_ids"],
                    "available_candidate_l3_ids": row["available_candidate_l3_ids"],
                    "gt_found_in_candidates": row["gt_found_in_candidates"],
                    "gt_missing_from_candidates": row["gt_missing_from_candidates"],
                    "predicted_l3_ids": json.dumps([]),
                    "model_reasons": json.dumps([]),
                    "status": "error",
                    "error": error,
                })

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(call_rows, columns=call_columns),
    )
'''


HIERARCHY_CANDIDATES = '''def candidate_rows_for_stage(stage_id):
    rows = stage_capability_map.loc[
        stage_capability_map["Value Stream Stage ID"].astype(str).str.strip()
        == stage_id
    ].copy()
    rows = (
        rows.drop_duplicates(subset=["Capability ID"], keep="first")
        .sort_values(["Capability Name", "Capability ID"], kind="stable")
    )

    return [
        {
            "capability_id": clean_text(row["Capability ID"]),
            "capability_name": clean_text(row["Capability Name"]),
            "capability_description": clean_text(row["Capability Description"]),
            "capability_tier": clean_text(row["Capability Tier"]),
            "level_1_name": clean_text(row["Level 1 Name"]),
            "level_2_name": clean_text(row["Level 2 Name"]),
        }
        for _, row in rows.iterrows()
    ]
'''


def build_notebook(output_name, experiment_name, title, hierarchy):
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    nb = copy.deepcopy(base)

    nb["cells"][0]["source"] = title

    hierarchy_lines = (
        "\n  ├─ level_1_name\n  └─ level_2_name"
        if hierarchy
        else ""
    )
    hierarchy_note = "" if hierarchy else "L1/L2 hierarchy, "
    nb["cells"][1]["source"] = f'''## What the LLM sees

One LLM call is made per Theme for all preflight-valid Epics under that Theme.

```text
task
theme
  ├─ business_needs
  └─ description
epics[]
  ├─ epic_key
  └─ stages[]
      ├─ value_stream_stage
      │   ├─ stage_id
      │   ├─ stage_name
      │   ├─ stage_description
      │   ├─ entrance_criteria
      │   └─ exit_criteria
      └─ candidate_l3_capabilities[]
          ├─ capability_id
          ├─ capability_name
          ├─ capability_description
          └─ capability_tier{hierarchy_lines}
selection_instruction
```

**Not sent to the LLM:** Epic description, Epic success criteria, {hierarchy_note}ground truth.
'''

    config = source(nb["cells"][3])
    config = config.replace(
        "# Run every Theme in epic_gen.csv. Ground truth is not consulted here.",
        "# Match the current comparison population: first 20 Themes. Ground truth is not consulted here.",
    )
    config = config.replace(
        "    .drop_duplicates()\n    .tolist()",
        "    .drop_duplicates()\n    .head(20)\n    .tolist()",
    )
    config = config.replace(
        'EXPERIMENT_NAME = "E1_THEME_STAGE"',
        f'EXPERIMENT_NAME = "{experiment_name}"',
    )
    nb["cells"][3]["source"] = config

    if hierarchy:
        nb["cells"][7]["source"] = HIERARCHY_CANDIDATES

    nb["cells"][9]["source"] = SYSTEM_AND_BUILDER
    nb["cells"][11]["source"] = PREDICTION_CODE

    nb["cells"][14]["source"] = '''## Batch preflight, Theme-batch execution, and evaluation

Ground truth is loaded **before any LLM call** only to validate the experiment population. Only preflight-valid Epics are included in a Theme batch. GT is never included in the model prompt.
'''

    batch = source(nb["cells"][15])
    batch = batch.replace(
        '    print(f"LLM calls will run ONLY for: {valid_count} Epics")',
        '    valid_theme_count = int(preflight.loc[preflight["evaluation_eligible"], "theme_id"].nunique())\n'
        '    print(f"Valid Epics available for batching: {valid_count}")\n'
        '    print(f"Theme-level LLM calls planned: {valid_theme_count}")',
    )
    nb["cells"][15]["source"] = batch

    # Remove the E1 single-Epic inspection section; this experiment executes at Theme level.
    del nb["cells"][12:14]

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            ast.parse(source(cell))

    output = ROOT / output_name
    output.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output.name}")


build_notebook(
    "07_theme_batch.ipynb",
    "E7_THEME_BATCH",
    "# Experiment 7 — Theme Batch\n\nE1 context, but all preflight-valid Epics in a Theme are classified together in one LLM call. Epic description/success criteria and hierarchy are excluded.",
    hierarchy=False,
)
build_notebook(
    "08_theme_batch_with_hierarchy.ipynb",
    "E8_THEME_BATCH_WITH_HIERARCHY",
    "# Experiment 8 — Theme Batch + Hierarchy\n\nSame Theme-batch design as E7, with L1/L2 hierarchy added to each L3 candidate. Epic description and success criteria remain excluded.",
    hierarchy=True,
)
