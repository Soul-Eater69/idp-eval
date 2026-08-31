import json
from copy import deepcopy
from pathlib import Path

BASE = Path("l3_experiments")
E12 = BASE / "12_business_needs_stage_custom_prompt.ipynb"
E14 = BASE / "14_theme_needs_description_stage_custom_prompt.ipynb"

E12_ENHANCED_PROMPT = '''You are performing Level 3 business capability classification for one Epic.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use Theme Business Needs as the business evidence.
Use Value Stream Stage only to identify the relevant business boundary and candidate space.

IMPORTANT

A candidate must be supported by the Theme Business Needs.
Value Stream Stage membership alone must never justify a selection.

EVIDENCE

Theme Business Needs describes the business outcomes and functions the Theme is intended to address.

Value Stream Stage identifies which portion of those Business Needs is relevant to this Epic.

For each candidate L3:
- capability_id is the exact identifier to return when selected.
- capability_description is the primary definition of the business function.
- capability_name is a supporting label.
- capability_tier is taxonomy context only.

Do not infer meaning from capability_id.

CLASSIFICATION

1. Identify the parts of Theme Business Needs that fall within the supplied Value Stream Stage boundary.
2. Ignore Theme needs that belong outside that Stage.
3. Compare the remaining business evidence against each candidate's capability_description.
4. Select a candidate only when the Business Needs directly support that business function.

Do not select a capability merely because:
- it belongs to the Stage,
- it shares terminology,
- it is related to another supported capability,
- it is upstream or downstream,
- it supplies data or supporting functionality,
- it is generally relevant to the Theme.

Before returning the result, remove any capability that is supported only by Stage membership or general relatedness rather than Theme Business Needs.

Only return capability_id values from the supplied candidates.

If none are supported, return an empty list.

OUTPUT

{"l3":["CAP00000123","CAP00000456"]}

Return JSON only. Do not return reasons, explanations, Markdown, or additional fields.'''

E14_ENHANCED_PROMPT = '''You are performing Level 3 business capability classification for one Epic.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use Theme Business Needs as the primary business evidence.

Use Theme Description only to clarify the meaning, scope, or ambiguity of the Business Needs. Theme Description must not introduce a new business function that is not supported by Theme Business Needs.

Use Value Stream Stage only to constrain the relevant business boundary and candidate space.

EVIDENCE

Theme Business Needs determines what business functions may be selected.

Theme Description helps interpret those needs when their intent or scope is unclear.

Value Stream Stage determines which portion of that Theme evidence is relevant to this Epic.

For each candidate L3:
- capability_id is the exact identifier to return when selected.
- capability_description is the primary definition of the business function.
- capability_name is a supporting label.
- capability_tier is taxonomy context only.

Do not infer meaning from capability_id.

CLASSIFICATION

1. Identify the business functions expressed by Theme Business Needs.
2. Use Theme Description to clarify those functions, not broaden them.
3. Restrict the evidence to the supplied Value Stream Stage boundary.
4. Compare that evidence against each candidate's capability_description.
5. Select a candidate only when its business function is directly supported.

When multiple candidates appear related, use their capability descriptions to distinguish the actual business functions and select only those supported by the Theme evidence.

Do not select a capability merely because:
- it belongs to the Stage,
- Theme Description mentions a related area,
- it shares terminology,
- it is adjacent, upstream, or downstream,
- it enables or supports another capability.

If Theme Description suggests something not supported by Theme Business Needs, do not use it as selection evidence.

Before returning the result, remove any capability whose support comes only from Stage membership, Description-only context, or general relatedness.

Only return capability_id values from the supplied candidates.

If none are supported, return an empty list.

OUTPUT

{"l3":["CAP00000123","CAP00000456"]}

Return JSON only. Do not return reasons, explanations, Markdown, or additional fields.'''

E15_PROMPT = '''You are performing Level 3 business capability classification for multiple Epics belonging to one Theme.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

The Theme Business Needs and Theme Description are shared context for all Epics in this request.

Each Epic must be classified independently using only:
- the shared Theme Business Needs,
- the shared Theme Description,
- that Epic's own Value Stream Stage context,
- that Epic's own candidate L3 capabilities.

EVIDENCE

Theme Business Needs is the primary business evidence and describes the business outcomes and needs the Theme is intended to address.

Theme Description provides supporting context to clarify the scope and intent of those Business Needs.

Each Epic's Value Stream Stage defines the business activity boundary relevant to that Epic.

The Stage constrains the interpretation and candidate space. Stage membership alone is not evidence that a candidate capability should be selected.

For each candidate L3:
- capability_id is the exact identifier to return when selected.
- capability_description is the primary semantic definition of the business function.
- capability_name is the supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer meaning from capability_id.

CLASSIFICATION RULES

For each Epic independently:

1. Determine the business functions supported by the shared Theme Business Needs.
2. Use Theme Description only to clarify those needs and their scope.
3. Constrain that interpretation using that Epic's Value Stream Stage.
4. Compare the resulting evidence against only that Epic's candidate L3 definitions.
5. Select every candidate whose business function is directly supported.

Do not select a capability merely because:
- it belongs to the Epic's Stage,
- it shares terminology,
- it is broadly related to the Theme,
- it is adjacent, upstream, or downstream,
- it supports another capability.

EPIC ISOLATION

Do not use one Epic's Stage or candidates as evidence for another Epic.

Shared Theme context does not imply that Epics should receive identical capability selections.

Only return capability_id values supplied under that specific Epic.

If no candidate is supported for an Epic, return an empty list for that Epic.

Return exactly one result for every supplied epic_key.

OUTPUT

Return JSON only:

{"epics":[{"epic_key":"GROUP-12345","l3":["CAP00000123","CAP00000456"]},{"epic_key":"GROUP-67890","l3":[]}]}

Do not return reasons, explanations, Markdown, or additional fields.'''

E16_PROMPT = '''You are performing Level 3 business capability classification for multiple Epics belonging to one Theme.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

The Theme Business Needs is shared context for all Epics in this request.

Each Epic must be classified independently using only:
- the shared Theme Business Needs,
- that Epic's own Value Stream Stage context,
- that Epic's own candidate L3 capabilities.

EVIDENCE

Theme Business Needs is the primary business evidence and describes the business outcomes and needs the Theme is intended to address.

Each Epic's Value Stream Stage defines the business activity boundary relevant to that Epic.

The Stage constrains the interpretation and candidate space. Stage membership alone is not evidence that a candidate capability should be selected.

For each candidate L3:
- capability_id is the exact identifier to return when selected.
- capability_description is the primary semantic definition of the business function.
- capability_name is the supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer meaning from capability_id.

CLASSIFICATION RULES

For each Epic independently:

1. Determine the business functions supported by Theme Business Needs.
2. Constrain that interpretation using that Epic's Value Stream Stage.
3. Compare the evidence against only that Epic's candidate L3 definitions.
4. Select every candidate whose business function is directly supported.

Do not select a capability merely because:
- it belongs to the Epic's Stage,
- it shares terminology,
- it is broadly related to the Theme,
- it is adjacent, upstream, or downstream,
- it supports another capability.

EPIC ISOLATION

Do not use one Epic's Stage or candidates as evidence for another Epic.

Shared Theme membership does not mean different Epics should receive the same capability selections.

Only return capability_id values supplied under that specific Epic.

If no candidate is supported for an Epic, return an empty list.

Return exactly one result for every supplied epic_key.

OUTPUT

Return JSON only:

{"epics":[{"epic_key":"GROUP-12345","l3":["CAP00000123"]}]}

Do not return reasons, explanations, Markdown, or additional fields.'''


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, notebook):
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") == "code":
            compile(cell.get("source", ""), f"{path.name}:cell_{index}", "exec")
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def heading_index(notebook, heading):
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "markdown" and cell.get("source", "").strip() == heading:
            return index
    raise RuntimeError(f"Heading not found: {heading}")


def code_after(notebook, heading):
    index = heading_index(notebook, heading)
    for cell in notebook["cells"][index + 1:]:
        if cell.get("cell_type") == "code":
            return cell
    raise RuntimeError(f"Code cell not found after {heading}")


def clone_enhanced(template_path, output_path, title, experiment_name, prompt):
    notebook = deepcopy(load(template_path))
    notebook["cells"][0]["source"] = title
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["source"] = cell.get("source", "").replace(
                'EXPERIMENT_NAME = "E12_BUSINESS_NEEDS_STAGE_CUSTOM_PROMPT"',
                f'EXPERIMENT_NAME = "{experiment_name}"',
            ).replace(
                'EXPERIMENT_NAME = "E14_THEME_NEEDS_DESCRIPTION_STAGE_CUSTOM_PROMPT"',
                f'EXPERIMENT_NAME = "{experiment_name}"',
            )
    prompt_cell = code_after(notebook, "## Production prompt")
    _, tail = prompt_cell["source"].split("\ndef build_user_prompt", 1)
    prompt_cell["source"] = f"SYSTEM_PROMPT = {prompt!r}\n\ndef build_user_prompt" + tail
    save(output_path, notebook)


def shared_prefix(template):
    end = heading_index(template, "## Production prompt")
    return deepcopy(template["cells"][:end])


def batch_prediction_code(include_description):
    theme_payload = '{"business_needs": theme["theme_business_needs"], "description": theme["theme_description"]}' if include_description else '{"business_needs": theme["theme_business_needs"]}'
    return f'''def build_theme_batch_prompt(theme, epic_payloads):
    payload = {{
        "task": "Classify every supplied Epic independently using only its own Stage and candidates plus the shared Theme evidence.",
        "theme": {theme_payload},
        "epics": epic_payloads,
    }}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def validate_theme_batch_response(payload, expected_epic_keys, allowed_ids_by_epic):
    if not isinstance(payload, dict) or set(payload) != {{"epics"}}:
        raise ValueError("Theme-batch response must contain only the epics field.")
    if not isinstance(payload["epics"], list):
        raise ValueError("Theme-batch epics must be a list.")

    expected = list(expected_epic_keys)
    expected_set = set(expected)
    predictions = {{}}

    for item in payload["epics"]:
        if not isinstance(item, dict) or set(item) != {{"epic_key", "l3"}}:
            raise ValueError("Each Epic result must contain exactly epic_key and l3.")
        epic_key = str(item["epic_key"]).strip()
        if epic_key not in expected_set:
            raise ValueError(f"Unexpected epic_key: {{epic_key}}")
        if epic_key in predictions:
            raise ValueError(f"Duplicate epic_key: {{epic_key}}")
        if not isinstance(item["l3"], list):
            raise ValueError(f"l3 must be a list for {{epic_key}}")

        allowed = set(allowed_ids_by_epic[epic_key])
        selected = []
        seen = set()
        for capability_id in item["l3"]:
            if not isinstance(capability_id, str):
                raise ValueError(f"L3 IDs must be strings for {{epic_key}}")
            capability_id = capability_id.strip()
            if capability_id not in allowed:
                raise ValueError(f"{{capability_id}} is not a candidate for {{epic_key}}")
            if capability_id in seen:
                raise ValueError(f"Duplicate L3 ID {{capability_id}} for {{epic_key}}")
            seen.add(capability_id)
            selected.append(capability_id)
        predictions[epic_key] = selected

    if set(predictions) != expected_set:
        missing = sorted(expected_set - set(predictions))
        raise ValueError(f"Missing Epic results: {{missing}}")
    return predictions


def build_batch_epics(theme_rows):
    epic_payloads = []
    allowed_ids_by_epic = {{}}

    for row in theme_rows.to_dict(orient="records"):
        epic_key = str(row["epic_key"]).strip()
        stages = []
        allowed_ids = set()
        for stage_id in parse_list_value(row["stage_ids"]):
            candidates = candidate_rows_for_stage(stage_id)
            stages.append({{
                "value_stream_stage": stage_context(stage_id),
                "candidate_l3_capabilities": candidates,
            }})
            allowed_ids.update(candidate["capability_id"] for candidate in candidates)
        epic_payloads.append({{"epic_key": epic_key, "stages": stages}})
        allowed_ids_by_epic[epic_key] = sorted(allowed_ids)

    return epic_payloads, allowed_ids_by_epic


def predict_theme_batch(gateway, theme_id, theme_rows):
    theme = themes[theme_id]
    epic_payloads, allowed_ids_by_epic = build_batch_epics(theme_rows)
    user_prompt = build_theme_batch_prompt(theme, epic_payloads)
    raw_response, metrics = call_llm_with_metrics(
        gateway,
        SYSTEM_PROMPT,
        user_prompt,
        reasoning_effort="low",
    )
    predictions = validate_theme_batch_response(
        parse_json_response(raw_response),
        [item["epic_key"] for item in epic_payloads],
        allowed_ids_by_epic,
    )
    return predictions, metrics, user_prompt, raw_response
'''


def evaluation_code():
    return '''def run_experiment():
    gateway = load_gateway()
    result_rows = []
    call_rows = []

    for theme_id, theme_rows in evaluation_population.groupby("theme_key", sort=True):
        theme_rows = theme_rows.sort_values("epic_key", kind="stable").reset_index(drop=True)
        started = perf_counter()
        try:
            predictions, metrics, user_prompt, raw_response = predict_theme_batch(
                gateway,
                theme_id,
                theme_rows,
            )
            call_rows.append({
                "experiment": EXPERIMENT_NAME,
                "theme_id": theme_id,
                "epic_count": len(theme_rows),
                "status": "ok",
                "latency_seconds": metrics.get("latency_seconds"),
                "input_tokens": metrics.get("input_tokens"),
                "output_tokens": metrics.get("output_tokens"),
                "total_tokens": metrics.get("total_tokens"),
                "error": None,
            })

            for row in theme_rows.to_dict(orient="records"):
                epic_key = str(row["epic_key"]).strip()
                predicted = predictions[epic_key]
                truth = parse_list_value(row["gt_l3_ids"])
                scores = score_sets(predicted, truth)
                result_rows.append({
                    "experiment": EXPERIMENT_NAME,
                    "theme_id": theme_id,
                    "epic_key": epic_key,
                    "stage_ids": row["stage_ids"],
                    "predicted_l3_ids": predicted,
                    "gt_l3_ids": truth,
                    "status": "ok",
                    "error": None,
                    **scores,
                })
        except Exception as exc:
            call_rows.append({
                "experiment": EXPERIMENT_NAME,
                "theme_id": theme_id,
                "epic_count": len(theme_rows),
                "status": "error",
                "latency_seconds": perf_counter() - started,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": str(exc),
            })
            for row in theme_rows.to_dict(orient="records"):
                result_rows.append({
                    "experiment": EXPERIMENT_NAME,
                    "theme_id": theme_id,
                    "epic_key": str(row["epic_key"]).strip(),
                    "stage_ids": row["stage_ids"],
                    "predicted_l3_ids": None,
                    "gt_l3_ids": parse_list_value(row["gt_l3_ids"]),
                    "status": "error",
                    "error": str(exc),
                    "exact_match": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "predicted_count": None,
                    "truth_count": len(parse_list_value(row["gt_l3_ids"])),
                })

    return pd.DataFrame(result_rows), pd.DataFrame(call_rows)


results, call_metrics = run_experiment()
scored = results.loc[results["status"].eq("ok")].copy()
successful_calls = call_metrics.loc[call_metrics["status"].eq("ok")].copy()

summary = pd.DataFrame([{
    "scope": "fixed_50_valid_epics_seed_42_theme_batch",
    "evaluated_epics": len(scored),
    "exact_match_accuracy": scored["exact_match"].mean() if len(scored) else 0.0,
    "mean_precision": scored["precision"].mean() if len(scored) else 0.0,
    "mean_recall": scored["recall"].mean() if len(scored) else 0.0,
    "mean_f1": scored["f1"].mean() if len(scored) else 0.0,
}])

diagnostics = pd.DataFrame([{
    "sample_seed": SAMPLE_SEED,
    "sample_size": SAMPLE_SIZE,
    "themes_selected": evaluation_population["theme_key"].nunique(),
    "total_epics": len(evaluation_population),
    "successful_calls": int(call_metrics["status"].eq("ok").sum()),
    "failed_calls": int(call_metrics["status"].eq("error").sum()),
    "scored_epics": len(scored),
    "avg_epics_per_call": successful_calls["epic_count"].mean() if len(successful_calls) else 0.0,
}])

latency_tokens = pd.DataFrame([{
    "successful_calls": len(successful_calls),
    "failed_calls": int(call_metrics["status"].eq("error").sum()),
    "avg_latency_seconds": successful_calls["latency_seconds"].mean() if len(successful_calls) else None,
    "p50_latency_seconds": successful_calls["latency_seconds"].quantile(0.50) if len(successful_calls) else None,
    "p95_latency_seconds": successful_calls["latency_seconds"].quantile(0.95) if len(successful_calls) else None,
    "avg_input_tokens": successful_calls["input_tokens"].mean() if len(successful_calls) else None,
    "avg_output_tokens": successful_calls["output_tokens"].mean() if len(successful_calls) else None,
    "avg_total_tokens": successful_calls["total_tokens"].mean() if len(successful_calls) else None,
    "total_input_tokens": successful_calls["input_tokens"].sum() if len(successful_calls) else 0,
    "total_output_tokens": successful_calls["output_tokens"].sum() if len(successful_calls) else 0,
    "total_tokens": successful_calls["total_tokens"].sum() if len(successful_calls) else 0,
    "tokens_per_scored_epic": successful_calls["total_tokens"].sum() / len(scored) if len(scored) else None,
}])

print("Evaluation summary")
display(summary)
print("Population diagnostics")
display(diagnostics)
print("LLM latency / token summary")
display(latency_tokens)
print("Per-call LLM metrics")
display(call_metrics)

output_path = save_results_excel(
    results,
    EXPERIMENT_NAME,
    extra_sheets={
        "evaluation_summary": summary,
        "population_diagnostics": diagnostics,
        "llm_metrics": call_metrics,
        "latency_tokens": latency_tokens,
    },
)
print(f"Saved: {output_path}")
'''


def make_batch(template_path, output_path, title, what_llm_sees, experiment_name, prompt, include_description):
    template = load(template_path)
    cells = shared_prefix(template)
    cells[0]["source"] = title
    cells[1]["source"] = what_llm_sees

    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["source"] = cell.get("source", "").replace(
                'EXPERIMENT_NAME = "E12_BUSINESS_NEEDS_STAGE_CUSTOM_PROMPT"',
                f'EXPERIMENT_NAME = "{experiment_name}"',
            ).replace(
                'EXPERIMENT_NAME = "E14_THEME_NEEDS_DESCRIPTION_STAGE_CUSTOM_PROMPT"',
                f'EXPERIMENT_NAME = "{experiment_name}"',
            )

    cells.extend([
        {"cell_type": "markdown", "metadata": {}, "source": "## Production prompt"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": f"SYSTEM_PROMPT = {prompt!r}\n\n" + batch_prediction_code(include_description).split("\n\ndef validate_theme_batch_response", 1)[0]},
    ])
    first, rest = batch_prediction_code(include_description).split("\n\ndef validate_theme_batch_response", 1)
    cells[-1]["source"] = f"SYSTEM_PROMPT = {prompt!r}\n\n" + first
    cells.extend([
        {"cell_type": "markdown", "metadata": {}, "source": "## Prediction"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "def validate_theme_batch_response" + rest},
        {"cell_type": "markdown", "metadata": {}, "source": "## Evaluation"},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": evaluation_code()},
    ])
    notebook = {"cells": cells, "metadata": deepcopy(template.get("metadata", {})), "nbformat": 4, "nbformat_minor": template.get("nbformat_minor", 5)}
    save(output_path, notebook)


clone_enhanced(
    E12,
    BASE / "12_business_needs_stage_enhanced_prompt.ipynb",
    "# Experiment 12 Enhanced — Theme Business Needs + Stage + L3 — Enhanced Prompt, IDs Only\n\nSame fixed-50 data and execution as E12; only the decision prompt is changed to treat Stage as a boundary rather than positive evidence.",
    "E12_BUSINESS_NEEDS_STAGE_ENHANCED_PROMPT",
    E12_ENHANCED_PROMPT,
)
clone_enhanced(
    E14,
    BASE / "14_theme_needs_description_stage_enhanced_prompt.ipynb",
    "# Experiment 14 Enhanced — Theme Business Needs + Description + Stage + L3 — Enhanced Prompt, IDs Only\n\nSame fixed-50 data and execution as E14; only the decision prompt is changed to make Business Needs primary, Description clarifying, and Stage a boundary.",
    "E14_THEME_NEEDS_DESCRIPTION_STAGE_ENHANCED_PROMPT",
    E14_ENHANCED_PROMPT,
)

make_batch(
    E14,
    BASE / "15_theme_needs_description_theme_batch_custom_prompt.ipynb",
    "# Experiment 15 — Theme Needs + Description — Theme Batch, Custom Prompt, IDs Only",
    "## What the LLM sees\n\nOne call per Theme for the sampled valid Epics in that Theme. Shared Theme Business Needs + Description; each Epic carries only its own Stage(s) and candidate L3s. Ground truth is never sent.",
    "E15_THEME_NEEDS_DESCRIPTION_THEME_BATCH_CUSTOM_PROMPT",
    E15_PROMPT,
    True,
)
make_batch(
    E12,
    BASE / "16_business_needs_theme_batch_custom_prompt.ipynb",
    "# Experiment 16 — Theme Business Needs — Theme Batch, Custom Prompt, IDs Only",
    "## What the LLM sees\n\nOne call per Theme for the sampled valid Epics in that Theme. Shared Theme Business Needs; each Epic carries only its own Stage(s) and candidate L3s. Theme Description and ground truth are never sent.",
    "E16_BUSINESS_NEEDS_THEME_BATCH_CUSTOM_PROMPT",
    E16_PROMPT,
    False,
)

print("Generated E12 Enhanced, E14 Enhanced, E15, and E16")
