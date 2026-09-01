from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOKS = {
    17: Path("l3_experiments/17_theme_needs_description_stage_individual.ipynb"),
    18: Path("l3_experiments/18_business_needs_stage_individual.ipynb"),
    19: Path("l3_experiments/19_theme_needs_description_stage_batch.ipynb"),
    20: Path("l3_experiments/20_business_needs_stage_batch.ipynb"),
}
COMBINED = {17, 19}
COMMON = Path("l3_experiments/common.py")
REPORT = Path("l3_experiments/e17_e20_results_comparison.md")


def source(cell):
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


# 1) Shared exact-prompt audit. This captures the literal strings passed to the gateway,
# including failed calls, rather than reconstructing them later.
common = COMMON.read_text(encoding="utf-8")
if "_prompt_audit_log" not in common:
    anchor = "def call_llm(gateway: Any, system_prompt: str, user_prompt: str) -> str:\n"
    audit_helpers = '''_prompt_audit_log: list[dict[str, Any]] = []


def reset_prompt_audit_log() -> None:
    """Clear exact prompt audit entries before an experiment run."""
    _prompt_audit_log.clear()


def get_prompt_audit_log() -> list[dict[str, Any]]:
    """Return a copy of exact prompt audit entries in gateway-call order."""
    return [dict(entry) for entry in _prompt_audit_log]


'''
    if anchor not in common:
        raise RuntimeError("Could not locate call_llm() anchor in common.py")
    common = common.replace(anchor, audit_helpers + anchor, 1)

new_metrics_function = '''def call_llm_with_metrics(
    gateway: Any,
    system_prompt: str,
    user_prompt: str,
    **options: Any,
) -> tuple[str, dict[str, float | int | None]]:
    """Call the gateway, recording the exact prompts plus latency/token usage."""
    started = perf_counter()
    audit_entry: dict[str, Any] = {
        "call_index": len(_prompt_audit_log) + 1,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "status": "started",
        "latency_seconds": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "error": None,
    }
    _prompt_audit_log.append(audit_entry)

    try:
        response = gateway.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **options,
        )
    except Exception as exc:
        latency_seconds = perf_counter() - started
        audit_entry.update(
            status="error",
            latency_seconds=latency_seconds,
            error=str(exc),
        )
        raise

    latency_seconds = perf_counter() - started
    token_metrics = _token_metrics(response)
    audit_entry.update(
        status="ok",
        latency_seconds=latency_seconds,
        **token_metrics,
    )

    metrics: dict[str, float | int | None] = {
        "latency_seconds": latency_seconds,
        **token_metrics,
        "prompt_call_index": audit_entry["call_index"],
    }
    return _assistant_text(response), metrics
'''

pattern = re.compile(
    r"def call_llm_with_metrics\(.*?\n\n\ndef parse_json_response",
    flags=re.DOTALL,
)
if not pattern.search(common):
    raise RuntimeError("Could not locate call_llm_with_metrics() in common.py")
common = pattern.sub(new_metrics_function + "\n\n\ndef parse_json_response", common, count=1)
COMMON.write_text(common, encoding="utf-8")


# 2) Notebook fixes: restore theme_context(), reset audit before execution, and export
# full exact prompt log + one prompt sample after each run.
for number, path in NOTEBOOKS.items():
    nb = json.loads(path.read_text(encoding="utf-8"))

    prompt_index = None
    for index, cell in enumerate(nb["cells"]):
        text = source(cell)
        if cell.get("cell_type") == "code" and "SYSTEM_PROMPT =" in text and "def build_user_prompt" in text:
            prompt_index = index
            break
    if prompt_index is None:
        raise RuntimeError(f"No production prompt cell found in {path}")

    prompt_source = source(nb["cells"][prompt_index])
    # Remove any prior theme_context definition if this updater is re-run.
    prompt_source = re.sub(
        r"\n\ndef theme_context\(.*?(?=\n\ndef |\Z)",
        "",
        prompt_source,
        flags=re.DOTALL,
    )
    if number in COMBINED:
        helper = '''


def theme_context(row):
    """Return exactly the shared Theme fields that are model-visible."""
    return {
        "theme_business_needs": row["theme_business_needs"],
        "theme_description": row["theme_description"],
    }
'''
    else:
        helper = '''


def theme_context(row):
    """Return exactly the shared Theme fields that are model-visible."""
    return {
        "theme_business_needs": row["theme_business_needs"],
    }
'''
    nb["cells"][prompt_index]["source"] = prompt_source.rstrip() + helper

    # Remove prior audit cells if re-run.
    cleaned = []
    for cell in nb["cells"]:
        text = source(cell)
        if "PROMPT_AUDIT_RESET_E17_E20" in text or "PROMPT_AUDIT_EXPORT_E17_E20" in text:
            continue
        if cell.get("cell_type") == "markdown" and text == "## Exact prompt audit from this run":
            continue
        cleaned.append(cell)
    nb["cells"] = cleaned

    # Find the prediction/evaluation code cell by the actual run call.
    run_index = None
    for index, cell in enumerate(nb["cells"]):
        text = source(cell)
        if cell.get("cell_type") == "code" and "=run()" in text.replace(" ", "") and "call_metrics" in text:
            run_index = index
            break
    if run_index is None:
        raise RuntimeError(f"No run/evaluation cell found in {path}")

    reset_source = '''# PROMPT_AUDIT_RESET_E17_E20
from common import reset_prompt_audit_log

reset_prompt_audit_log()
'''
    nb["cells"].insert(run_index, code_cell(reset_source))
    run_index += 1

    audit_source = '''# PROMPT_AUDIT_EXPORT_E17_E20
from common import get_prompt_audit_log

prompt_log = pd.DataFrame(get_prompt_audit_log())
if not prompt_log.empty:
    prompt_log.insert(0, "experiment", EXPERIMENT_NAME)

prompt_sample = prompt_log.head(1).copy()

print(f"Exact prompt audit entries captured: {len(prompt_log)}")
if not prompt_sample.empty:
    sample = prompt_sample.iloc[0]
    print("\nSYSTEM PROMPT SAMPLE — EXACT TEXT SENT TO LLM")
    print("=" * 80)
    print(sample["system_prompt"])
    print("\nUSER PROMPT SAMPLE — EXACT TEXT SENT TO LLM")
    print("=" * 80)
    print(sample["user_prompt"])

extra_sheets = {
    "call_metrics": call_metrics,
    "prompt_log": prompt_log,
    "prompt_sample": prompt_sample,
}
if "summary" in globals():
    extra_sheets["run_summary"] = summary

prompt_audit_output_path = save_results_excel(
    results,
    EXPERIMENT_NAME,
    extra_sheets=extra_sheets,
)
print(f"Saved results with prompt audit: {prompt_audit_output_path}")
'''
    nb["cells"].insert(run_index + 1, markdown_cell("## Exact prompt audit from this run"))
    nb["cells"].insert(run_index + 2, code_cell(audit_source))

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# 3) Concise numeric comparison from the completed runs supplied by the user.
REPORT.write_text('''# E17–E20 Results Comparison

All four experiments use the same fixed seed-42 sample. E17/E18 are individual Stage calls; E19/E20 batch unique Stages by Theme. Metrics below are from the completed notebook runs shown on 2026-09-01.

| Experiment | Model-visible Theme context | Mode | Evaluated | Exact | Precision | Recall | F1 | Successful / failed calls | Avg latency | p95 latency | Total tokens | Tokens / scored record |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E17** | Needs + Description | Individual | 50 | **36.00%** | **56.50%** | 70.67% | **59.74%** | 50 / 0 | **8.00s** | **11.94s** | 138,758 | 2,775.16 |
| **E18** | Needs only | Individual | 50 | 28.00% | 51.87% | 70.67% | 56.69% | 50 / 0 | 8.11s | 12.99s | 112,273 | 2,245.46 |
| **E19** | Needs + Description | Theme batch by Stage ID | 50 | 22.00% | 47.47% | 71.33% | 54.60% | **30 / 0** | 8.78s | 13.49s | 90,771 | **1,815.42** |
| **E20** | Needs only | Theme batch by Stage ID | 48 | 20.83% | 46.24% | **75.35%** | 54.48% | 29 / 1 | 10.13s | 35.20s | **64,181** | 1,337.10 |

## Readout

**Highest raw classification quality: E17.** It leads exact match, precision, and F1. Adding Theme Description to the individual setup improves E17 over E18 by +8 exact-match points, +4.63 precision points, and +3.05 F1 points while recall is unchanged.

**Preferred batch configuration: E19.** It is the cleaner production batch tradeoff: all 50 records scored with zero failed calls, only 30 LLM calls, and 1,815 tokens per scored record. Relative to E17 it cuts calls by **40%** and total tokens by about **34.6%**, while recall is slightly higher (+0.67 points). The cost is lower exact match (-14 points), precision (-9.03 points), and F1 (-5.14 points).

E20 is cheapest and has the highest measured recall, but one Theme call failed, only 48 records were scored, and p95 latency rose to 35.20s. Its quality comparison is therefore not fully apples-to-apples with the three complete 50-record runs.

## Recommendation

- Use **E17** when classification quality is the primary objective.
- Use **E19** when batching/cost/call-count efficiency matters and the quality tradeoff is acceptable. It is the recommended batch architecture from these runs.
- Keep Theme Description in the candidate architecture: it clearly helps individual quality and E19 is more reliable than the Needs-only batch run.

## Prompt audit added to E17–E20

Each notebook now records the literal strings passed to the gateway for every actual LLM call. The result workbook contains:

- `prompt_log`: every exact `system_prompt` and formatted `user_prompt`, in gateway-call order, including failed calls.
- `prompt_sample`: the first actual call as an easy-to-read sample.
- `call_metrics`: latency/token metrics from the experiment.
- `run_summary`: the experiment's displayed aggregate metrics when available.

The notebook also prints the exact formatted prompt sample so the visible text can be compared directly with what was sent in `messages[].content`.
''', encoding="utf-8")

print("Updated common.py, E17-E20 notebooks, and comparison report.")
