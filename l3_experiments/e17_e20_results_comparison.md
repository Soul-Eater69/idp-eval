# E17–E20 Results Comparison

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
