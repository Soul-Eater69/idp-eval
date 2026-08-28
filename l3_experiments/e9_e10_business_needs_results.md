# E9 vs E10 — Business-Needs-Only L3 Experiments

## Setup

Both experiments use the same model-visible evidence:

- Theme business needs
- Value Stream Stage context
- Base L3 candidate fields

Excluded from both:

- Theme description
- Epic description
- Epic success criteria
- L1/L2 hierarchy
- Ground truth

The difference is execution:

- **E9:** classify each valid Epic separately.
- **E10:** classify all valid Epics under a Theme together in one LLM call.

## Results

| Experiment | Exact Match | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **E9 — Business Needs + Stage, per Epic** | **63.64%** | **83.33%** | **86.36%** | **83.94%** |
| E10 — Business Needs + Stage, Theme batch | 36.36% | 53.03% | 53.03% | 50.61% |

Evaluation population:

- Themes selected: **20**
- Total Epics: **28**
- Valid/scored Epics: **11**
- Excluded Epics: **17**
- `gt_not_fully_retrievable`: **17**
- LLM prediction errors: **0**

## Efficiency

| Metric | E9 | E10 |
|---|---:|---:|
| Successful LLM calls | 11 | 9 |
| Avg Epics / call | 1.00 | 1.22 |
| Avg latency | 22.35 s | 17.28 s |
| Total input tokens | 16,262 | 15,010 |
| Total output tokens | 22,139 | 14,159 |
| Total tokens | 38,401 | 29,169 |
| Tokens / Epic | ~3,491 | ~2,652 |

## Findings

- **E9 is substantially stronger than E10 on every quality metric.**
- E9 achieved the best result seen so far on this 11-Epic evaluation population: **83.94% F1**.
- Removing Theme description from the per-Epic setup improved performance relative to E1, suggesting Theme description may add noise when classification is already isolated to one Epic.
- Theme batching reduced token usage and call count, but quality dropped sharply.
- With no Epic description or success criteria, multiple Epics in the same Theme have limited semantic separation. Their Stage/candidate contexts can interfere with one another in the grouped prompt.
- The current sample is small: with 11 scored Epics, one Epic changes exact-match accuracy by about 9.1 percentage points.

## Current Takeaway

**E9 is the preferred configuration from these two experiments.**

It is both simpler and substantially more accurate than E10. Before treating it as the production winner, it should be validated on a larger valid-Epic population.
