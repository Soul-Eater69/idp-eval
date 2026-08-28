# L3 Capability Selection Experiments — Results

## Current Decision

**Use E5 — Full Context + Hierarchy as the baseline.**

E5 produced the best overall result on the current valid evaluation population of **11 Epics**.

## Results

| Experiment | Context | Exact Match | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| E1 — Theme + Stage | Theme + Stage + L3 candidates | 45.45% | 69.70% | 80.30% | 73.33% |
| E2 — Full Context | Theme + Epic + Stage + L3 candidates | 27.27% | 62.12% | 74.24% | 63.33% |
| E3 — No Theme Description | Theme needs + Epic + Stage + L3 candidates | 45.45% | 71.21% | 74.24% | 69.39% |
| E4 — No Theme | Epic + Stage + L3 candidates | — | — | — | — |
| **E5 — Full + Hierarchy** | E2 + L1/L2 hierarchy | **63.64%** | **80.30%** | 74.24% | **75.45%** |
| E6 — Enhanced E5 Prompt | Same context as E5; prompt only changed | 36.36% | 68.18% | 74.24% | 67.27% |

> E4 evaluation-summary values were not visible in the captured output, so they are intentionally left blank rather than inferred.

## Evaluation Population

For the 20-Theme run:

- Total Epics: **28**
- Valid Epics: **11**
- Excluded Epics: **17**
- `gt_not_fully_retrievable`: **17**
- Missing GT: **0**
- No Stage: **0**
- No Candidates: **0**

Only valid Epics were sent to the LLM and scored.

## Key Findings

- **E5 is the strongest configuration.**
- Adding **L1/L2 hierarchy** improved precision and exact-match accuracy substantially compared with E2.
- E2's extra context without hierarchy did not help and performed worse than E1/E3.
- E6's longer coverage/pruning prompt **hurt precision without improving recall**, so it should not replace E5.
- Prompt complexity should remain limited; further tuning should be driven by manual review of E5 mismatches.
- Ground truth should be treated as a benchmark, not assumed infallible; some disagreements may reflect GT ambiguity or mapping issues.

## Baseline Going Forward

Use:

**E5 — Full Context + Hierarchy**

with the existing E5 prompt unchanged as the current reference configuration.
