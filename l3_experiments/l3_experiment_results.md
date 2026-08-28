# L3 Capability Selection Experiments — Results

## Current Decision

**Use E5 — Full Context + Hierarchy as the quality baseline.**

E5 remains the strongest overall result on the current valid evaluation population of **11 Epics**. E8 is the strongest Theme-batch variant when scalability matters.

## Results

| Experiment | Context | Exact Match | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| E1 — Theme + Stage | Theme + Stage + L3 candidates, per Epic | 45.45% | 69.70% | 80.30% | 73.33% |
| E2 — Full Context | Theme + Epic + Stage + L3 candidates | 27.27% | 62.12% | 74.24% | 63.33% |
| E3 — No Theme Description | Theme needs + Epic + Stage + L3 candidates | 45.45% | 71.21% | 74.24% | 69.39% |
| E4 — No Theme | Epic + Stage + L3 candidates | — | — | — | — |
| **E5 — Full + Hierarchy** | E2 + L1/L2 hierarchy | **63.64%** | **80.30%** | 74.24% | **75.45%** |
| E6 — Enhanced E5 Prompt | Same context as E5; prompt only changed | 36.36% | 68.18% | 74.24% | 67.27% |
| E7 — Theme Batch | E1-style context; all valid Epics for a Theme in one call | 27.27% | 63.64% | 65.15% | 59.70% |
| **E8 — Theme Batch + Hierarchy** | E7 + L1/L2 hierarchy | **45.45%** | **72.73%** | **74.24%** | **70.30%** |

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

## Theme-Batch Efficiency

| Experiment | LLM Calls | Valid Epics | Avg Epics / Call | Total Tokens | Tokens / Epic |
|---|---:|---:|---:|---:|---:|
| E1 — Per Epic | 11 | 11 | 1.00 | 41,819 | 3,802 |
| E7 — Theme Batch | 9 | 11 | 1.22 | 33,419 | 3,038 |
| E8 — Theme Batch + Hierarchy | 9 | 11 | 1.22 | 33,564 | 3,051 |

E8 used about **20% fewer total tokens per Epic than E1** while preserving the same exact-match rate, but E1 retained higher recall and F1.

## Key Findings

- **E5 is the strongest quality configuration** and remains the baseline.
- **Hierarchy consistently helps disambiguation.** E8 substantially improved over E7 across exact match, precision, recall, and F1.
- Theme batching without hierarchy (E7) reduced quality significantly versus E1.
- E8 recovered much of that loss: it matched E1 on exact match, improved precision, and reduced token usage, though F1 remained lower.
- Current batching savings are limited because the 11 valid Epics span **9 Theme calls** (only **1.22 Epics per call** on average). Benefits may be larger for Themes containing more valid Epics.
- E6 showed that a longer, more prescriptive prompt can reduce quality; avoid unnecessary prompt over-engineering.
- Ground truth is a benchmark, not guaranteed truth. Some prediction/GT disagreements may reflect GT ambiguity or mapping issues.

## Baselines Going Forward

- **Best quality:** E5 — Full Context + Hierarchy
- **Best no-Epic-context quality:** E1 — Theme + Stage
- **Best scalable Theme-batch variant:** E8 — Theme Batch + Hierarchy
