# L3 Capability Classification — Theme Context & Theme-Batch Experiment Comparison

## Executive Summary

This report compares five L3 capability-classification configurations on the **same fixed evaluation population of 50 valid Epics (seed = 42)**.

The experiments isolate two questions:

1. **Which Theme-level semantic context is most useful?**
   - Theme Business Needs
   - Theme Description
   - Theme Business Needs + Theme Description
2. **Does grouping Epics by Theme improve quality and efficiency compared with one LLM call per Epic?**

### Current best overall configuration

**E15 — Theme Business Needs + Theme Description, Theme Batch** is the strongest overall configuration.

- **Best F1:** 59.31%
- **Best Recall:** 83.00%
- Precision: 52.37%
- Exact Match: 28.00%
- **p95 latency:** 16.05 s — comfortably below the 30 s target
- **28 LLM calls** for 50 Epics instead of 50 calls
- **107,380 total tokens**
- **2,147.6 tokens per scored Epic**

E14 remains the strongest per-Epic configuration when exact match and precision are prioritized, but E15 provides the best overall balance of recall, F1, latency, call count, and token efficiency.

---

## Evaluation Setup

All five experiments use the same evaluation conditions so that the results are directly comparable.

| Setting | Value |
|---|---|
| Evaluation population | 50 valid Epics |
| Sampling | Fixed random sample |
| Seed | 42 |
| Themes represented | 28 |
| Ground truth | Jira-configured L3 capabilities |
| Candidate space | L3 capabilities mapped to each Epic's Value Stream Stage(s) |
| Output | L3 capability IDs only |
| Reasoning effort | `low` |
| Ground truth sent to LLM | No |
| Primary quality metrics | Exact Match, Precision, Recall, F1 |
| Production latency target | p95 < 30 seconds |

Only Epics whose complete ground-truth L3 set is retrievable from their Stage-derived candidate set are included in this evaluation.

---

## Experiment Matrix

| Experiment | Theme Business Needs | Theme Description | Epic-specific Stage | Epic-specific L3 Candidates | Execution |
|---|:---:|:---:|:---:|:---:|---|
| **E12** | ✅ | ❌ | ✅ | ✅ | One call per Epic |
| **E13** | ❌ | ✅ | ✅ | ✅ | One call per Epic |
| **E14** | ✅ | ✅ | ✅ | ✅ | One call per Epic |
| **E16** | ✅ | ❌ | ✅ | ✅ | One call per Theme |
| **E15** | ✅ | ✅ | ✅ | ✅ | One call per Theme |

### Per-Epic candidate context

For each Epic, the model receives the Epic's own Value Stream Stage context and only the L3 candidates available within that Stage-derived candidate space.

Stage context includes:

- `stage_id`
- `stage_name`
- `stage_description`
- `entrance_criteria`
- `exit_criteria`

Candidate L3 context includes:

- `capability_id`
- `capability_name`
- `capability_description`
- `capability_tier`

The model does **not** receive Epic description, Epic success criteria, L1/L2 hierarchy, or ground truth in these five experiments.

---

# 1. Quality Comparison

| Experiment | Context / Execution | Exact Match | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| **E12** | Business Needs + Stage, per Epic | 22.00% | 48.37% | 74.67% | 54.04% |
| **E13** | Theme Description + Stage, per Epic | 26.00% | 45.67% | 54.00% | 46.77% |
| **E14** | Business Needs + Description + Stage, per Epic | **36.00%** | **54.83%** | 70.67% | 57.85% |
| **E16** | Business Needs + Stage, Theme batch | 26.00% | 50.13% | 69.00% | 55.60% |
| **E15** | Business Needs + Description + Stage, Theme batch | 28.00% | 52.37% | **83.00%** | **59.31%** |

## Quality observations

### Theme Business Needs is stronger than Theme Description alone

E12 materially outperforms E13 on recall and F1:

- Recall: **74.67% vs 54.00%**
- F1: **54.04% vs 46.77%**

This indicates that **Theme Business Needs contains the stronger direct business-function signal** for L3 selection.

### Theme Description adds useful context when combined with Business Needs

E14 improves substantially over both single-context per-Epic experiments:

- Exact Match: **36.00%** — highest among the five configurations
- Precision: **54.83%** — highest among the five configurations
- F1: **57.85%**

The result suggests that Theme Description is most useful as **supporting context that clarifies the Business Needs**, rather than as a standalone classification signal.

### Theme batching changes the quality profile

E15 produces the highest overall F1 and recall:

- Recall rises to **83.00%**
- F1 rises to **59.31%**

The tradeoff is lower exact match than E14:

- E14 Exact Match: **36.00%**
- E15 Exact Match: **28.00%**

This suggests that Theme batching helps the model capture a broader set of correct capabilities, but it may also introduce additional selections that reduce exact-set accuracy.

---

# 2. Latency Comparison

## Per-call latency

| Experiment | Calls | Avg Latency | p50 Latency | p95 Latency | < 30 s p95? |
|---|---:|---:|---:|---:|:---:|
| **E12** | 50 | 7.37 s | 7.19 s | 11.93 s | ✅ |
| **E13** | 50 | **6.23 s** | **6.03 s** | **9.54 s** | ✅ |
| **E14** | 50 | 7.39 s | 7.31 s | **10.87 s** | ✅ |
| **E16** | **28** | 10.05 s | 10.21 s | 15.57 s | ✅ |
| **E15** | **28** | 10.39 s | 10.64 s | 16.05 s | ✅ |

All five configurations satisfy the production requirement of **p95 < 30 seconds**.

### Latency interpretation

The Theme-batch experiments have slightly higher latency **per LLM call** because each request can contain multiple Epics. However, they require only **28 calls for 50 Epics**, compared with 50 calls for the per-Epic experiments.

Average Epics per Theme-batch call:

- E15: **1.79 Epics/call**
- E16: **1.79 Epics/call**

The batch configurations therefore reduce LLM request count by grouping sampled Epics that share the same Theme context.

---

# 3. Token / Cost Efficiency

## Token usage

| Experiment | Avg Input Tokens / Call | Avg Output Tokens / Call | Avg Total Tokens / Call | Total Tokens | Tokens / Scored Epic |
|---|---:|---:|---:|---:|---:|
| **E12** | 1,638.46 | 621.44 | 2,259.90 | 112,995 | 2,259.90 |
| **E13** | 1,415.92 | **515.10** | **1,931.02** | 96,551 | 1,931.02 |
| **E14** | 2,301.04 | 614.70 | 2,915.74 | 145,787 | 2,915.74 |
| **E16** | 2,327.68 | 861.14 | 3,188.82 | **89,287** | **1,785.74** |
| **E15** | 2,942.50 | 892.50 | 3,835.00 | 107,380 | 2,147.60 |

## Token observations

### E16 is the most token-efficient configuration

E16 has the lowest total token usage:

- **89,287 total tokens**
- **1,785.74 tokens per Epic**

It achieves this by sharing Theme Business Needs once across multiple Epics in each Theme-level call.

### E15 gives the best quality / efficiency balance

Although E15 includes both Business Needs and Theme Description, Theme batching keeps total usage to **107,380 tokens**.

This is lower than:

- E12: 112,995 tokens
- E14: 145,787 tokens

while E15 still produces the **best F1 and recall**.

### E14 is the most expensive of the five

E14 sends both Theme Business Needs and Theme Description independently for every Epic, resulting in:

- **145,787 total tokens**
- **2,915.74 tokens per Epic**

E15 reuses that same shared Theme-level evidence in a batch call and therefore achieves better overall quality with substantially lower total token usage.

---

# 4. Per-Epic vs Theme-Batch Comparison

## Business Needs only

| Metric | E12 — Per Epic | E16 — Theme Batch |
|---|---:|---:|
| Exact Match | 22.00% | **26.00%** |
| Precision | 48.37% | **50.13%** |
| Recall | **74.67%** | 69.00% |
| F1 | 54.04% | **55.60%** |
| p95 latency | **11.93 s** | 15.57 s |
| LLM calls | 50 | **28** |
| Total tokens | 112,995 | **89,287** |
| Tokens / Epic | 2,259.90 | **1,785.74** |

**Takeaway:** Business-Needs batching improves exact match, precision, F1, call count, and token efficiency, with a moderate reduction in recall.

## Business Needs + Theme Description

| Metric | E14 — Per Epic | E15 — Theme Batch |
|---|---:|---:|
| Exact Match | **36.00%** | 28.00% |
| Precision | **54.83%** | 52.37% |
| Recall | 70.67% | **83.00%** |
| F1 | 57.85% | **59.31%** |
| p95 latency | **10.87 s** | 16.05 s |
| LLM calls | 50 | **28** |
| Total tokens | 145,787 | **107,380** |
| Tokens / Epic | 2,915.74 | **2,147.60** |

**Takeaway:** Theme batching trades some precision and exact-set accuracy for a large recall improvement, slightly higher F1, fewer calls, and materially lower token usage.

---

# 5. Overall Ranking

## 1. E15 — Business Needs + Theme Description, Theme Batch

**Recommended current production candidate.**

Why:

- Highest F1: **59.31%**
- Highest Recall: **83.00%**
- p95: **16.05 s**
- 28 calls instead of 50
- 107,380 total tokens
- Shared Theme context is reused efficiently across Epics

Primary weakness:

- Exact Match (**28%**) and Precision (**52.37%**) are below E14.

## 2. E14 — Business Needs + Theme Description, Per Epic

Best choice when precision and exact-set agreement are more important than recall or efficiency.

- Best Exact Match: **36.00%**
- Best Precision: **54.83%**
- F1: **57.85%**
- Very strong p95: **10.87 s**

Primary weakness:

- Highest token usage: **145,787**
- 50 independent LLM calls

## 3. E16 — Business Needs, Theme Batch

Best efficiency-oriented configuration.

- Lowest total tokens: **89,287**
- Lowest tokens per Epic: **1,785.74**
- Only 28 calls
- F1: **55.60%**
- p95: **15.57 s**

## 4. E12 — Business Needs, Per Epic

Strong recall but less efficient than E16 and lower overall F1 than E15/E14/E16.

## 5. E13 — Theme Description, Per Epic

Fastest per-call configuration, but Theme Description alone provides the weakest overall classification quality.

---

# Recommendation

**Lock E15 as the current baseline / production candidate for the next round of work.**

The evidence from the 50-Epic evaluation supports the following architecture:

> **Theme Business Needs + Theme Description shared once per Theme, with each Epic carrying its own Value Stream Stage context and Stage-derived L3 candidate set, classified in a single Theme-level LLM call using low reasoning.**

This configuration gives the strongest overall F1 and recall while remaining comfortably inside the p95 latency target and reducing both LLM call count and total token usage compared with the strongest per-Epic configuration.

Future prompt refinements should therefore use **E15 as the baseline** and focus specifically on improving precision / exact match **without sacrificing its 83% recall or its Theme-batch efficiency advantage**.
