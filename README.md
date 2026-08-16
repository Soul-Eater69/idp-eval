# IDP LLM Evaluation Framework

A reusable evaluation framework on top of [Arize Phoenix](https://github.com/Arize-ai/phoenix)
that can evaluate **any** generated AI output. It is not tied to Jira, RAG,
summarization, or test-case generation. Every evaluation uses the same generic
triple:

```text
input        = what the model was asked to do (the task/request)
context      = authoritative source information
output       = generated content being evaluated
instructions = explicit instructions to check (optional; instruction_adherence only)
```

`input` and `instructions` are separate fields with fixed meanings, so the same
`EvaluationCase` can be run through every metric without any field changing
meaning. `instruction_adherence` reads `instructions` and never falls back to
`input`.

## Metrics (v1)

| Metric                  | Question                                      | Direction               | Better |
| ----------------------- | --------------------------------------------- | ----------------------- | ------ |
| `faithfulness`          | Is the output grounded in the context?        | `output -> context`     | higher |
| `coverage`              | How much relevant context reached the output? | `context -> output`     | higher |
| `instruction_adherence` | Did the output obey the supplied explicit instructions? | `instructions -> output` | higher |

Three complementary questions:

- **Faithfulness** — did the output **ADD** unsupported information?
- **Coverage** — did the output **OMIT** important task-relevant source information?
- **Instruction Adherence** — did the output **OBEY** the supplied explicit instructions?

### Faithfulness vs. hallucination

There is deliberately **no separate `hallucination` metric**.

- **Hallucination** = the failure / problem (the output states things the
  context does not support).
- **Faithfulness** = the metric used to evaluate grounding and detect that
  failure. Higher faithfulness means fewer hallucinated / unsupported additions.

Faithfulness is **Phoenix built-in** (`FaithfulnessEvaluator`) and measures
whether the output **adds unsupported information**.

### Coverage

Coverage measures how completely the generated output represents the
task-relevant information in the supplied context. It is a **two-stage** metric —
**two judge calls** per case:

```
Stage 1 (extraction):     input + context        → atomic requirements   (no output)
Stage 2 (classification): requirements + output  → 2 booleans / requirement
```

**Stage 1** derives the atomic, task-relevant requirements from `input + context`
and **never sees the output**, so extraction can't be biased by what was
generated. It keeps important qualifiers (e.g. "25%", "real-time") attached and
splits compound requirements. Python then dedups (**normalized-exact**: lowercase
+ collapsed whitespace, first kept) and assigns stable ids `r1, r2, …`.

**Stage 2** classifies exactly that fixed requirement set against the `output`
(it cannot add/remove/rewrite requirements, so the denominator is fixed). For each
requirement the judge returns two booleans — `meaningfully_present` and
`fully_present` — and **Python derives the status**:

```python
not meaningfully_present            -> "missing"   (0.0)
meaningfully_present & fully_present -> "covered"  (1.0)
meaningfully_present & not full      -> "partial"  (0.5)
coverage = sum(item_scores) / number_of_requirements
```

The **LLM never returns a number**. For example four requirements judged
`covered, missing, partial, covered` give `(1.0 + 0.0 + 0.5 + 1.0) / 4 = 0.625` →
**62.5%**. `EvaluationResult.details` exposes `total_requirements`, the
covered/partial/missing counts, and `items` (each with `id`, `requirement`, both
booleans, derived `status`, Python `score`, and `reason`).

If Stage 1 finds **no** task-relevant requirements, coverage is **not-applicable**
(`score=None`, `label="not_applicable"`) and Stage 2 is skipped (one call only) —
a failure to identify requirements is not treated as perfect coverage.

Coverage measures completeness (omissions). Unsupported *additions* are **not**
penalized here — that is faithfulness's job; coverage never does hallucination
detection.

**Cost:** coverage uses **two judge calls total** per case (extraction +
batched classification), never one call per requirement. This is intentional — it
isolates the requirement denominator from the output and improves auditability and
stability.

**Known limitation (v1):** deduplication is normalized-exact only, so semantic
near-duplicates may remain distinct. See
[Roadmap](#coverage-roadmap-not-yet-implemented).

Uses two versioned Phoenix-style prompts: `prompts/coverage_extract.py`
(`COVERAGE_EXTRACT_PROMPT_V1`) and `prompts/coverage_classify.py`
(`COVERAGE_CLASSIFY_PROMPT_V1`).

### Instruction Adherence

Instruction adherence is a **custom** LLM-as-a-judge metric measuring whether the
output **obeys the explicit instructions** it was given. It uses a versioned
Phoenix-style prompt (`prompts/instruction_adherence.py`). The judge decomposes
the instructions into atomic instructions and classifies each `followed`,
`partial`, `violated`, or `not_applicable`; Python computes the score over the
**applicable** instructions only:

```python
INSTRUCTION_ADHERENCE_VALUES = {"followed": 1.0, "partial": 0.5, "violated": 0.0}
applicable = [i for i in instructions if i["status"] != "not_applicable"]
instruction_adherence = sum(values) / len(applicable)
```

`not_applicable` covers instructions that genuinely do not apply — e.g. a
conditional "If the account is inactive, include a warning." when the context
says the account is active. It is **excluded from the denominator**, not scored
as a success.

For this metric, put the explicit instruction text in the dedicated
**`EvaluationCase.instructions`** field (not `input`). `context` is optional and
consulted only when an instruction requires it (e.g. "only use information from
the context").

The metric returns `score=None`, `label="not_applicable"` when there is nothing
applicable to evaluate, with an explanation distinguishing the reason: no
instructions supplied, no meaningful instructions found, or all supplied
instructions were not applicable. None of these is treated as a perfect or
failing score.

### Which fields each metric reads

`EvaluationCase` fields have fixed meanings; each metric reads only what it needs:

- **faithfulness** — `input` (task) + `context` + `output`, passed to Phoenix.
- **coverage** — `input` (task, used to scope relevant context) + `context` + `output`.
- **instruction_adherence** — `instructions` + `context` + `output`. Reads the
  dedicated `instructions` field, never `input`.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```python
from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
    create_judge,
    register_tracing,
)

register_tracing(project_name="idp-eval")   # once, at startup (optional)
judge = create_judge()                       # configure the judge once

framework = EvaluationFramework(
    evaluators=[
        FaithfulnessEvaluator,
        CoverageEvaluator,
        InstructionAdherenceEvaluator,
    ],
    judge=judge,
)

results = framework.evaluate(EvaluationCase(
    input=user_task,
    context=source_context,
    output=generated_output,
))
# results["coverage"].score -> 0.75
```

You pass evaluator **classes** plus one shared `judge`; the framework constructs
each with `cls(llm=judge)`. (Passing already-constructed instances still works.)
Run a subset with `framework.evaluate(case, metrics=["faithfulness", "coverage"])`.

For instruction adherence, put the instruction text in the `instructions` field:

```python
case = EvaluationCase(
    input=user_task,
    instructions="Use exactly 3 bullet points.\nDo not mention customer names.",
    context=source_context,
    output=generated_output,
)
results = framework.evaluate(case, metrics=["instruction_adherence"])
```

See `example.py` for a full runnable script.

## Configuring the judge

`create_judge()` builds a Phoenix judge backed by the corporate IDP gateway. It
resolves configuration with the precedence **explicit argument > environment
variable > YAML file**, and raises a `ValueError` listing any missing field names
(never secret values) if configuration is incomplete. There are no built-in
defaults for required fields.

Preferred environment variables:

| Field               | Env var                   | Secret |
| ------------------- | ------------------------- | ------ |
| `model`             | `IDP_EVAL_MODEL`          |        |
| `base_url`          | `IDP_EVAL_BASE_URL`       |        |
| `app_id`            | `IDP_EVAL_APP_ID`         |        |
| `idp_auth_url`      | `IDP_EVAL_AUTH_URL`       |        |
| `idp_client_id`     | `IDP_EVAL_CLIENT_ID`      |        |
| `idp_client_secret` | `IDP_EVAL_CLIENT_SECRET`  | ✓      |
| `idp_user`          | `IDP_EVAL_USER`           |        |
| `idp_password`      | `IDP_EVAL_PASSWORD`       | ✓      |

Optional YAML (see `config.example.yaml`); point at it with `IDP_EVAL_CONFIG` or
`create_judge(config_path=...)`. Keep secrets out of committed YAML — provide
`idp_client_secret` / `idp_password` via environment variables or explicit args.

```python
judge = create_judge()                          # all from env / YAML
judge = create_judge(model="gpt-5-idp-test")    # override just the model
judge = create_judge(verify_ssl=False)          # local self-signed testing only
```

TLS verification defaults to `True`; set `verify_ssl=False` (or
`IDP_EVAL_VERIFY_SSL=false`) only for local testing. Secrets and JWTs are never
logged or included in error messages.

## Every metric returns the same shape

```python
EvaluationResult(metric, score, label, explanation, details)
```

This keeps the public API independent of Phoenix's internal `Score` object, so
the backend can change later.

## Project layout

```text
idp_eval/
├── models.py            # EvaluationCase, EvaluationResult, Evaluator interface
├── framework.py         # EvaluationFramework orchestrator
├── scoring.py           # deterministic scoring functions
├── judge.py             # JudgeConfig + create_judge (IDP gateway wiring)
├── phoenix_client.py    # Phoenix tracing registration
├── evaluators/          # faithfulness, coverage, instruction_adherence
└── prompts/             # versioned judge prompts + JSON schemas (custom metrics)
```

## Adding a metric

Implement the `Evaluator` interface and pass it to `EvaluationFramework`. No core
change is required.

```python
class InstructionAdherenceEvaluator(Evaluator):
    name = "instruction_adherence"
    def evaluate(self, case: EvaluationCase) -> EvaluationResult: ...
```

## Testing

```bash
pytest
```

- `tests/test_scoring.py` — scoring logic, no LLM.
- `tests/test_evaluators.py` — evaluators + framework via a `FakeJudge` and a
  fake Phoenix module, no real LLM calls.

Unit tests never call a real LLM or the IDP gateway.

## Benchmarking coverage

### Determinism / stability (developer tool)

LLM output is not deterministic, and extraction can vary its requirement set run
to run. To *observe* how stable it actually is, run the same case repeatedly
against the configured real judge:

```bash
python -m scripts.coverage_stability --runs 20
```

It prints each run's requirement count and score, then a summary: score
mean/min/max/range/stddev, requirement-count spread, mean pairwise
normalized-exact requirement overlap (Stage 1 extraction stability), and status
consistency for recurring requirements (Stage 2 classification stability). This is
a manual developer tool — it uses the real judge (two calls per run) and is
**not** part of the unit test suite (only the pure `summarize_runs` statistics are
unit-tested with fake data).

### Ground truth for coverage quality

Do not treat "another LLM said coverage = 78%" as ground truth. Recommended:

1. Human reviewers identify/review the golden atomic task-relevant requirements.
2. Humans label each requirement `covered` / `partial` / `missing`.
3. Python computes the GT score with the same `1.0 / 0.5 / 0.0` mapping.
4. Compare the evaluator's labels and score against those. A strong LLM may
   *bootstrap* proposed labels, but humans establish the final GT.

```json
{
  "input": "...", "context": "...", "output": "...",
  "gold_requirements": [
    {"requirement": "Reduce onboarding time by 25%", "status": "covered"},
    {"requirement": "Reduce abandoned registrations", "status": "missing"},
    {"requirement": "Automate identity verification", "status": "covered"},
    {"requirement": "Reduce verification effort by 40%", "status": "partial"}
  ],
  "gold_score": 0.625
}
```

## Coverage roadmap (not yet implemented)

Deliberately deferred until stability benchmarking shows they are needed:

- **Pinned / golden requirement checklist** — evaluate different outputs against a
  fixed human-reviewed requirement set (stable denominator) for benchmark use.
- **Importance weighting** — `must` / `should` / `nice` weights, only after
  benchmark evidence that the flat mean is misleading.
- **Semantic deduplication** — only if normalized-exact dedup proves insufficient.
