# idp-eval Setup and Usage

## 1. Installation

From a repository checkout, install the base package with `uv`:

```bash
uv sync
```

Install development and test dependencies when working on the project:

```bash
uv sync --extra dev
```

With `pip`, install the local package instead:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"  # development dependencies
```

The project metadata uses the package name `idp-eval`, but this guide does not
assume it has been published to a package index.

## 2. Environment Configuration

`create_judge()` resolves explicit arguments first, then environment variables,
then an optional YAML file. These environment variables supply the corporate IDP
gateway configuration:

| Variable | Purpose |
|---|---|
| `IDP_EVAL_MODEL` | Gateway model name |
| `IDP_EVAL_BASE_URL` | LLM gateway base URL |
| `IDP_EVAL_APP_ID` | Gateway `app-id` value |
| `IDP_EVAL_AUTH_URL` | IDP authentication endpoint |
| `IDP_EVAL_CLIENT_ID` | IDP client identifier |
| `IDP_EVAL_CLIENT_SECRET` | IDP client secret |
| `IDP_EVAL_USER` | IDP username |
| `IDP_EVAL_PASSWORD` | IDP password |

Example placeholders:

```bash
export IDP_EVAL_MODEL="model-name"
export IDP_EVAL_BASE_URL="https://gateway.example"
export IDP_EVAL_APP_ID="application-id"
export IDP_EVAL_AUTH_URL="https://auth.example"
export IDP_EVAL_CLIENT_ID="client-id"
export IDP_EVAL_CLIENT_SECRET="client-secret"
export IDP_EVAL_USER="service-user"
export IDP_EVAL_PASSWORD="service-password"
```

Optional settings are `IDP_EVAL_CONFIG`, which points to a YAML configuration
file, and `IDP_EVAL_VERIFY_SSL`. TLS verification defaults to enabled; disable it
only for an explicitly controlled local test environment.

Never commit secrets. This repository ignores `.env`, but `idp-eval` does not
load it automatically; export variables through your normal environment-loading
workflow.

## 3. Optional Phoenix Setup

Tracing and judge creation are separate concerns (Phoenix config never touches
`create_judge`). Phoenix is configured primarily through its standard environment
variables, which the Phoenix SDK resolves natively:

| Variable                    | Used by                          | Purpose                                          |
| --------------------------- | -------------------------------- | ------------------------------------------------ |
| `PHOENIX_COLLECTOR_ENDPOINT`| `phoenix.otel` (trace export)    | Where OTEL traces are sent                       |
| `PHOENIX_BASE_URL`          | `phoenix.client` (REST)          | Phoenix REST/client URL for span annotations     |
| `PHOENIX_API_KEY`           | both OTEL export and the client  | Authentication for OTEL export and REST calls    |
| `PHOENIX_PROJECT_NAME`      | `phoenix.otel`                   | Default project name                             |

`PHOENIX_COLLECTOR_ENDPOINT` (trace export) and `PHOENIX_BASE_URL` (REST
annotations) are **distinct paths**; set both to your Phoenix host for a remote
instance. `PHOENIX_API_KEY` authenticates both.

### Local Phoenix (no API key)

```bash
phoenix serve
```

```python
from idp_eval import register_tracing

register_tracing(project_name="my-eval-project")
```

Unauthenticated local Phoenix needs **no** API key and no endpoint — the Phoenix
SDK defaults to `http://localhost:6006`.

### Authenticated / remote Phoenix (environment-driven — recommended)

```bash
export PHOENIX_COLLECTOR_ENDPOINT="https://phoenix.example.com"
export PHOENIX_BASE_URL="https://phoenix.example.com"
export PHOENIX_API_KEY="your-api-key"        # keep secrets in the environment
```

```python
register_tracing(project_name="my-eval-project")
```

The same configuration drives trace export, root evaluation spans, native model
spans, and the `PhoenixEvaluationWriter` span annotations (`output="phoenix"`).

### Explicit arguments (optional)

`register_tracing` also accepts explicit `endpoint` / `api_key`, resolved as
**explicit argument → Phoenix env var → SDK default**. Prefer env vars for
secrets; a placeholder is shown only to illustrate the argument:

```python
register_tracing(
    project_name="my-eval-project",
    endpoint="https://phoenix.example.com",
    # api_key="your-api-key",  # prefer PHOENIX_API_KEY in the environment
)
```

For programmatic (non-env) REST config, `PhoenixEvaluationWriter(base_url=...,
api_key=...)` forwards to the native `phoenix.client.Client`. `EvaluationFramework`
itself has no Phoenix parameters — configure via environment variables.

When tracing is registered, each `framework.evaluate(...)` call creates one root
`idp_eval.evaluate` trace for that case. Tracing is optional.

## 4. Basic Framework Setup

```python
from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
    SourceCoverageEvaluator,
    TaskCoverageEvaluator,
    create_judge,
)

judge = create_judge()

framework = EvaluationFramework(
    evaluators=[
        FaithfulnessEvaluator,
        SourceCoverageEvaluator,
        TaskCoverageEvaluator,
        InstructionAdherenceEvaluator,
    ],
    judge=judge,
)
```

The framework constructs each evaluator class with the shared judge.

## 5. EvaluationCase Fields

| Field | Meaning |
|---|---|
| `input` | What the model was asked to do |
| `context` | Authoritative source, retrieved, or reference information |
| `output` | Generated response being evaluated |
| `instructions` | Optional explicit user or HITL instructions that apply to the output |

An `EvaluationCase` is exactly **one** logical evaluation unit. All content
fields are optional at the model level and default to `None`; each *evaluator*
declares which fields it actually requires (see §17), so you only supply what the
metrics you run will use.

### Structured values

`input`, `context`, `output`, and `instructions` may be **structured values** —
recursively `str`, `int`, `float`, `bool`, `None`, `dict[str, value]`, or
`list[value]` — not just pre-rendered strings. The framework renders them to
readable, labeled text for the judge deterministically (no LLM, no raw JSON):

```python
case = EvaluationCase(
    context={
        "description": "Improve customer onboarding",
        "business_needs": [
            "Reduce onboarding time by 25%",
            "Retain current identity provider",
        ],
    },
    output={
        "title": "Improve onboarding workflow",
        "success_criteria": ["Cut setup to under 10 minutes"],
    },
)
```

`context` above renders as:

```
Description: Improve customer onboarding

Business Needs:
- Reduce onboarding time by 25%
- Retain current identity provider
```

Rules: dictionary insertion order is preserved, `snake_case` keys become
`Title Case` labels (uppercase acronyms kept), scalar values render inline as
`Label: value`, scalar lists become bullet lists, nested structures stay
hierarchical (e.g. `Metadata:` then `  Priority: high`, and list-of-dict entries
as `- Title: Epic`), and value text passes through unchanged.
Unsupported objects (sets, tuples, bytes, arbitrary instances) raise a clear
`TypeError` at case construction.

> A `list` in `output` is a **single structured output**, not a request to
> evaluate many outputs. `output=["Step 1", "Step 2"]` is one case. Use
> `evaluate_many` / `evaluate_groups` (§10a) for many outputs.

## 6. Instruction Adherence Example

`instructions` is one text field. Use a multiline string for multiple explicit
instructions:

```python
case = EvaluationCase(
    input="",
    context="",
    instructions="""
Keep the response under 40 words.
Use exactly 3 bullet points.
Do not mention pricing.
""",
    output="""
- Fast setup
- Easy integration
- Reliable support
""",
)

results = framework.evaluate(
    case,
    metrics=["instruction_adherence"],
)
```

Stage 1 extracts independently checkable instructions. Stage 2 marks each fixed
instruction `followed` or `violated`. Python calculates `followed / total`.

## 7. Coverage Example

Coverage measures how completely the output represents important information from
the context. There are two explicit variants:

| Evaluator                | Metric name       | Scope                                                            |
| ------------------------ | ----------------- | --------------------------------------------------------------- |
| `SourceCoverageEvaluator`| `source_coverage` | The **whole source** (context) defines what should be covered.  |
| `TaskCoverageEvaluator`  | `task_coverage`   | Only the source information **relevant to the task** (`input`). |

Both share the same two-stage mechanics (extract items, then classify each
against the output; Python derives `covered`/`partial`/`missing` = `1.0/0.5/0.0`
and averages). They differ only in what Stage 1 extracts. Coverage focuses on
omissions — unsupported additions are evaluated separately by faithfulness.

### Task coverage — `input` scopes which parts of the context are relevant

```python
case = EvaluationCase(
    input="Summarize the product's supported deployment options.",
    context=(
        "The product supports cloud and on-premises deployment. "
        "On-premises deployments require version 4.2 or later. "
        "Billing is invoiced monthly."   # irrelevant to the task
    ),
    output="The product supports cloud and on-premises deployment.",
)

framework = EvaluationFramework(
    evaluators=[TaskCoverageEvaluator],
    judge=judge,
)
result = framework.evaluate(case)["task_coverage"]
```

Only deployment (task-relevant) information becomes coverage items; the billing
sentence is ignored.

### Source coverage — the whole context defines what should be covered

```python
case = EvaluationCase(
    context=document,          # no input required for source coverage
    output=summary,
)

framework = EvaluationFramework(
    evaluators=[SourceCoverageEvaluator],
    judge=judge,
)
result = framework.evaluate(case)["source_coverage"]
```

Stage 1 receives only the context (no task, instructions, or output), so every
important item in the source is expected to be represented. Typical uses:
summarization, document compression, and generic source-to-output transforms.
Details use `source_item` / `total_items`; task coverage details use `requirement`
/ `total_requirements`. The score means the same for both.

## 8. Faithfulness Example

Faithfulness checks whether output claims are supported by the authoritative
context.

```python
case = EvaluationCase(
    input="Summarize the payment methods.",
    context="Customers can pay by credit card or bank transfer.",
    output="Customers can pay by credit card, bank transfer, or cryptocurrency.",
)

faithfulness = framework.evaluate(
    case,
    metrics=["faithfulness"],
)["faithfulness"]
```

## 9. Evaluator Selection

You configure evaluators **once**, in the constructor. You do not repeat them on
every call.

- **constructor `evaluators=[...]`** — the configured/available evaluator set.
- **`evaluate(case)`** — runs *all* configured evaluators.
- **`evaluate(case, metrics=[...])`** — optional subset filter: runs only those
  configured evaluators.

```python
framework = EvaluationFramework(
    evaluators=[SourceCoverageEvaluator, FaithfulnessEvaluator],
    judge=judge,
)

results = framework.evaluate(case)              # both configured evaluators
subset = framework.evaluate(case, metrics=["faithfulness"])  # just one
```

`metrics=` never instantiates an evaluator that was not configured; an unknown or
unconfigured metric name raises `KeyError`. Required-field validation (§17)
applies only to the **selected** evaluators — a field an unselected evaluator
would need does not block the call.

## 10. Running a Subset of Metrics

```python
results = framework.evaluate(
    case,
    metrics=["faithfulness", "instruction_adherence"],
)
```

Unknown metric names raise `KeyError`.

## 10a. Bulk and Grouped Evaluation

`evaluate_many` runs many independent cases. Each case keeps its own validation,
its own root `idp_eval.evaluate` trace, its own results, and its own Excel /
Phoenix rows — there is no batch-level trace.

```python
cases = [
    EvaluationCase(input=task1, context=theme1, output=epic1, case_id="theme-1:epic-1"),
    EvaluationCase(input=task1, context=theme1, output=epic2, case_id="theme-1:epic-2"),
    EvaluationCase(input=task2, context=theme2, output=epic3, case_id="theme-2:epic-3"),
]

results = framework.evaluate_many(cases)                      # all configured
subset = framework.evaluate_many(cases, metrics=["source_coverage", "faithfulness"])
```

Failure behavior is **fail fast**: the whole batch is validated for the selected
evaluators *before any judge call*, so a malformed later case never triggers paid
work on earlier cases. Invalid cases raise a case-aware `ValueError` (naming the
`case_id`) rather than being skipped or coerced to not-applicable.

For a source (Theme) with several generated outputs (Epics), `evaluate_groups`
is a thin convenience that fans out to one case per output and reuses
`evaluate_many`:

```python
results = framework.evaluate_groups([
    {"input": task1, "context": theme1, "outputs": [epic1, epic2], "group_id": "theme-1"},
    {"input": task2, "context": theme2, "outputs": [epic3],        "group_id": "theme-2"},
])
```

This produces **three independent cases / traces** (`theme-1:0`, `theme-1:1`,
`theme-2:0`). Each output is evaluated on its own — `theme1 + [epic1, epic2]` is
never sent as one coverage unit. Case ids come from an optional `case_ids` list,
else `f"{group_id}:{i}"`, else `f"{group_index}:{i}"`; `group_id` is carried on
`case.metadata` and output objects are never mutated.

## 11. Reading Results

`evaluate()` returns a dictionary keyed by metric name. Every value is an
`EvaluationResult` with `metric`, `score`, `label`, `explanation`, and `details`.

```python
for result in results.values():
    print(result.metric)
    print(result.score)
    print(result.label)
    print(result.explanation)
    print(result.details)
```

`score` is the quantitative result (`[0, 1]`, higher is better). `label` is a
short qualitative interpretation **derived from the score**, and is
metric-specific rather than a generic high/medium/low bucket:

| Metric | `score == 1.0` | `0 < score < 1` | `score == 0.0` | not applicable |
|---|---|---|---|---|
| `source_coverage`, `task_coverage` | `complete` | `incomplete` | `missing` | `not_applicable` |
| `instruction_adherence` | `fully_followed` | `violations_present` | `violated` | `not_applicable` |
| `faithfulness` | provided by Phoenix (e.g. `faithful` / `unfaithful`) — not derived here |

A partial instruction-adherence result (any violation) is therefore
`violations_present`, never a misleading `high`. Not-applicable results carry
`score=None` and `label="not_applicable"`.

## 12. Phoenix Logging

Enable tracing before creating evaluations, select Phoenix output on the
framework, and put the case identifier on `EvaluationCase`:

```python
from idp_eval import register_tracing

register_tracing(project_name="my-eval-project")

framework = EvaluationFramework(
    evaluators=[InstructionAdherenceEvaluator],
    judge=judge,
    output="phoenix",
)

case = EvaluationCase(
    case_id="case-001",
    input="",
    context="",
    instructions="Use exactly 3 bullet points.",
    output="- First\n- Second\n- Third",
)

results = framework.evaluate(
    case,
    run_name="test-run-1",
    dataset_name="instruction-test",
)
```

Phoenix receives one root evaluation trace, evaluator stage spans, native model
spans when model instrumentation is active, and native metric annotations on the
root span. Requesting Phoenix output without an active root span raises a clear
persistence error.

Annotation logging uses the `phoenix.client.Client`, which reads `PHOENIX_BASE_URL`
and `PHOENIX_API_KEY` from the environment (see section 3) — the same
authenticated remote setup that drives trace export also drives annotations, with
no manual client or header construction.

## 13. Excel Output

Install the Excel extra, then provide `excel_path`:

```bash
python -m pip install -e ".[excel]"
```

```python
framework = EvaluationFramework(
    evaluators=[TaskCoverageEvaluator],
    judge=judge,
    output="excel",
    excel_path="evaluation_results.xlsx",
)
```

Use `output="both"` with the same `excel_path` to publish the one computed result
set to Phoenix and Excel. Evaluators are not run twice.

The workbook has a summary sheet plus structured per-metric detail sheets so
results are readable without inspecting JSON:

| Sheet | One row per | Columns |
|---|---|---|
| `evaluations` | case + metric | `run_name`, `dataset_name`, `case_id`, `trace_id`, `metric`, `score`, `label`, `explanation`, `annotator_kind`, `timestamp`, `raw_details_json` |
| `source_coverage_items` | extracted source item | identity cols + `item_id`, `source_item`, `meaningfully_present`, `fully_present`, `status`, `item_score`, `reason` |
| `task_coverage_items` | task-relevant requirement | identity cols + `item_id`, `requirement`, `meaningfully_present`, `fully_present`, `status`, `item_score`, `reason` |
| `instruction_adherence_items` | instruction | identity cols + `instruction_id`, `instruction`, `status`, `item_score`, `reason` |

The identity columns (`run_name`, `dataset_name`, `case_id`, `trace_id`,
`metric`) are repeated on every detail row so you can filter or pivot a single
sheet. Detail sheets appear only when a metric with a registered item layout is
written. Faithfulness and custom code metrics have no item list, so they show up
only in `evaluations`; their full `details` are preserved in the trailing
`raw_details_json` column. Numeric scores are stored as numbers, and header rows
are bold, frozen, and auto-filtered.

## 14. Custom Evaluation Logging

Custom results use the framework's configured output writers:

```python
framework.log_custom_evaluation(
    name="json_validity",
    score=1.0,
    label="valid",
    explanation="The output is valid JSON.",
    details={"validator": "json.loads"},
    kind="CODE",
    case_id="case-001",
    run_name="test-run-1",
    dataset_name="instruction-test",
)
```

You may instead pass an existing `EvaluationResult` to
`framework.log_evaluation(result, case=case, annotator_kind="CODE")`.
Supported annotator kinds are `LLM`, `CODE`, and `HUMAN`.

## 15. Reusing a Framework Instance

Create the judge and framework once, then reuse them:

```python
for case in cases:
    results = framework.evaluate(case)
```

This also reuses the same configured judge and output writers.

## 16. Minimal Application Example

```python
from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    TaskCoverageEvaluator,
    create_judge,
    register_tracing,
)

register_tracing(project_name="my-eval-project")
judge = create_judge()
framework = EvaluationFramework(
    evaluators=[TaskCoverageEvaluator],
    judge=judge,
)

case = EvaluationCase(
    case_id="example-001",
    input="Summarize supported payment methods.",
    context="Customers can pay by credit card or bank transfer.",
    output="Customers can pay by credit card.",
)

results = framework.evaluate(case)
for result in results.values():
    print(result.metric, result.score, result.label)
```

## 17. Required Fields by Evaluator

Each evaluator declares the case fields it **requires** (validated before its
first judge call). Fields it does not require may be omitted or left `None`.

| Evaluator | Required fields |
|---|---|
| `SourceCoverageEvaluator` | `context`, `output` |
| `TaskCoverageEvaluator` | `input`, `context`, `output` |
| `FaithfulnessEvaluator` | `context`, `output` |
| `InstructionAdherenceEvaluator` | `instructions`, `output` |

(`FaithfulnessEvaluator` also passes `input` to Phoenix when present, but does not
require it.)

**Missing / empty** required content is any of: `None`, `""`, a whitespace-only
string, `{}`, or `[]`. Scalars such as `0` and `False` are legitimate values and
are **not** treated as missing. When a required field is missing, the framework
raises a `ValueError` before any judge call, e.g.:

```
TaskCoverageEvaluator requires non-empty `input`.

Received:
  input: missing
  context: present
  output: present
```

This required-field contract is **consistent across entry points**: it is
enforced identically whether you call an evaluator directly
(`SourceCoverageEvaluator(judge).evaluate(case)`), via `framework.evaluate(case)`,
or via `framework.evaluate_many(cases)` (which pre-validates the whole batch). In
every case validation runs before the first judge call.

Distinguish two outcomes:

- **Missing required field** (`None` / `""` / `{}` / `[]`) → `ValueError` before
  any judge call.
- **Valid required field, but the metric extracts nothing** (e.g. instructions
  are supplied but no checkable instruction is found) → a metric-defined
  not-applicable result (`score=None`, `label="not_applicable"`).

**Extra fields are allowed.** A case may carry fields an evaluator does not use
(so one case can run several metrics). Running `SourceCoverageEvaluator` on a case
that also has `input` and `instructions` is valid — it consumes only `context`
and `output`. Validation applies only to the metrics selected for the call (§9).

## 19. Error and Not-Applicable Behavior

- Missing required fields for a *selected* evaluator raise `ValueError` before any
  judge call (fail fast) — they are never silently converted to not-applicable.
- An empty instruction *extraction* (instructions supplied but nothing checkable
  found) is a metric-defined not-applicable: `score=None`,
  `label="not_applicable"`, Stage 2 skipped.
- Coverage (`source_coverage` and `task_coverage`) is not applicable when Stage 1
  extraction identifies no items; classification is skipped and the result is
  `score=None`, `label="not_applicable"`.
- Unknown requested metric names raise `KeyError`.
- Missing judge configuration raises `ValueError` listing missing field names,
  without exposing secret values.
- A persistence failure raises `PersistenceError`; computed results remain
  available on `error.results` and evaluators are not rerun.

## 20. Development and Testing

Run the offline suite and syntax compilation with:

```bash
pytest
python -m compileall idp_eval
```

Development validation scripts live under `scripts/`. Some benchmark scripts
make paid model calls and should be run only intentionally; they are not part of
the getting-started workflow.
