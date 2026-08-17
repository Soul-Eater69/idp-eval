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

`input`, `context`, and `output` are required constructor fields. A particular
metric may use only a subset of them; `instructions` defaults to `None`.

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
    input="",                 # ignored by source coverage
    context=document,
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

## 9. Running All Configured Metrics

```python
results = framework.evaluate(case)
```

With no `metrics` argument, every evaluator configured on the framework runs.
Supply `instructions` when instruction adherence should be applicable.

## 10. Running a Subset of Metrics

```python
results = framework.evaluate(
    case,
    metrics=["faithfulness", "instruction_adherence"],
)
```

Unknown metric names raise `KeyError`.

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

## 17. Metric Input Requirements

| Metric | Fields used by the metric |
|---|---|
| `faithfulness` | `input`, `context`, `output` |
| `source_coverage` | `context`, `output` (`input` is ignored) |
| `task_coverage` | `input`, `context`, `output` |
| `instruction_adherence` | `instructions`, `output` |

Although instruction adherence does not read `input` or `context`, callers must
currently provide them when constructing `EvaluationCase`. Empty strings are
appropriate when running only that metric.

## 19. Error and Not-Applicable Behavior

- Missing or blank instructions produce an instruction-adherence result with
  `score=None` and `label="not_applicable"` without calling the judge.
- An empty instruction extraction is also not applicable and skips Stage 2.
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
