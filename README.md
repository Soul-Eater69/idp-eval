# IDP LLM Evaluation Framework

A reusable evaluation framework built on Arize Phoenix. Cases use generic
`input`, `context`, `output`, `instructions`, and
`retrieved_documents` fields.

## Metrics

| Metric | Semantic fields | Optional evidence | Descriptive input |
| --- | --- | --- | --- |
| `coverage` | `context`, `output` | — | allowed; ignored for scoring |
| `faithfulness` | `context`, `output` | — | allowed; ignored for scoring |
| `instruction_adherence` | `instructions`, `output` | `context` | allowed; ignored for scoring |
| `relevance_at_{k}` | `input`, `retrieved_documents` | — | `input` is semantic |
| `ndcg_at_{k}` | `input`, `retrieved_documents` | — | `input` is semantic |

Coverage detects omissions from the source. Faithfulness separately detects
unsupported additions to the output.

## Coverage

`CoverageEvaluator(judge, verbose=False)` performs whole-source coverage in one
structured judge call:

```text
context + output
      ↓
coverage.evaluate
      ↓
source items + binary judgments
      ↓
deterministic Python scoring
```

The judge identifies all materially distinct source information and returns
`meaningfully_present` and `fully_present` for each item. Python derives:

```text
fully_present                          covered   1.0
meaningfully_present, but not fully    partial   0.5
not meaningfully_present               missing   0.0
```

The final score is the mean of item scores. The judge never returns the
aggregate score, percentage, or label. Labels are `complete` for 1.0,
`incomplete` for a score strictly between 0 and 1, `missing` for 0.0, and
`not_applicable` when no source items are identified.

There is no fixed or approximate item-count target. The prompt asks for all
materially distinct facts, obligations, capabilities, requirements, objectives,
outcomes, constraints, prohibitions, actors, dependencies, thresholds, timing,
channels, and measurable targets while consolidating semantic redundancy.
Material qualifiers are preserved.

Headings, section labels, introductory phrases, structural instructions,
meta-statements, and filler are not independent items. A source objective
counts only when it adds meaning not already represented by detailed items; an
umbrella and equivalent children are not double-counted. Generic topical overlap
does not earn partial credit—a concrete semantic component must be present.

Compact mode returns counts, one-call accounting, and timing. `verbose=True`
also returns the item-level audit trail and concise reasons for partial or
missing items. Covered items use an empty reason. Both modes score identically.

## Faithfulness

`FaithfulnessEvaluator(judge, verbose=False)` asks whether distinct factual
claims in generated `output` are supported by authoritative `context`
(`output → context`). One structured judge call identifies factual claims and
classifies each as `supported` or `unsupported`; Python deduplicates normalized
exact repeats, assigns stable `F1`, `F2`, ... IDs, and computes:

```text
faithfulness = supported output claims / total factual output claims
```

For example, if context says cancellation is allowed within 24 hours and refunds
take 5 business days, output claiming "Cancellation is allowed within 24 hours"
is supported while "Refunds are instant" is unsupported, producing `0.5`.

Missing source facts do not reduce faithfulness: Coverage measures
`context → output` omissions, Faithfulness measures unsupported
`output → context` claims, and Instruction Adherence measures whether explicit
`instructions → output` constraints were followed.

The metric requires `context + output` and ignores `input`, `instructions`,
metadata, and retrieved documents. Dict/list/nested values are rendered through
`render_value()` and remain one case. Compact mode returns counts and timing;
`verbose=True` adds the claim-level status, item score, and reason audit trail.
A valid response with no checkable factual claims returns `not_applicable` after
the single judge call. Async evaluation uses native judge async generation when
available and otherwise uses the framework's shared-concurrency thread bridge.

## Instruction adherence

`InstructionAdherenceEvaluator(judge, verbose=False)` is a one-call holistic
judge. It sends the complete rendered `instructions` and complete rendered
`output` in one structured request, plus rendered `context` when present as
optional supporting evidence. The judge identifies materially distinct,
independently checkable instructions and classifies each as `followed` or
`violated`; Python deduplicates exact normalized repeats, assigns stable IDs,
and computes the fraction followed.

```python
instructions = """
Generate exactly 3 options.
Every option must contain a title.
Do not include implementation details.
Only use approved options from the supplied context.
"""

case = EvaluationCase(
    instructions=instructions,
    context={"approved_options": ["Option A", "Option B", "Option C"]},
    output=[
        {"title": "Option A"},
        {"title": "Option B"},
        {"title": "Option C"},
    ],
)
result = InstructionAdherenceEvaluator(judge, verbose=True).evaluate(case)
```

Plain multiline text is the common instruction shape. Dict, list, and nested
structured instructions are also supported through `render_value()`.

Count, range, universal (`each`/`every`/`all`), prohibition, structure, order,
language, and style constraints are interpreted generically by the judge—there
is no domain-specific Python rule engine. Context is evidence only: the judge
must not infer new instructions from it or score source completeness unless an
explicit instruction requires that. The evaluator ignores `input`, metadata,
and retrieved documents.
Compact mode returns counts, one-call accounting, and timing. Verbose mode also
returns item IDs, binary statuses, Python-assigned item scores, and concise
violation reasons. An empty judge-produced instruction set is
`not_applicable` after the one call; missing required case fields fail before it.

## Metric usage notebooks

| Notebook | Purpose |
| --- | --- |
| [`core_metrics_together_usage.ipynb`](notebooks/core_metrics_together_usage.ipynb) | **Recommended:** configure multiple metrics and run all or selected subsets |
| [`coverage_evaluator_usage.ipynb`](notebooks/coverage_evaluator_usage.ipynb) | Coverage-specific usage |
| [`faithfulness_evaluator_usage.ipynb`](notebooks/faithfulness_evaluator_usage.ipynb) | Faithfulness-specific usage |
| [`instruction_adherence_evaluator_usage.ipynb`](notebooks/instruction_adherence_evaluator_usage.ipynb) | Instruction Adherence-specific usage |

The guides are backend-independent and use application-owned Azure
configuration placeholders. The setup guide covers the equivalent gateway
configuration. Start with the combined guide when evaluating one generation
with several metrics; use the metric-specific guides for focused audit details.

## Install

```bash
uv sync
uv sync --extra dev
```

With pip:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

See [Setup and Usage](docs/SETUP_AND_USAGE.md) for full configuration.

## Usage

```python
from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessEvaluator,
    create_azure_judge,
    register_tracing,
)
from idp_eval.judges import AzureJudgeConfig

register_tracing(project_name="idp-eval")  # optional
azure_config = AzureJudgeConfig(
    model="...",
    azure_endpoint="...",
    tenant_id="...",
    client_id="...",
    client_secret="...",
    api_version="2024-12-01-preview",
    timeout=180,
    proxy_url=None,
    verify_ssl=True,
    reasoning_effort=None,
)
judge = create_azure_judge(config=azure_config)

framework = EvaluationFramework(
    judge=judge,
    evaluators=[CoverageEvaluator, FaithfulnessEvaluator],
)

case = EvaluationCase(
    input="Generate epics for the supplied business theme.",
    context={
        "customer_profile": {"region": "US", "tier": "enterprise"},
        "requirements": ["Response time under 2 seconds", "99.9% availability"],
    },
    output={
        "summary": "Proposed service configuration.",
        "actions": ["Apply regional routing", "Add availability monitoring"],
    },
)
results = framework.evaluate(case)
print(results["coverage"].score)
```

Evaluator classes receive the shared judge. Constructed instances also work,
including `CoverageEvaluator(judge, verbose=True)`. Each evaluator validates
only its required fields; extra fields are allowed. Structured dictionaries and
lists are rendered recursively and deterministically. Dictionary keys become
readable labels; no domain-specific schema is required. Case metadata is never
included in evaluator prompts. Optional `input` describes the task/query that
produced the output and is useful in traces, reports, debugging, heterogeneous
workloads, and retrieval evaluation. Merely being present on `EvaluationCase`
does not make it evidence for every metric: Coverage, Faithfulness, and
Instruction Adherence do not receive it in their judge prompts.

## Bulk, grouped, and async evaluation

```python
framework.evaluate(case)
framework.evaluate_many(cases)

groups = [
    {
        "group_id": "request-1",
        "context": {
            "requirements": ["Response time under 2 seconds"],
            "constraints": ["Use approved regions"],
        },
        "outputs": [
            {"summary": "Option A", "actions": ["Add caching"]},
            {"summary": "Option B", "actions": ["Scale workers"]},
        ],
    }
]
framework.evaluate_groups(groups)

result = await framework.a_evaluate(case, max_concurrency=4)
results = await framework.a_evaluate_many(cases, max_concurrency=4)
results = await framework.a_evaluate_groups(groups, max_concurrency=4)
```

Grouped evaluation simply fans each output into an ordinary independent case.
Async evaluation preserves order, and the framework enforces the shared
`max_concurrency` limit across judge calls.

For the common case where one generation contains multiple output objects, set
`evaluation_scope` directly on the case:

```python
case = EvaluationCase(
    input="Generate three options from the supplied source.",
    context=source,
    output=[option_a, option_b, option_c],
    evaluation_scope="both",  # "combined" | "individual" | "both"
)
results = framework.evaluate(case)

combined = results["combined"]
first_option = results["individual"][0]
```

The default is `"combined"`, so a list remains one structured output and the
normal flat `{metric: EvaluationResult}` return shape is unchanged. Individual
scope returns `{"combined": None, "individual": [...]}`. Both scope evaluates
the full list plus every top-level item and returns both sections. Each logical
evaluation gets its own root trace; per-item IDs derive from the parent case ID
as `case-id:0`, `case-id:1`, and so on. Only a non-empty top-level list can use
`individual` or `both`; dictionaries are never guessed to contain many outputs.

## Judge backends

Evaluators receive the same Phoenix-compatible judge regardless of transport.
For imported applications, the recommended path is to construct the resolved
backend config from the host application's settings/secrets layer and pass it
directly.

Corporate gateway:

```python
from idp_eval import create_gateway_judge
from idp_eval.judges import GatewayJudgeConfig

gateway_config = GatewayJudgeConfig(
    model="...",
    base_url="...",
    app_id="...",
    idp_auth_url="...",
    idp_client_id="...",
    idp_client_secret="...",
    idp_user="...",
    idp_password="...",
    verify_ssl=True,
    timeout=90,
)
judge = create_gateway_judge(config=gateway_config)
```

The standard corporate path preserves IDP authentication and Mule gateway
translation. Its client timeout cannot override a shorter upstream gateway
timeout.

| `GatewayJudgeConfig` field | Meaning |
| --- | --- |
| `model` | Gateway model identifier |
| `base_url` | Gateway base URL |
| `app_id` | Gateway application identifier |
| `idp_auth_url` | IDP authentication endpoint |
| `idp_client_id` | IDP client identifier |
| `idp_client_secret` | IDP client secret; keep outside source control |
| `idp_user` | IDP account name |
| `idp_password` | IDP account password; keep outside source control |
| `verify_ssl` | TLS certificate verification |
| `timeout` | Client-side request timeout |

Direct Azure OpenAI:

```python
from idp_eval import create_azure_judge
from idp_eval.judges import AzureJudgeConfig

azure_config = AzureJudgeConfig(
    model="...",
    azure_endpoint="...",
    tenant_id="...",
    client_id="...",
    client_secret="...",
    api_version="2024-12-01-preview",
    timeout=180,
    proxy_url=None,
    verify_ssl=True,
    reasoning_effort=None,
)
judge = create_azure_judge(config=azure_config)
```

This explicitly configured backend connects directly to the approved Azure
deployment using Azure AD client credentials. Optional `proxy_url` and
`verify_ssl` settings support controlled network environments.

| `AzureJudgeConfig` field | Meaning |
| --- | --- |
| `model` | Azure deployment name |
| `azure_endpoint` | Azure OpenAI resource endpoint |
| `tenant_id` | Azure tenant ID |
| `client_id` | Application/service-principal client ID |
| `client_secret` | Application/service-principal secret; keep outside source control |
| `api_version` | Azure OpenAI API version |
| `timeout` | Client-side request timeout |
| `proxy_url` | Optional HTTP proxy |
| `verify_ssl` | TLS certificate verification |
| `reasoning_effort` | Optional model request option, sent only when configured |

Convenience alternatives remain available: individual constructor keyword
arguments, environment variables, and optional YAML. Zero-argument calls such
as `create_gateway_judge()` and `create_azure_judge()` resolve environment/YAML
configuration when desired. Without `config=`, precedence remains **explicit
argument > environment variable > optional YAML**. These sources are optional,
not requirements. Do not mix `config=` with individual configuration arguments.

> **Security:** Never commit real client secrets, passwords, tokens, endpoints,
> or proxy URLs in notebooks, documentation examples, or config files. Keep
> repository examples as placeholders and use the host application's
> secret/configuration system in production.

The evaluator and framework wiring is identical for either judge:

```python
coverage = CoverageEvaluator(judge)
framework = EvaluationFramework(judge=judge, evaluators=[coverage])
```

Gateway convenience variables:

```text
IDP_EVAL_MODEL
IDP_EVAL_BASE_URL
IDP_EVAL_APP_ID
IDP_EVAL_AUTH_URL
IDP_EVAL_CLIENT_ID
IDP_EVAL_CLIENT_SECRET
IDP_EVAL_USER
IDP_EVAL_PASSWORD
```

Azure convenience variables:

```text
IDP_EVAL_AZURE_MODEL
IDP_EVAL_AZURE_ENDPOINT
IDP_EVAL_AZURE_TENANT_ID
IDP_EVAL_AZURE_CLIENT_ID
IDP_EVAL_AZURE_CLIENT_SECRET
IDP_EVAL_AZURE_API_VERSION
IDP_EVAL_AZURE_TIMEOUT
IDP_EVAL_AZURE_PROXY_URL
IDP_EVAL_AZURE_VERIFY_SSL
IDP_EVAL_AZURE_REASONING_EFFORT
```

`IDP_EVAL_CONFIG` may point to YAML containing `judge` and/or `azure_judge`
sections. TLS verification is on by default; disabling it is only for controlled
development. `reasoning_effort` is sent only when configured. Temperature is
never added.

For direct comparisons, populate both config objects in the latency notebook.
Gateway and Azure may use different deployment or model values, so the notebook
prints each configured model for auditability.

## Gateway vs Azure latency smoke test

[`notebooks/judge_backend_latency_comparison.ipynb`](notebooks/judge_backend_latency_comparison.ipynb)
runs the same `EvaluationCase` once through `create_gateway_judge(config=...)`
and `create_azure_judge(config=...)`. It compares end-to-end latency, coverage
score, label, extracted item count, and verbose item-level decisions.

This is a single-case smoke comparison, not a statistically meaningful
performance benchmark. It intentionally uses the production constructors and
backend config objects so it does not duplicate authentication or transport
code. Replace notebook placeholders locally—or wire values from the host
application's settings—and provide your own `golden_set_augmented_tagged.csv`
or update `GOLDEN_SET_PATH`. Never commit those values or the GT CSV.

## Tracing and output

Tracing is evaluation-only:

```text
one logical combined or individual evaluation = one idp_eval.evaluate root trace
one actual judge call = one evaluator stage span
```

Thus a default/combined case creates one root trace, an individual three-item
case creates three, and `evaluation_scope="both"` creates four. Scope expansion
is metric-agnostic; every evaluator still receives one ordinary case at a time.
When `case.input` is present, each root trace receives a bounded descriptive
preview (`idp_eval.input`) and a truncation flag; full unbounded input is not
copied into span attributes.

Coverage produces:

```text
idp_eval.evaluate
└── coverage.evaluate
    └── native model span, when emitted by Phoenix instrumentation
```

Instruction adherence similarly produces one
`instruction_adherence.evaluate` child stage for its single judge call.

Python scoring creates no extra spans. Phoenix native model instrumentation is
reused, not duplicated.

Evaluation executes once and the same records can go to Phoenix, Excel, both, or
neither:

```python
framework = EvaluationFramework(
    evaluators=[CoverageEvaluator, FaithfulnessEvaluator],
    judge=judge,
    output="both",  # "phoenix" | "excel" | "both" | None
    excel_path="evaluation_results.xlsx",
)
results = framework.evaluate(
    case,
    run_name="benchmark-v1",
    dataset_name="dataset-v1",
)
```

Phoenix output writes native annotations to the root case span. Excel output
uses these sheets where applicable:

| Sheet | Content |
| --- | --- |
| `evaluations` | one row per case and metric |
| `coverage_items` | verbose coverage item judgments |
| `instruction_adherence_items` | instruction judgments |
| `retrieval_documents` | retrieval judgments |

Compact coverage results omit item rows by design. Persistence errors retain the
computed results and never rerun evaluators.

Custom results use the same output path:

```python
framework.log_custom_evaluation(
    name="json_validity",
    score=1.0,
    label="valid",
    explanation="Output is valid JSON.",
    kind="CODE",  # LLM | CODE | HUMAN
    case_id="case-001",
)
```

## Retrieval metrics

`RelevanceAtKEvaluator(k)` and `NDCGAtKEvaluator(k)` evaluate ranked
`retrieved_documents` against `input`. Relevance@K is binary Precision@K;
nDCG@K is computed deterministically from the same relevance judgments.

## Testing

```bash
uv run pytest -q
python3 -m compileall -q idp_eval
```

Unit tests use fake judges and make no live LLM or gateway calls.
