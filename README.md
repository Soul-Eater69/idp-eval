# IDP LLM Evaluation Framework

A reusable evaluation framework built on Arize Phoenix. Cases use generic
`input`, `context`, `output`, `instructions`, and
`retrieved_documents` fields.

## Metrics

| Metric | Meaning | Required fields | Optional evidence |
| --- | --- | --- | --- |
| `coverage` | How much materially important source information reached the output? | `context`, `output` | — |
| `faithfulness` | Are output claims supported by the authoritative source? | `context`, `output` | — |
| `instruction_adherence` | Were explicit output instructions followed? | `instructions`, `output` | `context` |
| `relevance_at_{k}` | What fraction of the top-K documents are relevant to the query? | `input`, `retrieved_documents` | — |
| `ndcg_at_{k}` | How well are relevant documents ranked for the query? | `input`, `retrieved_documents` | — |

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

Faithfulness asks whether claims in the generated `output` are supported by the
authoritative `context` (`output → context`). It uses Phoenix's built-in
faithfulness evaluator and does not semantically use `input`, `instructions`,
metadata, or retrieved documents. Structured context and output values are
rendered through the same generic `render_value()` path.

## Instruction adherence

`InstructionAdherenceEvaluator(judge, verbose=False)` is a one-call holistic
judge. It sends the complete rendered `instructions` and complete rendered
`output` in one structured request, plus rendered `context` when present as
optional supporting evidence. The judge identifies materially distinct,
independently checkable instructions and classifies each as `followed` or
`violated`; Python deduplicates exact normalized repeats, assigns stable IDs,
and computes the fraction followed.

```python
case = EvaluationCase(
    instructions={
        "count": "Generate exactly 3 items",
        "requirements": [
            "Each item must contain a title",
            "Do not include implementation details",
        ],
    },
    context={"approved_options": ["Option A", "Option B", "Option C"]},
    output=[
        {"title": "Option A"},
        {"title": "Option B"},
        {"title": "Option C"},
    ],
)
result = InstructionAdherenceEvaluator(judge, verbose=True).evaluate(case)
```

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

| Metric | Required | Optional evidence | Meaning | Notebook |
| --- | --- | --- | --- | --- |
| Coverage | `context`, `output` | — | source information represented | [`coverage_evaluator_usage.ipynb`](notebooks/coverage_evaluator_usage.ipynb) |
| Faithfulness | `context`, `output` | — | output claims supported by source | [`faithfulness_evaluator_usage.ipynb`](notebooks/faithfulness_evaluator_usage.ipynb) |
| Instruction Adherence | `instructions`, `output` | `context` | explicit instructions followed | [`instruction_adherence_evaluator_usage.ipynb`](notebooks/instruction_adherence_evaluator_usage.ipynb) |

These are small, backend-independent examples using application-owned Azure
configuration placeholders. The setup guide covers the equivalent gateway
configuration.

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
included in evaluator prompts.

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
`max_concurrency` limit across judge calls. A list stored in `case.output`
remains one structured output, case, and trace; only the explicit
`group["outputs"]` list creates multiple cases and traces.

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
one EvaluationCase = one idp_eval.evaluate root trace
one actual judge call = one evaluator stage span
```

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
