# IDP LLM Evaluation Framework

A reusable evaluation framework built on Arize Phoenix. Cases use generic
`input`, `context`, `output`, `instructions`, and
`retrieved_documents` fields.

## Metrics

| Metric | Question | Required fields |
| --- | --- | --- |
| `coverage` | How much materially important source information is represented? | `context`, `output` |
| `faithfulness` | Is the output grounded in the context? | `context`, `output` |
| `instruction_adherence` | Did the output obey the explicit instructions? | `instructions`, `output` |
| `relevance_at_{k}` | What fraction of the top-K retrieved documents are relevant? | `input`, `retrieved_documents` |
| `ndcg_at_{k}` | How well are relevant documents ranked in the top K? | `input`, `retrieved_documents` |

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
meta-statements, and filler are not independent items. A business objective
counts only when it adds meaning not already represented by detailed items; an
umbrella and equivalent children are not double-counted. Generic topical overlap
does not earn partial credit—a concrete semantic component must be present.

Compact mode returns counts, one-call accounting, and timing. `verbose=True`
also returns the item-level audit trail and concise reasons for partial or
missing items. Covered items use an empty reason. Both modes score identically.

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
    InstructionAdherenceEvaluator,
    create_gateway_judge,
    register_tracing,
)

register_tracing(project_name="idp-eval")  # optional
judge = create_gateway_judge()

framework = EvaluationFramework(
    evaluators=[
        CoverageEvaluator,
        FaithfulnessEvaluator,
        InstructionAdherenceEvaluator,
    ],
    judge=judge,
)

case = EvaluationCase(
    input="Summarize the supplied product information.",
    context=source_context,
    output=generated_output,
    instructions="Use exactly three concise bullet points.",
)
results = framework.evaluate(case)
print(results["coverage"].score)
```

Evaluator classes receive the shared judge. Constructed instances also work,
including `CoverageEvaluator(judge, verbose=True)`. Each evaluator validates
only its required fields; extra fields are allowed. Structured dictionaries and
lists are rendered deterministically.

## Bulk, grouped, and async evaluation

```python
framework.evaluate_many([case1, case2, case3])

framework.evaluate_groups([
    {
        "context": shared_context,
        "outputs": [output1, output2],
        "group_id": "group-1",
    }
])

await framework.a_evaluate_many(cases, max_concurrency=4)
```

Grouped evaluation simply fans each output into an ordinary independent case.
Async evaluation preserves order and uses one global semaphore to cap judge
calls. Each coverage case makes one judge call.

## Judge backends

Evaluators receive the same Phoenix-compatible judge regardless of transport.
Choose one explicit constructor.

Corporate gateway:

```python
from idp_eval import create_gateway_judge

judge = create_gateway_judge(
    model="...",
    base_url="...",
    app_id="...",
    idp_auth_url="...",
    idp_client_id="...",
    idp_client_secret="...",
    idp_user="...",
    idp_password="...",
)
```

The standard corporate path preserves IDP authentication and Mule gateway
translation. Its client timeout cannot override a shorter upstream gateway
timeout.

Direct Azure OpenAI:

```python
from idp_eval import create_azure_judge

judge = create_azure_judge(
    model="...",                 # Azure deployment name
    azure_endpoint="...",
    tenant_id="...",
    client_id="...",
    client_secret="...",
    api_version="...",
    timeout=180,
)
```

This explicitly configured backend connects directly to the approved Azure
deployment using Azure AD client credentials. Optional `proxy_url` and
`verify_ssl` settings support controlled network environments.

The evaluator and framework wiring is identical for either judge:

```python
coverage = CoverageEvaluator(judge)
framework = EvaluationFramework(judge=judge, evaluators=[coverage])
```

Both backends resolve **explicit argument > environment variable > optional
YAML**. Gateway variables:

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

Azure variables:

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
never added. The older `create_judge()` remains a deprecated thin alias for the
gateway constructor.

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
