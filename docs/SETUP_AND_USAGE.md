# idp-eval Setup and Usage

## 1. Installation

```bash
uv sync
uv sync --extra dev
```

Or:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

Optional features are available through the `yaml`, `excel`, `tracing`, and
`phoenix` extras.

## 2. Judge backends

Both judge constructors return a Phoenix-compatible object and use:

```text
explicit argument > environment variable > optional YAML
```

### Corporate IDP/Mule gateway

```python
from idp_eval import create_gateway_judge

judge = create_gateway_judge()
judge = create_gateway_judge(model="test-model")
judge = create_gateway_judge(config_path="config.yaml")
```

| Variable | Purpose |
| --- | --- |
| `IDP_EVAL_MODEL` | gateway model name |
| `IDP_EVAL_BASE_URL` | gateway base URL |
| `IDP_EVAL_APP_ID` | gateway application ID |
| `IDP_EVAL_AUTH_URL` | IDP authentication endpoint |
| `IDP_EVAL_CLIENT_ID` | IDP client identifier |
| `IDP_EVAL_CLIENT_SECRET` | IDP client secret |
| `IDP_EVAL_USER` | IDP username |
| `IDP_EVAL_PASSWORD` | IDP password |
| `IDP_EVAL_GATEWAY_TIMEOUT` | client-side request timeout |

This preserves the corporate token and gateway contract. A larger client
timeout cannot override a shorter upstream Mule/gateway timeout.

### Direct Azure OpenAI

```python
from idp_eval import create_azure_judge

judge = create_azure_judge(
    model="azure-deployment",
    azure_endpoint="https://example-resource.openai.azure.com",
    tenant_id="tenant-id",
    client_id="client-id",
    client_secret="client-secret",
    api_version="2024-12-01-preview",
    timeout=180,
    proxy_url=None,
    verify_ssl=True,
    reasoning_effort=None,
)
```

| Variable | Purpose |
| --- | --- |
| `IDP_EVAL_AZURE_MODEL` | Azure deployment name |
| `IDP_EVAL_AZURE_ENDPOINT` | Azure OpenAI endpoint |
| `IDP_EVAL_AZURE_TENANT_ID` | Azure AD tenant |
| `IDP_EVAL_AZURE_CLIENT_ID` | service-principal client ID |
| `IDP_EVAL_AZURE_CLIENT_SECRET` | service-principal secret |
| `IDP_EVAL_AZURE_API_VERSION` | Azure OpenAI API version |
| `IDP_EVAL_AZURE_TIMEOUT` | client timeout, default 180 seconds |
| `IDP_EVAL_AZURE_PROXY_URL` | optional HTTP proxy |
| `IDP_EVAL_AZURE_VERIFY_SSL` | TLS verification, default true |
| `IDP_EVAL_AZURE_REASONING_EFFORT` | optional request setting |

Direct Azure uses `ClientSecretCredential` and the Cognitive Services bearer
token scope. Token caching and refresh are handled by Azure Identity. It does
not use an API key or the corporate gateway URL.

`IDP_EVAL_CONFIG` can point to YAML with a `judge` section for the gateway and an
`azure_judge` section for Azure. Keep secrets out of committed files. Setting
`verify_ssl=False` is only for controlled development environments.

Judge creation is separate from tracing and never adds temperature. The legacy
`create_judge()` name is a deprecated gateway alias.

Evaluator wiring does not depend on the backend:

```python
coverage = CoverageEvaluator(judge)
framework = EvaluationFramework(judge=judge, evaluators=[coverage])
```

## 3. Phoenix tracing

```python
from idp_eval import register_tracing

register_tracing(project_name="my-eval-project")
```

Local Phoenix started with `phoenix serve` normally needs no API key. Remote or
authenticated Phoenix uses its standard variables:

| Variable | Purpose |
| --- | --- |
| `PHOENIX_COLLECTOR_ENDPOINT` | trace export endpoint |
| `PHOENIX_BASE_URL` | REST endpoint for annotations |
| `PHOENIX_API_KEY` | authentication when enabled |
| `PHOENIX_PROJECT_NAME` | default project |

One `framework.evaluate(case)` call creates one `idp_eval.evaluate` root
trace. Every actual judge operation is a child stage span. Coverage creates one
`coverage.evaluate` stage. Native model spans may appear beneath it; the
package does not create duplicate model spans.

## 4. Framework setup

```python
from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
    NDCGAtKEvaluator,
    RelevanceAtKEvaluator,
    create_gateway_judge,
)

judge = create_gateway_judge()
framework = EvaluationFramework(
    evaluators=[
        CoverageEvaluator,
        FaithfulnessEvaluator,
        InstructionAdherenceEvaluator,
    ],
    judge=judge,
)
```

Classes receive the shared judge. Constructed instances are also accepted:

```python
framework = EvaluationFramework(
    evaluators=[CoverageEvaluator(judge, verbose=True)]
)
```

## 5. EvaluationCase fields

| Field | Meaning |
| --- | --- |
| `input` | task/request or retrieval query |
| `context` | authoritative source information |
| `output` | generated content being evaluated |
| `instructions` | explicit instructions for the output |
| `retrieved_documents` | ordered retrieval results |
| `case_id` | optional correlation identifier |
| `metadata` | optional non-prompt metadata |

Fields accept strings or nested dictionaries/lists of scalar values. Rendering
is deterministic. Required fields are evaluator-specific:

| Evaluator | Required fields |
| --- | --- |
| `CoverageEvaluator` | `context`, `output` |
| `FaithfulnessEvaluator` | `context`, `output` |
| `InstructionAdherenceEvaluator` | `instructions`, `output` |
| retrieval evaluators | `input`, `retrieved_documents` |

## 6. Coverage

Coverage asks how much materially important information from the full context is
represented in the output. It is recall-like and does not penalize unsupported
additions; faithfulness handles those.

```python
case = EvaluationCase(
    context=(
        "Users can view invoices. Invoices show the total due. "
        "A confirmation is sent after payment."
    ),
    output="Users can view invoices and see the total due.",
)
result = framework.evaluate(case, metrics=["coverage"])["coverage"]
```

`CoverageEvaluator(judge, verbose=False)` makes exactly one structured call
using `context + output`. The judge identifies all materially distinct source
items and returns two booleans:

- `fully_present=True` becomes `covered` and 1.0.
- `meaningfully_present=True, fully_present=False` becomes `partial` and 0.5.
- both false becomes `missing` and 0.0.

Python calculates the mean and label. An empty item set returns `score=None`
and `label="not_applicable"` after the one call.

There is no item-count target. The prompt preserves qualifiers, consolidates
semantic redundancy, excludes structural/meta wrapper text, and avoids counting
both an umbrella objective and equivalent detailed items. Generic topical
overlap is insufficient for partial credit.

Compact details:

```python
{
    "final_item_count": 3,
    "covered_count": 2,
    "partial_count": 0,
    "missing_count": 1,
    "judge_call_count": 1,
    "total_ms": 123.4,
    "verbose": False,
}
```

With `verbose=True`, `details["items"]` adds stable IDs, source text, the
binary judgments, Python-derived status and score, and reasons. Covered reasons
are empty; partial/missing reasons are non-empty.

## 7. Other evaluators

Instruction adherence reads only `instructions + output`. It extracts fixed
instructions and classifies each as `followed` or `violated`; Python computes
the score. No instructions returns `not_applicable` without a judge call.

Faithfulness uses Phoenix's native evaluator and asks whether output claims are
supported by context.

`RelevanceAtKEvaluator(k)` and `NDCGAtKEvaluator(k)` judge ranked retrieval
documents against the query in `input`. Relevance@K is binary Precision@K;
nDCG@K is derived from the same relevance judgments.

## 8. Bulk, grouped, and async APIs

```python
framework.evaluate_many([case1, case2])

framework.evaluate_groups([
    {
        "context": shared_context,
        "outputs": [output1, output2],
        "group_id": "group-1",
    }
])

await framework.a_evaluate_many(cases, max_concurrency=4)
```

Groups fan out into ordinary independent cases; evaluator work is not shared.
Async results preserve input order and a framework-level semaphore limits all
concurrent judge calls.

## 9. Output destinations

```python
framework = EvaluationFramework(
    evaluators=[CoverageEvaluator, FaithfulnessEvaluator],
    judge=judge,
    output="both",                 # "phoenix" | "excel" | "both" | None
    excel_path="evaluation_results.xlsx",
)
results = framework.evaluate(case, run_name="run-1", dataset_name="dataset-1")
```

Evaluation happens once. Phoenix writes native span annotations to the root case
span. Excel can contain:

| Sheet | One row per |
| --- | --- |
| `evaluations` | case and metric |
| `coverage_items` | verbose coverage item |
| `instruction_adherence_items` | instruction item |
| `retrieval_documents` | retrieval document |

Nested details remain JSON on the summary sheet. Persistence failures retain
computed results and never rerun evaluation.

## 10. Custom evaluation publishing

```python
framework.log_custom_evaluation(
    name="company_policy",
    score=1.0,
    label="pass",
    explanation="All policy rules passed.",
    kind="CODE",  # LLM | CODE | HUMAN
    case_id="case-001",
)
```

An existing `EvaluationResult` can be passed to `framework.log_evaluation`.

## 11. Validation and labels

Missing required fields fail before judge work. Empty structures count as
missing; numeric zero and boolean false are valid content. Extra fields are
allowed.

| Coverage score | Label |
| --- | --- |
| `1.0` | `complete` |
| `0 < score < 1` | `incomplete` |
| `0.0` | `missing` |
| `None` | `not_applicable` |

## 12. Verification

```bash
uv run pytest -q
python3 -m compileall -q idp_eval
```

Unit tests use fake judges and do not call a live model or gateway.
