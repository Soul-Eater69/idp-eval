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

Both constructors return a Phoenix-compatible judge. The recommended
configuration order for applications importing idp-eval is:

1. application-owned config object;
2. explicit constructor keyword arguments;
3. environment variables;
4. optional YAML.

### A. Recommended: application-owned config objects

Construct the concrete backend config from the host application's settings or
secret store and pass it directly:

```python
from idp_eval import create_azure_judge, create_gateway_judge
from idp_eval.judges import AzureJudgeConfig, GatewayJudgeConfig

gateway_config = GatewayJudgeConfig(
    model=settings.gateway_model,
    base_url=settings.gateway_base_url,
    app_id=settings.gateway_app_id,
    idp_auth_url=settings.idp_auth_url,
    idp_client_id=settings.idp_client_id,
    idp_client_secret=settings.idp_client_secret,
    idp_user=settings.idp_user,
    idp_password=settings.idp_password,
    verify_ssl=True,
    timeout=90,
)
gateway_judge = create_gateway_judge(config=gateway_config)

azure_config = AzureJudgeConfig(
    model=settings.azure_model,
    azure_endpoint=settings.azure_endpoint,
    tenant_id=settings.tenant_id,
    client_id=settings.client_id,
    client_secret=settings.client_secret,
    api_version=settings.api_version,
    timeout=180,
    proxy_url=None,
    verify_ssl=True,
    reasoning_effort=None,
)
azure_judge = create_azure_judge(config=azure_config)
```

When `config=` is supplied, idp-eval uses that frozen object directly: it does
not read environment variables or YAML, re-resolve values, or mutate the
object. Passing `config=` together with any individual configuration argument
raises `ValueError`.

### B. Explicit keyword arguments

For smaller integrations, pass all required fields directly:

```python
gateway_judge = create_gateway_judge(
    model="...",
    base_url="...",
    # remaining required gateway fields
)
azure_judge = create_azure_judge(
    model="...",
    azure_endpoint="...",
    # remaining required Azure fields
)
```

### C. Environment variables

Zero-argument constructors can resolve configuration from the environment:

```python
gateway_judge = create_gateway_judge()
azure_judge = create_azure_judge()
```

Gateway variables:

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

Azure variables:

| Variable | Purpose |
| --- | --- |
| `IDP_EVAL_AZURE_MODEL` | Azure deployment name |
| `IDP_EVAL_AZURE_ENDPOINT` | Azure OpenAI endpoint |
| `IDP_EVAL_AZURE_TENANT_ID` | Azure AD tenant |
| `IDP_EVAL_AZURE_CLIENT_ID` | service-principal client ID |
| `IDP_EVAL_AZURE_CLIENT_SECRET` | service-principal secret |
| `IDP_EVAL_AZURE_API_VERSION` | Azure OpenAI API version |
| `IDP_EVAL_AZURE_TIMEOUT` | direct client timeout, default 180 seconds |
| `IDP_EVAL_AZURE_PROXY_URL` | optional HTTP proxy |
| `IDP_EVAL_AZURE_VERIFY_SSL` | TLS verification, default true |
| `IDP_EVAL_AZURE_REASONING_EFFORT` | optional request setting |

### D. Optional YAML

`IDP_EVAL_CONFIG` may point to YAML containing a `judge` section for the gateway
and/or an `azure_judge` section for Azure. `config_path="config.yaml"` can also
be passed explicitly. Without a config object, precedence remains **explicit
keyword argument > environment variable > optional YAML**.

The gateway preserves the established corporate token and translation contract.
`IDP_EVAL_GATEWAY_TIMEOUT` cannot override a shorter upstream Mule timeout.
Direct Azure uses Azure Identity for token caching/refresh and does not traverse
the gateway, so `IDP_EVAL_AZURE_TIMEOUT` controls its direct client timeout.
`reasoning_effort` is sent only when configured; temperature is never added.

> **Security:** Never commit real client secrets, passwords, tokens, endpoints,
> or proxy URLs into notebooks, documentation examples, or config files. Use
> placeholders in repository examples and the host application's
> secret/configuration system in production. Disable TLS verification only in a
> controlled development environment.

Evaluator wiring does not depend on the backend:

```python
coverage = CoverageEvaluator(judge)
framework = EvaluationFramework(judge=judge, evaluators=[coverage])
```

## Comparing Gateway and Azure on the same evaluation case

The repository includes a single-case latency smoke test using the production
judge constructors and final `CoverageEvaluator`:

1. Install or sync the project.
2. Replace the gateway config placeholders in the notebook locally, or populate
   `GatewayJudgeConfig` from your application's settings.
3. Replace the Azure config placeholders locally, or populate
   `AzureJudgeConfig` from application settings. Configure optional reasoning
   effort on that object only when the selected model supports it.
4. Place your local `golden_set_augmented_tagged.csv` where the notebook expects
   it, or update `GOLDEN_SET_PATH`.
5. Open
   [`notebooks/judge_backend_latency_comparison.ipynb`](../notebooks/judge_backend_latency_comparison.ipynb).
6. Run the notebook from top to bottom.
7. Inspect the comparison DataFrame and verbose item-level tables.
8. Run the final cell to close both judge resources.

The notebook constructs both backend config objects explicitly and measures
`time.perf_counter()` around `framework.evaluate(case)`, so latency is the
application-visible end-to-end evaluator time, including the judge call. It
performs one evaluation per backend and is not a statistically meaningful
performance benchmark. Each configured model is printed because the gateway
and Azure configurations may select different deployments. Never commit the
locally substituted configuration values.

`IDP_EVAL_GATEWAY_TIMEOUT` is only the gateway client's timeout and cannot
override a shorter upstream Mule timeout. In contrast,
`IDP_EVAL_AZURE_TIMEOUT` controls the direct Azure client because that path does
not traverse the corporate gateway.

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
)

# Construct the judge using one of the configuration paths in section 2.
judge = gateway_judge
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
| `input` | task/request/query when one exists |
| `context` | authoritative source/reference evidence |
| `output` | generated result being evaluated |
| `instructions` | explicit behavioral/output constraints |
| `retrieved_documents` | ordered retrieval results |
| `case_id` | optional correlation identifier |
| `metadata` | optional non-prompt metadata |

Fields accept strings or nested dictionaries/lists of scalar values. Rendering
is deterministic. Required fields are evaluator-specific:

| Evaluator | Required fields | Optional evidence |
| --- | --- | --- |
| `CoverageEvaluator` | `context`, `output` | — |
| `FaithfulnessEvaluator` | `context`, `output` | — |
| `InstructionAdherenceEvaluator` | `instructions`, `output` | `context` |
| retrieval evaluators | `input`, `retrieved_documents` | — |

For non-chat generation workflows, `input` may legitimately be `None` when no
selected evaluator requires it. It is not a synonym for a system prompt.
Applications that want to evaluate explicit system/developer output constraints
may map those constraints into `instructions`.

Structured values are rendered recursively for prompts. Dictionary keys become
readable labels (`max_latency` becomes `Max Latency`), lists become bullets, and
nested dictionaries retain their hierarchy. Metadata is retained for reporting
but is never rendered into prompts.

```python
case = EvaluationCase(
    context={
        "service_limits": {
            "max_latency": "2 seconds",
            "regions": ["US", "EU"],
        },
        "requirements": ["99.9% availability"],
    },
    output={
        "summary": "Recommended service configuration.",
        "actions": ["Add regional routing", "Monitor availability"],
    },
    metadata={"source": "benchmark"},
)
```

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

## 7. Metric-specific examples

### Coverage

Requires `context + output`; it ignores `input`, `instructions`, metadata, and
retrieved documents. Structured dictionaries/lists are rendered recursively.
Use `CoverageEvaluator(judge, verbose=True)` for item details. See
[`notebooks/coverage_evaluator_usage.ipynb`](../notebooks/coverage_evaluator_usage.ipynb).

### Faithfulness

Requires `context + output`; it ignores `input`, `instructions`, metadata, and
retrieved documents semantically. Structured values use `render_value()` and
remain one evaluation case. Use `FaithfulnessEvaluator(judge, verbose=True)` for
the complete claim audit. See
[`notebooks/faithfulness_evaluator_usage.ipynb`](../notebooks/faithfulness_evaluator_usage.ipynb).

The evaluator makes one structured judge call over rendered context and output.
It identifies materially distinct factual output claims and labels each
`supported` or `unsupported`. Python assigns stable IDs, removes normalized-exact
duplicates, and computes:

```text
faithfulness = supported output claims / total factual output claims
```

Unsupported claims include contradictions, invented facts, unsupported
specificity, changed qualifiers, and certainty stronger than context supports.
Omitted context information is not unfaithful—it belongs to Coverage. Compact
details contain claim counts, timing, `judge_call_count=1`, and `verbose=False`;
verbose details add each claim, status, deterministic item score, and reason.
If no checkable factual claims are identified, the result is `not_applicable`
after that one call. Async evaluation uses native judge async generation when
available, otherwise the same call runs in a worker thread under the framework's
shared concurrency limit.

### Instruction Adherence

`InstructionAdherenceEvaluator(judge, verbose=False)` requires
`instructions + output` and optionally uses `context` as supporting evidence.
It sends their full rendered values in one structured judge call. The judge
identifies materially distinct, independently checkable output constraints and
returns only `followed` or `violated`; Python deduplicates exact normalized
repeats, assigns IDs, and computes:

```text
instruction adherence = followed instructions / identified instructions
```

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

The holistic judge sees the complete structured output, so exact/minimum/maximum
counts, universal `each`/`every`/`all` requirements, prohibitions, required
fields, order, format, language, and style can be judged together. There is no
domain-specific Python rule engine. Context remains evidence only: contextual
facts do not become instructions, and source completeness is not scored unless
an explicit instruction requires it. The metric ignores `input`, metadata, and
retrieved documents.

Compact details contain instruction/followed/violated counts,
`judge_call_count=1`, timing, and `verbose=False`. With `verbose=True`, the
`instructions` audit list adds stable IDs, statuses, scores (1.0 or 0.0), and
concise reasons. Both modes score identically. A valid judge response with no
checkable instructions returns `not_applicable` after one call; missing
`instructions` or `output` fails validation before any judge work. Async
evaluation uses the judge's native async method when available and otherwise
bridges the same single call through a worker thread under the framework's
global semaphore.

See
[`notebooks/instruction_adherence_evaluator_usage.ipynb`](../notebooks/instruction_adherence_evaluator_usage.ipynb)
for the compact structured-data example.

### Retrieval metrics

`RelevanceAtKEvaluator(k)` and `NDCGAtKEvaluator(k)` judge ranked retrieval
documents against the query in `input`. Relevance@K is binary Precision@K;
nDCG@K is derived from the same relevance judgments.

## 8. Bulk, grouped, and async APIs

```python
result = framework.evaluate(case)
results = framework.evaluate_many(cases)

groups = [
    {
        "group_id": "request-1",
        "input": {"operation": "compare options"},
        "context": {
            "requirements": ["Response time under 2 seconds"],
            "constraints": ["Use approved regions"],
        },
        "instructions": ["Keep the summary concise"],
        "outputs": [
            {"summary": "Option A", "actions": ["Add caching"]},
            {"summary": "Option B", "actions": ["Scale workers"]},
        ],
        "case_ids": ["request-1:a", "request-1:b"],
        "metadata": {"source": "benchmark"},
    }
]
results = framework.evaluate_groups(groups)

result = await framework.a_evaluate(case, max_concurrency=4)
results = await framework.a_evaluate_many(cases, max_concurrency=4)
results = await framework.a_evaluate_groups(groups, max_concurrency=4)
```

Groups fan out into ordinary independent cases; evaluator work is not shared.
Shared `input`, `context`, `instructions`, and `metadata` are copied to each case.
Explicit `case_ids` win, followed by IDs derived from `group_id`, then stable
group/output indexes. Inputs are not mutated. Async results preserve order, and
the framework enforces the shared `max_concurrency` limit across all judge calls.

`case.output=[a, b]` is one structured output, one `EvaluationCase`, and one
trace. `group["outputs"]=[a, b]` creates two cases, evaluations, and traces. This
distinction applies equally to coverage, faithfulness, and instruction adherence.

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
