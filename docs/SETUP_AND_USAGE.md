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
    ContextualPrecisionAtKEvaluator,
    ContextualRecallEvaluator,
    ContextualRelevancyEvaluator,
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessEvaluator,
    HitRateAtKEvaluator,
    InstructionAdherenceEvaluator,
    MRRAtKEvaluator,
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
    judge=judge,
    evaluators=[CoverageEvaluator(max_items=5, verbose=True)]
)
```

Configured built-in LLM evaluator instances may be created before a judge is
available. The framework binds its shared judge when the instance has no
explicit judge; an explicitly supplied evaluator judge takes precedence.

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
| `evaluation_scope` | `combined` (default), `individual`, or `both` for top-level list outputs |

Fields accept strings or nested dictionaries/lists of scalar values. Rendering
is deterministic. Required fields are evaluator-specific:

| Evaluator | Semantic fields | Optional evidence | Descriptive input |
| --- | --- | --- | --- |
| `CoverageEvaluator` | `context`, `output` | — | allowed; ignored for scoring |
| `FaithfulnessEvaluator` | `context`, `output` | — | allowed; ignored for scoring |
| `FewShotContentLeakageEvaluator` | `context`, `retrieved_documents`, `output` | — | allowed; ignored for scoring |
| `InstructionAdherenceEvaluator` | `instructions`, `output` | `context` | allowed; ignored for scoring |
| document retrieval evaluators and Contextual Relevancy | `input`, `retrieved_documents` | — | `input` is semantic |
| `ContextualRecallEvaluator` | `input`, `context`, `retrieved_documents` | — | `input` is semantic |

For non-chat generation workflows, `input` may legitimately be `None` when no
selected evaluator requires it. It is not a synonym for a system prompt.
Applications that want to evaluate explicit system/developer output constraints
may map those constraints into `instructions`.

Available on `EvaluationCase` does not mean used by every metric. For example:

```python
case = EvaluationCase(
    input="Generate three release recommendations.",
    context=source,
    instructions="Return exactly three recommendations.",
    output=generated,
)
```

Coverage evaluates `context + output`; Faithfulness evaluates
`output + context`; Instruction Adherence evaluates `instructions + output`
with optional context evidence. For those metrics, `input` remains descriptive
for tracing and reporting and is not sent to the judge. Retrieval metrics use
`input` semantically as their query.

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

### Selecting metrics per evaluation

`framework.metrics` lists the configured metric names. Pass `metrics=[...]` to
run only a configured subset; unknown names raise `KeyError` before judge work.
Validation applies only to the selected metrics, so a case without
`instructions` can still run Coverage or Faithfulness.

```python
framework.metrics

results = framework.evaluate(
    case,
    metrics=["coverage", "faithfulness"],
)

results = await framework.a_evaluate(
    case,
    metrics=["coverage"],
    max_concurrency=4,
)
```

Metric selection also applies to `individual`/`both` output scopes and all bulk,
grouped, and async methods. It never instantiates an evaluator that was not
configured. See the recommended combined walkthrough in
[`notebooks/core_metrics_together_usage.ipynb`](../notebooks/core_metrics_together_usage.ipynb).

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

`CoverageEvaluator(judge, verbose=False, max_items=None,
reason_mode="overall")` makes exactly one
structured call using `context + output`. The judge identifies materially distinct source
items and returns two booleans:

- `fully_present=True` becomes `covered` and 1.0.
- `meaningfully_present=True, fully_present=False` becomes `partial` and 0.5.
- both false becomes `missing` and 0.0.

Python calculates the mean and label. An empty item set returns `score=None`
and `label="not_applicable"` after the one call.

By default there is no item-count limit. Set `max_items=5` to extract at most
five material, representative source items; the judge returns fewer when fewer
meaningful items exist and never pads to the limit. The prompt preserves
qualifiers, examines the complete context before selection, and keeps selection
independent of covered/partial/missing status. It consolidates semantic
redundancy, excludes structural/meta wrapper text, and avoids counting
both an umbrella objective and equivalent detailed items. Generic topical
overlap is insufficient for partial credit.

The item cap controls selection count, not atomicity: independent source facts
are never merged merely to fit the cap, and the judge never pads. The structured
schema does not use provider-sensitive JSON Schema `maxItems`; Python rejects
over-limit responses without silently truncating them.

Reason generation is independent from `verbose`:

- `reason_mode="overall"` (recommended) returns one semantic explanation and
  keeps concise diagnostics on partial/missing units for audit use.
- `reason_mode="per_item"` requires a concise reason for every source item and
  also returns one semantic overall explanation.
- `reason_mode="none"` returns no reasons and sets `explanation=None`.

All modes classify every selected unit in the same single judge call. The LLM
does not return score or label; Python derives both. Overall explanations
intentionally omit scores, percentages, status counts, and count summaries.

Compact details:

```python
{
    "final_item_count": 3,
    "max_items": None,
    "evaluated_items": 3,
    "covered_count": 2,
    "partial_count": 0,
    "missing_count": 1,
    "judge_call_count": 1,
    "total_ms": 123.4,
    "verbose": False,
}
```

With `verbose=True`, `details["items"]` adds stable IDs, source text, the
binary judgments, Python-derived status and score, and any reasons generated by
the configured `reason_mode`. `verbose` controls detail exposure only.

## 7. Metric-specific examples

For a complete shared-judge example that runs all three core metrics, selects a
subset per call, inspects verbose details, and combines scope with bulk/async
evaluation, start with
[`notebooks/core_metrics_together_usage.ipynb`](../notebooks/core_metrics_together_usage.ipynb).

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
verbose details add each claim, status, deterministic item score, and any
generated internal reason.
If no checkable factual claims are identified, the result is `not_applicable`
after that one call. `max_items=5` limits extraction to at most five material,
representative claims without padding. The complete output is examined before
selection, and supported/unsupported status does not influence which claims are
selected. Score 1.0 is labeled
`not_hallucinated`; any lower scored result is `hallucinated`.
Async evaluation uses native judge async generation when available; otherwise
the same call runs in a worker thread under the framework's shared concurrency
limit.

Faithfulness supports the same reason modes as Coverage. In the recommended
`overall` mode, supported claims need no semantic item reason while unsupported
claims retain concise internal diagnostics; the visible explanation is the one
semantic `overall_reason`. `per_item` requires a reason for every claim, and
`none` produces no reasons or explanation. `max_items=None` evaluates all
materially distinct atomic claims. A finite cap selects representative atomic
claims without merging independent claims, uses no schema `maxItems`, and is
validated in Python. Every configuration still uses one judge call.

Recommended production configuration:

```python
framework = EvaluationFramework(
    judge=judge,
    evaluators=[
        CoverageEvaluator(max_items=None, reason_mode="overall"),
        FaithfulnessEvaluator(max_items=None, reason_mode="overall"),
    ],
)
```

### Few-shot content leakage

`FewShotContentLeakageEvaluator` requires `context`, `retrieved_documents`, and
`output`. Current `context` is authoritative evidence for the generation;
`retrieved_documents` are historical few-shot examples and are explicitly
non-authoritative for current business facts. In one judge call, the evaluator
extracts output claims and independently judges current-context and example
support. Python counts only claims supported by examples but not current context:

```text
few_shot_content_leakage = example_only claims / evaluated output claims
```

The result is an example-only content-overlap/leakage signal, not strict causal
proof that the model copied an example. Use `verbose=True` to expose claim-level
source classifications.

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
for the practical guide, including collection-level and per-item scope.

### Retrieval and context metrics

No generated output is required. List order in `retrieved_documents` is rank.

| Evaluator | Unit | Question | Required fields |
| --- | --- | --- | --- |
| `RelevanceAtKEvaluator(k)` | document | How many retrieved documents are relevant? | `input`, `retrieved_documents` |
| `HitRateAtKEvaluator(k)` | ranked list | Was any relevant document retrieved? | `input`, `retrieved_documents` |
| `MRRAtKEvaluator(k)` | ranked list | How early is the first relevant document? | `input`, `retrieved_documents` |
| `NDCGAtKEvaluator(k)` | ranked document | How close is the relevance order to ideal? | `input`, `retrieved_documents` |
| `ContextualRelevancyEvaluator()` | context item | How much retrieved content is useful? | `input`, `retrieved_documents` |
| `ContextualPrecisionAtKEvaluator(k)` | ranked document | Are relevant documents ranked high? | `input`, `retrieved_documents` |
| `ContextualRecallEvaluator()` | reference item | How much useful reference information was retrieved? | `input`, `context`, `retrieved_documents` |

Contextual Relevancy evaluates materially distinct information inside retrieved
text, not the fraction of whole documents marked relevant. Contextual
Precision@K accumulates precision at ranks containing relevant documents; it is
AP-style ranking quality over the evaluated list and does not assume corpus-wide
relevance labels. nDCG instead applies logarithmic rank discount and compares
against the ideal ordering. Contextual Recall treats `context` as
authoritative/gold information and asks how much query-relevant reference
information appears somewhere in retrieval. It is distinct from Coverage,
which compares authoritative context with generated output.

`effective_k = min(k, document_count)`. For document ranking metrics and
Contextual Relevancy, an empty list returns `not_applicable` without a judge
call. Contextual Recall still makes its one holistic call so it can identify
relevant reference items and return `0.0` when they exist but none were
retrieved. `None` or a non-list value fails validation. Documents may be strings
or mappings using the default `text` key (configurable with
`document_text_key`). Only rendered semantic fields and document text reach the
judge; `document_id`, retriever `score`, and metadata remain diagnostics.

```python
framework = EvaluationFramework(
    judge=judge,
    evaluators=[
        RelevanceAtKEvaluator(k=5),
        HitRateAtKEvaluator(k=5),
        MRRAtKEvaluator(k=5),
        NDCGAtKEvaluator(k=5),
        ContextualPrecisionAtKEvaluator(k=5),
        ContextualRelevancyEvaluator(),
        ContextualRecallEvaluator(),
    ],
)
results = framework.evaluate(case)
```

The four existing document metrics plus Contextual Precision share one relevance
call through the deepest selected effective K. Contextual Relevancy and
Contextual Recall each add one holistic call, so selecting all seven metrics
makes three semantic calls total. Native async structured generation is
preferred; each holistic call consumes one slot from the framework's shared
concurrency limit. Tracing uses `retrieval.relevance.evaluate`,
`contextual_relevancy.evaluate`, and `contextual_recall.evaluate` stages—not one
span per document or item.
See [`notebooks/retrieval_metrics_usage.ipynb`](../notebooks/retrieval_metrics_usage.ipynb).

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

### Resilient and resumable batches

Use Excel as the checkpoint for long-running provider-backed evaluations:

```python
framework = EvaluationFramework(
    judge=judge,
    evaluators=[
        CoverageEvaluator(max_items=5, verbose=True),
        FaithfulnessEvaluator(max_items=5, verbose=True),
    ],
    output="excel",
    excel_path="generation_eval.xlsx",
    resume=True,
    report_fields=["context", "output", "metadata.theme_id"],
)

case = EvaluationCase(
    case_id="EPIC-1234",
    metadata={"theme_id": "THEME-42"},
    context=source,
    output=generation,
)

results = framework.evaluate_many(
    cases,
    metrics=["coverage", "faithfulness"],
    run_name="generation_eval_v1",
    dataset_name="dataset.parquet",
    on_error="continue",
    retry_until_complete=True,
    retry_interval_seconds=180,
    show_progress=True,
)

results = await framework.a_evaluate_many(
    cases,
    metrics=["coverage", "faithfulness"],
    run_name="generation_eval_v1",
    dataset_name="dataset.parquet",
    max_concurrency=2,
    on_error="continue",
    show_progress=True,
)
```

The default `on_error="raise"` remains fail-fast. With `"continue"`, only
recognized exhausted rate-limit, provider 5xx, timeout, and connection failures
become `score=None`, `label="error"` results. Validation, malformed judge
schemas, implementation bugs, and persistence failures still raise. The
provider retry layer remains unchanged. Framework-level retry rounds are off by
default; `retry_until_complete=True` requires `on_error="continue"`,
`resume=True`, and Excel output. The framework waits
`retry_interval_seconds` only between completed rounds, skips checkpointed
successes, and reruns operational failures until completion or interruption.

Successful metric rows (including `not_applicable`) are saved incrementally.
Rerunning with the same Excel path and exact run, dataset, case content, metric,
and evaluator configuration reconstructs those results without judge calls;
missing/error metrics rerun and replace their prior rows. Async evaluation
serializes writer mutation while preserving concurrent judge work and
input-ordered returns.

By default, `case.output=[a, b]` is one structured output and one trace. For a
single generation containing multiple output objects, orchestration can be
selected directly:

```python
case = EvaluationCase(
    input="Generate two options from the supplied source.",
    context=source,
    output=[option_a, option_b],
    evaluation_scope="both",
)
results = framework.evaluate(case)

combined_results = results["combined"]
individual_results = results["individual"]
```

- `combined` preserves the existing flat metric-result mapping and evaluates the
  whole list once.
- `individual` returns `combined=None` plus one metric-result mapping per item.
- `both` returns the whole-list result plus every item result.

`individual` and `both` require a non-empty top-level list; dictionaries are not
implicitly fanned out. Every expanded item receives the same input, context,
instructions, metadata, and retrieved documents, and gets its own root trace.
When the parent has a case ID, item IDs are `case-id:0`, `case-id:1`, and so on.
This orchestration applies uniformly to coverage, faithfulness, instruction
adherence, retrieval metrics, and custom evaluators. `evaluate_groups()` remains
available for externally grouped records and explicit per-output IDs.

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
| `evaluations` | published case/metric result (upserted when resuming) |
| `_idp_eval_checkpoint` | hidden technical resume/result state |
| `coverage_items` | verbose coverage item |
| `few_shot_content_leakage_items` | verbose leakage claim judgment |
| `instruction_adherence_items` | instruction item |
| `retrieval_documents` | retrieval document |
| `contextual_relevancy_items` | verbose retrieved-content item |
| `contextual_recall_items` | verbose reference-item capture judgment |

The visible `evaluations` summary displays Python `case_id` as `key_id` and
includes the ordered case fields selected by `report_fields` (default: `input`,
`context`, `output`, `instructions`). `retrieved_documents` is available
explicitly but omitted by default because it can be large. Select one-level
metadata with `metadata.<key>`; metadata remains reporting-only. Technical
fingerprints, trace IDs, and annotator kinds live in the hidden
`_idp_eval_checkpoint` sheet. Nested details remain JSON on the summary sheet.
Persistence failures retain
computed results and never rerun evaluation. Existing workbooks that predate the
fingerprint/status schema are rejected for `resume=True` rather than matched
unsafely by case ID.
Built-in fingerprints include evaluator configuration and safe judge
type/model/provider/client identity; endpoint and authentication values are
excluded.
Changing `max_items` on either evaluator changes the evaluation fingerprint;
changing `report_fields` does not. A resumed workbook must nevertheless use the
same visible report schema, or construction fails before judge work.
Custom evaluators with score-affecting options should override
`resume_signature()` and return compact JSON-safe configuration values.

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
| `1.0` | `covered` |
| `0 < score < 1` | `partial` |
| `0.0` | `missing` |
| `None` | `not_applicable` |

Faithfulness uses `not_hallucinated` only at 1.0, `hallucinated` for every
score below 1.0, and `not_applicable` when no checkable claims exist.

## 12. Verification

```bash
uv run pytest -q
python3 -m compileall -q idp_eval
```

Unit tests use fake judges and do not call a live model or gateway.
