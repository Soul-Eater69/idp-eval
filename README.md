# IDP LLM Evaluation Framework

A reusable evaluation framework on top of [Arize Phoenix](https://github.com/Arize-ai/phoenix)
that can evaluate **any** generated AI output. It is not tied to Jira, RAG,
summarization, or test-case generation. Every evaluation uses the same generic
triple:

```text
input   = what the model was asked to do
context = authoritative source information
output  = generated content being evaluated
```

## Metrics (v1)

| Metric         | Question                                     | Direction          | Better |
| -------------- | -------------------------------------------- | ------------------ | ------ |
| `faithfulness` | Is the output grounded in the context?       | `output -> context`| higher |
| `coverage`     | How much relevant context reached the output?| `context -> output`| higher |

Two complementary questions:

- **Faithfulness** — did the output **ADD** unsupported information?
- **Coverage** — did the output **OMIT** important relevant information?

### Faithfulness vs. hallucination

There is deliberately **no separate `hallucination` metric**.

- **Hallucination** = the failure / problem (the output states things the
  context does not support).
- **Faithfulness** = the metric used to evaluate grounding and detect that
  failure. Higher faithfulness means fewer hallucinated / unsupported additions.

Faithfulness is **Phoenix built-in** (`FaithfulnessEvaluator`) and measures
whether the output **adds unsupported information**.

### Coverage

Coverage is our **custom** LLM-as-a-judge metric. It measures whether the output
**omits material, task-relevant information** from the context:

> How much of the material information in the authoritative context that is
> relevant to satisfying the task (`input`) is represented in the `output`?

It uses a **versioned, Phoenix-style judge prompt** (a system + user message
list in `prompts/coverage.py`, `COVERAGE_PROMPT_V1`). The system message holds
the rubric; the user message holds the `[BEGIN DATA]` block. The judge uses
`input` to scope the task, extracts only material, task-relevant context items
(ignoring boilerplate, IDs, repetition, formatting), judges semantically
(paraphrases count; no fuzzy string matching), and classifies each item as
`covered`, `partial`, or `missing`. Unsupported *additions* are **not** penalized
here — those belong to faithfulness.

The **LLM only classifies** (never returns a number); deterministic Python in
`scoring.py` computes the coverage score:

```python
COVERAGE_VALUES = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
coverage = sum(values) / len(items)
```

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
    FaithfulnessMetric,
)
from idp_eval.phoenix_client import get_judge_llm, register_tracing

register_tracing(project_name="idp-eval")   # once, at startup
judge_llm = get_judge_llm()                  # once

framework = EvaluationFramework(evaluators=[
    FaithfulnessMetric(llm=judge_llm),
    CoverageEvaluator(llm=judge_llm),
])

results = framework.evaluate(EvaluationCase(
    input=user_instruction,
    context=source_context,
    output=generated_output,
))
# results["coverage"].score -> 0.75
```

Run a subset with `framework.evaluate(case, metrics=["faithfulness", "coverage"])`.
See `example.py` for a full runnable script.

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
├── phoenix_client.py    # judge LLM + tracing wiring (one place)
├── evaluators/          # faithfulness, coverage
└── prompts/             # versioned judge prompt + JSON schema (coverage)
```

## Adding a metric

Implement the `Evaluator` interface and pass it to `EvaluationFramework`. No core
change is required.

```python
class InstructionFollowingEvaluator(Evaluator):
    name = "instruction_following"
    def evaluate(self, case: EvaluationCase) -> EvaluationResult: ...
```

## Testing

```bash
pytest
```

- `tests/test_scoring.py` — scoring logic, no LLM.
- `tests/test_evaluators.py` — evaluators + framework via a `FakeJudge` and a
  fake Phoenix module, no real LLM calls.
