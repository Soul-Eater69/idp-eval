# Holistic LLM Evaluation Framework

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

| Metric           | Question                                                             | Direction          | Better |
| ---------------- | ------------------------------------------------------------------- | ------------------ | ------ |
| `faithfulness`   | Is the output grounded in the context?                              | `output -> context`| higher |
| `hallucination`  | Which claims in the output are unsupported by the context?          | `output -> context`| lower  |
| `input_coverage` | How much important context information is represented in the output?| `context -> output`| higher |

The judge LLM only **classifies semantics** (supported / unsupported, covered /
partial / missing). Deterministic Python in `scoring.py` computes the number.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```python
from idp_eval import (
    EvaluationCase, EvaluationFramework,
    FaithfulnessMetric, HallucinationEvaluator, InputCoverageEvaluator,
)
from idp_eval.phoenix_client import get_judge_llm, register_tracing

register_tracing(project_name="idp-eval")   # once, at startup
judge_llm = get_judge_llm()                        # once

framework = EvaluationFramework(evaluators=[
    FaithfulnessMetric(llm=judge_llm),
    HallucinationEvaluator(llm=judge_llm),
    InputCoverageEvaluator(llm=judge_llm),
])

results = framework.evaluate(EvaluationCase(
    input=user_instruction,
    context=source_context,
    output=generated_output,
))
# results["hallucination"].score -> 0.20
```

Run a subset with `framework.evaluate(case, metrics=["faithfulness", "input_coverage"])`.
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
├── evaluators/          # faithfulness, hallucination, input_coverage
└── prompts/             # versioned judge prompts + JSON schemas
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
- `tests/test_evaluators.py` — evaluators + framework via a `FakeJudge`, no real
  LLM calls.
