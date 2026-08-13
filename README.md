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

| Metric                  | Question                                      | Direction               | Better |
| ----------------------- | --------------------------------------------- | ----------------------- | ------ |
| `faithfulness`          | Is the output grounded in the context?        | `output -> context`     | higher |
| `coverage`              | How much relevant context reached the output? | `context -> output`     | higher |
| `instruction_adherence` | Did the output obey the supplied explicit instructions? | `instructions -> output` | higher |

Three complementary questions:

- **Faithfulness** — did the output **ADD** unsupported information?
- **Coverage** — did the output **OMIT** important task-relevant source information?
- **Instruction Adherence** — did the output **OBEY** the supplied explicit instructions?

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

### Instruction Adherence

Instruction adherence is a **custom** LLM-as-a-judge metric measuring whether the
output **obeys the explicit instructions** it was given. It uses a versioned
Phoenix-style prompt (`prompts/instruction_adherence.py`). The judge decomposes
the instructions into atomic instructions and classifies each `followed`,
`partial`, `violated`, or `not_applicable`; Python computes the score over the
**applicable** instructions only:

```python
INSTRUCTION_ADHERENCE_VALUES = {"followed": 1.0, "partial": 0.5, "violated": 0.0}
applicable = [i for i in instructions if i["status"] != "not_applicable"]
instruction_adherence = sum(values) / len(applicable)
```

`not_applicable` covers instructions that genuinely do not apply — e.g. a
conditional "If the account is inactive, include a warning." when the context
says the account is active. It is **excluded from the denominator**, not scored
as a success.

For this metric, **`EvaluationCase.input` must contain only the explicit
instruction text to evaluate** — not the full generation prompt. `context` is
optional and consulted only when an instruction requires it (e.g. "only use
information from the context").

The metric returns `score=None`, `label="not_applicable"` when there is nothing
applicable to evaluate, with an explanation distinguishing the reason: no
instructions supplied, no meaningful instructions found, or all supplied
instructions were not applicable. None of these is treated as a perfect or
failing score.

### The generic `input` field

`EvaluationCase` stays generic; the meaning of `input` depends on the metric:

- **faithfulness** — task information passed to Phoenix alongside context/output.
- **coverage** — the task/request used to scope relevant context.
- **instruction_adherence** — the explicit instructions to evaluate.

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
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
)
from idp_eval.phoenix_client import get_judge_llm, register_tracing

register_tracing(project_name="idp-eval")   # once, at startup
judge_llm = get_judge_llm()                  # once

framework = EvaluationFramework(evaluators=[
    FaithfulnessEvaluator(llm=judge_llm),
    CoverageEvaluator(llm=judge_llm),
    InstructionAdherenceEvaluator(llm=judge_llm),
])

results = framework.evaluate(EvaluationCase(
    input=user_instruction,
    context=source_context,
    output=generated_output,
))
# results["coverage"].score -> 0.75
```

Run a subset with `framework.evaluate(case, metrics=["faithfulness", "coverage"])`.

For instruction adherence, put the instruction text in `input`:

```python
case = EvaluationCase(
    input="Use exactly 3 bullet points.\nDo not mention customer names.",
    context=source_context,
    output=generated_output,
)
results = framework.evaluate(case, metrics=["instruction_adherence"])
```

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
├── evaluators/          # faithfulness, coverage, instruction_adherence
└── prompts/             # versioned judge prompts + JSON schemas (custom metrics)
```

## Adding a metric

Implement the `Evaluator` interface and pass it to `EvaluationFramework`. No core
change is required.

```python
class InstructionAdherenceEvaluator(Evaluator):
    name = "instruction_adherence"
    def evaluate(self, case: EvaluationCase) -> EvaluationResult: ...
```

## Testing

```bash
pytest
```

- `tests/test_scoring.py` — scoring logic, no LLM.
- `tests/test_evaluators.py` — evaluators + framework via a `FakeJudge` and a
  fake Phoenix module, no real LLM calls.
