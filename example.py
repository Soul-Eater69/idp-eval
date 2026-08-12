"""End-to-end usage example.

Requires Phoenix and a reachable judge LLM. Run tracing registration once at
startup, build the framework once, then evaluate cases.
"""

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessMetric,
    HallucinationEvaluator,
    InputCoverageEvaluator,
)
from idp_eval.phoenix_client import get_judge_llm, register_tracing


def main() -> None:
    # 1. Tracing stays separate from scoring logic.
    register_tracing(project_name="idp-eval")

    # 2. Build the judge once (wire the corporate gateway http client here).
    judge_llm = get_judge_llm()

    # 3. Build the framework once.
    framework = EvaluationFramework(
        evaluators=[
            FaithfulnessMetric(llm=judge_llm),
            HallucinationEvaluator(llm=judge_llm),
            InputCoverageEvaluator(llm=judge_llm),
        ]
    )

    # 4. Describe any generated output with the generic triple.
    case = EvaluationCase(
        input="Generate a feature summary from the provided source.",
        context=(
            "Users can view invoices. Invoices show the total amount due. "
            "Users should receive a confirmation after payment."
        ),
        output=(
            "Users can view invoices, which are stored in AWS S3. "
            "Invoices show the total amount due."
        ),
    )

    # 5. Run everything (or pass metrics=[...] for a subset).
    results = framework.evaluate(case)

    for name, result in results.items():
        print(f"[{name}] score={result.score} label={result.label}")
        print(f"    {result.explanation}")
        if result.details:
            print(f"    details={result.details}")


if __name__ == "__main__":
    main()
