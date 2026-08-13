"""End-to-end usage example.

Requires Phoenix and a reachable judge LLM. Run tracing registration once at
startup, build the framework once, then evaluate cases.
"""

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
    create_judge,
    register_tracing,
)


def _print(results) -> None:
    for name, result in results.items():
        print(f"[{name}] score={result.score} label={result.label}")
        print(f"    {result.explanation}")
        if result.details:
            print(f"    details={result.details}")


def main() -> None:
    # 1. Tracing stays separate from judge creation.
    register_tracing(project_name="idp-eval")

    # 2. Configure the judge once (from args / env / optional YAML).
    judge = create_judge()

    # 3. Build the framework from evaluator classes + the shared judge.
    framework = EvaluationFramework(
        evaluators=[
            FaithfulnessEvaluator,
            CoverageEvaluator,
            InstructionAdherenceEvaluator,
        ],
        judge=judge,
    )

    # 4. Faithfulness/coverage: input is the task used to scope the context.
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
    _print(framework.evaluate(case, metrics=["faithfulness", "coverage"]))

    # 5. Instruction adherence: the explicit instructions go in `instructions`.
    instruction_case = EvaluationCase(
        input="Summarize the invoice features.",
        instructions=(
            "Use exactly 3 bullet points.\n"
            "Keep each bullet concise.\n"
            "Do not mention customer names."
        ),
        context="",
        output=(
            "- Invoices are viewable by users.\n"
            "- Each invoice shows the total due.\n"
            "- Payment triggers a confirmation."
        ),
    )
    _print(framework.evaluate(instruction_case, metrics=["instruction_adherence"]))


if __name__ == "__main__":
    main()
