"""Descriptive input is isolated from non-retrieval metric semantics."""

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
)


class RepeatJudge:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_object(self, prompt, schema):
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.response


def _cases():
    common = {
        "context": "Authoritative source.",
        "instructions": "Return one concise statement.",
        "output": "Authoritative source.",
    }
    return (
        EvaluationCase(input="FIRST DESCRIPTIVE TASK", **common),
        EvaluationCase(input="SECOND DESCRIPTIVE TASK", **common),
    )


def test_coverage_accepts_but_does_not_send_descriptive_input():
    judge = RepeatJudge(
        {
            "items": [
                {
                    "source_item": "Authoritative source.",
                    "meaningfully_present": True,
                    "fully_present": True,
                }
            ]
        }
    )
    evaluator = CoverageEvaluator(judge)
    first, second = _cases()
    assert evaluator.evaluate(first).score == evaluator.evaluate(second).score
    assert judge.calls[0]["prompt"] == judge.calls[1]["prompt"]
    assert len(judge.calls) == 2
    assert "FIRST DESCRIPTIVE TASK" not in repr(judge.calls)
    assert "SECOND DESCRIPTIVE TASK" not in repr(judge.calls)


def test_faithfulness_accepts_but_does_not_send_descriptive_input():
    judge = RepeatJudge(
        {"claims": [{"claim": "Authoritative source.", "status": "supported"}]}
    )
    evaluator = FaithfulnessEvaluator(judge)
    first, second = _cases()
    assert evaluator.evaluate(first).score == evaluator.evaluate(second).score
    assert judge.calls[0]["prompt"] == judge.calls[1]["prompt"]
    assert len(judge.calls) == 2
    assert "FIRST DESCRIPTIVE TASK" not in repr(judge.calls)
    assert "SECOND DESCRIPTIVE TASK" not in repr(judge.calls)


def test_instruction_adherence_accepts_but_does_not_send_descriptive_input():
    judge = RepeatJudge(
        {
            "instructions": [
                {
                    "instruction": "Return one concise statement.",
                    "status": "followed",
                }
            ]
        }
    )
    evaluator = InstructionAdherenceEvaluator(judge)
    first, second = _cases()
    assert evaluator.evaluate(first).score == evaluator.evaluate(second).score
    assert judge.calls[0]["prompt"] == judge.calls[1]["prompt"]
    assert len(judge.calls) == 2
    assert "FIRST DESCRIPTIVE TASK" not in repr(judge.calls)
    assert "SECOND DESCRIPTIVE TASK" not in repr(judge.calls)
