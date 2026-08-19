"""Grouped DAG coverage: shared extraction reuse (sync + async), no LLM."""

import asyncio
import json
import threading
import time
import warnings

import pytest

from idp_eval import CoverageEvaluator, EvaluationCase, EvaluationFramework


class DagGroupJudge:
    """DAG judge: extraction (fixed items) + output-sensitive classification.

    Marks every item covered iff the OUTPUT contains ``covered_token``. Counts
    extract vs classify calls and tracks peak concurrency (for the async test).
    """

    def __init__(self, *, n_items=3, delay=0.0, covered_token="GOOD"):
        self.n_items = n_items
        self.delay = delay
        self.covered_token = covered_token
        self.extract_calls = 0
        self.classify_calls = 0
        self._lock = threading.Lock()
        self._current = 0
        self.max_concurrent = 0
        self.order: list[str] = []

    def generate_object(self, prompt, schema):
        user = prompt[1]["content"]
        is_classify = "requirements" in schema["properties"]
        with self._lock:
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
            if is_classify:
                self.classify_calls += 1
            else:
                self.extract_calls += 1
            self.order.append("classify" if is_classify else "extract")
        try:
            if self.delay:
                time.sleep(self.delay)
            if not is_classify:
                return {"items": [{"source_item": f"item {k}"} for k in range(self.n_items)]}
            reqs = json.loads(user.split("[REQUIREMENTS]\n", 1)[1].split("\n\n[OUTPUT]", 1)[0])
            covered = self.covered_token in user.split("[OUTPUT]\n", 1)[1]
            return {"requirements": [
                {"id": r["id"], "meaningfully_present": covered, "fully_present": covered}
                for r in reqs
            ]}
        finally:
            with self._lock:
                self._current -= 1


def _framework(judge):
    return EvaluationFramework(evaluators=[CoverageEvaluator(judge, mode="dag")], judge=judge)


def _group(outputs, gid="A", context="Theme A"):
    return {"context": context, "outputs": outputs, "group_id": gid}


# --- sync grouped reuse (item 32) -------------------------------------------


def test_one_group_three_outputs_extracts_once():
    judge = DagGroupJudge()
    results = _framework(judge).evaluate_groups([_group(["GOOD 1", "BAD 2", "GOOD 3"])])
    assert judge.extract_calls == 1          # shared extraction, not 3
    assert judge.classify_calls == 3         # one per output
    assert [r["coverage"].score for r in results] == [1.0, 0.0, 1.0]  # order preserved
    d = results[0]["coverage"].details
    assert d["shared_extraction"] is True
    assert d["classification_calls"] == 1
    assert d["judge_call_count"] == 1        # per output: classify only (not 2)


def test_two_groups_each_extract_once():
    judge = DagGroupJudge()
    _framework(judge).evaluate_groups([
        _group(["GOOD 1", "GOOD 2"], gid="A", context="A"),
        _group(["GOOD 3", "BAD 4"], gid="B", context="B"),
    ])
    assert judge.extract_calls == 2          # one per group, not reused across groups
    assert judge.classify_calls == 4


def test_single_output_group_does_not_reuse():
    judge = DagGroupJudge()
    _framework(judge).evaluate_groups([_group(["GOOD 1"], gid="A")])
    # One output -> normal extract + classify (no shared-extraction result flag).
    assert judge.extract_calls == 1 and judge.classify_calls == 1


def test_one_bad_output_propagates_without_poisoning_others():
    class Boom(DagGroupJudge):
        def generate_object(self, prompt, schema):
            resp = super().generate_object(prompt, schema)
            if "requirements" in schema["properties"] and "BOOM" in \
                    prompt[1]["content"].split("[OUTPUT]\n", 1)[1]:
                raise RuntimeError("gateway timeout")
            return resp

    judge = Boom()
    with pytest.raises(RuntimeError, match="gateway timeout"):
        _framework(judge).evaluate_groups([_group(["GOOD 1", "BOOM 2"])])
    assert judge.extract_calls == 1  # extraction still shared, not retried


# --- async grouped reuse (item 33) ------------------------------------------


def test_async_grouped_extract_once_and_overlap():
    judge = DagGroupJudge(delay=0.05)
    results = asyncio.run(
        _framework(judge).a_evaluate_groups([_group(["GOOD 1", "BAD 2", "GOOD 3"])],
                                            max_concurrency=4)
    )
    assert judge.extract_calls == 1 and judge.classify_calls == 3
    assert [r["coverage"].score for r in results] == [1.0, 0.0, 1.0]  # order preserved
    # Extraction happened first; the classifications overlapped afterwards.
    assert judge.order[0] == "extract"
    assert judge.order.count("extract") == 1
    assert judge.max_concurrent >= 2


def test_async_grouped_respects_max_concurrency():
    judge = DagGroupJudge(delay=0.05)
    asyncio.run(
        _framework(judge).a_evaluate_groups(
            [_group([f"GOOD {i}" for i in range(6)])], max_concurrency=2
        )
    )
    assert judge.max_concurrent <= 2


def test_async_group1_extraction_not_reused_for_group2():
    judge = DagGroupJudge(delay=0.02)
    asyncio.run(_framework(judge).a_evaluate_groups([
        _group(["GOOD 1", "GOOD 2"], gid="A", context="A"),
        _group(["GOOD 3", "GOOD 4"], gid="B", context="B"),
    ]))
    assert judge.extract_calls == 2


# --- backward-compat deprecated alias ---------------------------------------


def test_source_coverage_alias_is_deprecated_but_works():
    from idp_eval import SourceCoverageEvaluator

    class OneCall:
        def generate_object(self, prompt, schema):
            return {"items": [{"source_item": "a", "meaningfully_present": True,
                               "fully_present": True}]}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev = SourceCoverageEvaluator(OneCall())
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    result = ev.evaluate(EvaluationCase(context="c", output="o"))
    assert result.metric == "source_coverage"   # legacy metric name preserved
    assert result.details["mode"] == "g_eval"    # historical one-call default
    assert result.score == 1.0
