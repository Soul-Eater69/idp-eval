"""CoverageEvaluator tests (whole-source coverage, dag + g_eval modes; no LLM)."""

import json

import pytest

from idp_eval import CoverageEvaluator, EvaluationCase

CASE = EvaluationCase(
    input="TASK_SHOULD_BE_IGNORED",
    context="Retain the identity provider. SSO must remain supported.",
    output="UNIQUE_OUTPUT keeps the identity provider.",
)


# --- fake judges -------------------------------------------------------------


class GEvalJudge:
    """One-call g_eval judge. ``entries``: (text, mp, fp[, reason])."""

    def __init__(self, *entries):
        self._entries = entries
        self.calls = []

    def generate_object(self, prompt, schema):
        self.calls.append({"prompt": prompt, "schema": schema})
        items = []
        for text, mp, fp, *reason in self._entries:
            item = {"source_item": text, "meaningfully_present": mp, "fully_present": fp}
            if reason:
                item["reason"] = reason[0]
            items.append(item)
        return {"items": items}


class RawJudge:
    """Returns a fixed raw response (for malformed-input tests)."""

    def __init__(self, response):
        self._response = response
        self.calls = 0

    def generate_object(self, prompt, schema):
        self.calls += 1
        return self._response


class DagJudge:
    """Two-stage judge: extraction then per-id classification.

    ``statuses``: list of (mp, fp) in item order; item texts are auto-generated.
    Classification maps each requested id (S1..) to its status, so it works across
    safety batches.
    """

    def __init__(self, statuses, *, extract_items=None):
        self._statuses = statuses
        self._extract_items = extract_items
        self.calls = 0
        self.prompts = []

    def generate_object(self, prompt, schema):
        self.calls += 1
        self.prompts.append(prompt)
        user = prompt[1]["content"]
        if "requirements" in schema["properties"]:  # classify
            reqs = json.loads(user.split("[REQUIREMENTS]\n", 1)[1].split("\n\n[OUTPUT]", 1)[0])
            index = {f"S{i + 1}": i for i in range(len(self._statuses))}
            return {"requirements": [
                {"id": r["id"],
                 "meaningfully_present": self._statuses[index[r["id"]]][0],
                 "fully_present": self._statuses[index[r["id"]]][1]}
                for r in reqs
            ]}
        texts = self._extract_items or [f"item {i + 1}" for i in range(len(self._statuses))]
        return {"items": [{"source_item": t} for t in texts]}


def _dag(judge, **kw):
    return CoverageEvaluator(judge, mode="dag", **kw).evaluate(CASE)


def _geval(judge, **kw):
    return CoverageEvaluator(judge, mode="g_eval", **kw).evaluate(CASE)


# --- metric identity / modes ------------------------------------------------


def test_metric_name_is_coverage_for_both_modes():
    assert _dag(DagJudge([(True, True)])).metric == "coverage"
    assert _geval(GEvalJudge(("a", True, True))).metric == "coverage"


def test_default_mode_is_dag():
    assert CoverageEvaluator(object())._mode == "dag"


def test_invalid_mode_raises_listing_allowed_modes():
    with pytest.raises(ValueError, match=r"mode must be one of \('dag', 'g_eval'\)"):
        CoverageEvaluator(object(), mode="bogus")


def test_required_fields_context_output_no_input():
    assert CoverageEvaluator.required_fields == ("context", "output")
    ev = CoverageEvaluator(object())
    ev.validate_case(EvaluationCase(context="c", output="o"))  # no input needed
    with pytest.raises(ValueError, match="requires non-empty `context`"):
        ev.validate_case(EvaluationCase(output="o"))
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        ev.validate_case(EvaluationCase(context="c"))


def test_extra_input_and_instructions_allowed():
    CoverageEvaluator(object()).validate_case(
        EvaluationCase(input="t", context="c", output="o", instructions="i")
    )


# --- DAG ---------------------------------------------------------------------


def test_dag_normal_is_two_calls_and_stage_isolation():
    judge = DagJudge([(True, True), (True, False)])
    result = _dag(judge)
    assert judge.calls == 2
    assert result.details["mode"] == "dag"
    assert result.details["judge_call_count"] == 2
    assert result.details["batch_count"] == 1
    # Stage 1 is context-only (no output); Stage 2 sees output.
    extract_user = judge.prompts[0][1]["content"]
    classify_user = judge.prompts[1][1]["content"]
    assert CASE.context in extract_user and CASE.output not in extract_user
    assert CASE.input not in extract_user
    assert CASE.output in classify_user


def test_dag_scoring_and_labels():
    result = _dag(DagJudge([(True, True), (True, False), (False, False)]))
    assert result.score == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result.label == "incomplete"
    assert result.details["covered_count"] == 1
    assert result.details["partial_count"] == 1
    assert result.details["missing_count"] == 1
    assert result.explanation == "1 of 3 source items are fully covered; 1 partial and 1 missing."


@pytest.mark.parametrize("n,expected", [(8, 2), (10, 2), (12, 2), (20, 2)])
def test_dag_normal_call_count_single_classify(n, expected):
    judge = DagJudge([(True, True)] * n)
    result = _dag(judge)
    assert judge.calls == expected
    assert result.details["batch_count"] == 1
    assert result.details["judge_call_count"] == 2


@pytest.mark.parametrize("n", [21, 25, 30])
def test_dag_safety_batching_over_threshold(n):
    judge = DagJudge([(True, True)] * n)
    result = _dag(judge)
    assert judge.calls == 4  # 1 extract + 3 safety classify batches of 10
    assert result.details["batch_count"] == 3
    assert result.details["judge_call_count"] == 4
    assert result.details["final_item_count"] == n


def test_dag_zero_items_is_not_applicable_one_call():
    judge = DagJudge([])
    result = _dag(judge)
    assert judge.calls == 1
    assert result.score is None and result.label == "not_applicable"
    assert result.details["judge_call_count"] == 1
    assert result.details["batch_count"] == 0
    assert result.details["mode"] == "dag"


def test_dag_stable_ids_and_normalized_dedup():
    judge = DagJudge(
        [(True, True), (False, False)],
        extract_items=["Retain the IdP.", " retain   the idp. ", "Support SSO."],
    )
    result = _dag(judge, verbose=True)
    items = result.details["items"]
    assert [i["id"] for i in items] == ["S1", "S2"]
    assert [i["source_item"] for i in items] == ["Retain the IdP.", "Support SSO."]


def test_dag_compact_omits_items_verbose_includes_them():
    assert "items" not in _dag(DagJudge([(True, True)])).details
    assert "items" in _dag(DagJudge([(True, True)]), verbose=True).details


# --- DAG safety-batch merge integrity ---------------------------------------


class _BadMergeJudge(DagJudge):
    def __init__(self, statuses, corruption):
        super().__init__(statuses)
        self._corruption = corruption

    def generate_object(self, prompt, schema):
        resp = super().generate_object(prompt, schema)
        if "requirements" in resp:
            items = resp["requirements"]
            if self._corruption == "missing" and any(x["id"] == "S25" for x in items):
                resp["requirements"] = [x for x in items if x["id"] != "S25"]
            elif self._corruption == "duplicate" and any(x["id"] == "S1" for x in items):
                resp["requirements"] = items + [dict(items[0])]
            elif self._corruption == "unknown" and any(x["id"] == "S1" for x in items):
                resp["requirements"] = items + [
                    {"id": "S999", "meaningfully_present": True, "fully_present": True}
                ]
        return resp


@pytest.mark.parametrize(
    "corruption,match",
    [("missing", "Missing classification"),
     ("duplicate", "Duplicate requirement id"),
     ("unknown", "Unknown requirement id")],
)
def test_dag_batch_merge_integrity(corruption, match):
    with pytest.raises(ValueError, match=match):
        _dag(_BadMergeJudge([(True, True)] * 25, corruption))


# --- G-Eval ------------------------------------------------------------------


def test_geval_is_one_call():
    judge = GEvalJudge(("a", True, True), ("b", True, False))
    result = _geval(judge)
    assert len(judge.calls) == 1
    assert result.details["mode"] == "g_eval"
    assert result.details["judge_call_count"] == 1


def test_geval_context_and_output_not_input():
    judge = GEvalJudge(("a", True, True))
    _geval(judge)
    user = judge.calls[0]["prompt"][1]["content"]
    assert CASE.context in user and CASE.output in user
    assert CASE.input not in user


def test_geval_scoring_matches_status_mapping():
    result = _geval(GEvalJudge(("a", True, True), ("b", True, False), ("c", False, False)))
    assert result.score == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result.label == "incomplete"


def test_geval_empty_items_not_applicable_after_one_call():
    judge = RawJudge({"items": []})
    result = _geval(judge)
    assert judge.calls == 1
    assert result.score is None and result.label == "not_applicable"
    assert result.details["judge_call_count"] == 1


def test_geval_compact_omits_items_verbose_includes():
    assert "items" not in _geval(GEvalJudge(("a", True, True))).details
    assert "items" in _geval(GEvalJudge(("a", True, True, "")), verbose=True).details


@pytest.mark.parametrize(
    "response,match",
    [({}, "missing `items`"),
     ({"items": None}, "must be a list"),
     ({"items": ["bad"]}, "expected an object"),
     ({"items": [{"source_item": "", "meaningfully_present": True, "fully_present": True}]},
      "non-empty string"),
     ({"items": [{"source_item": "a", "meaningfully_present": "y", "fully_present": False}]},
      "must be booleans")],
)
def test_geval_malformed_raises_clearly(response, match):
    with pytest.raises(ValueError, match=match):
        _geval(RawJudge(response))


def test_geval_impossible_binary_combo_raises():
    with pytest.raises(ValueError, match="Invalid coverage classification"):
        _geval(GEvalJudge(("a", False, True)))


# --- verbose invariance (both modes) ----------------------------------------


def test_dag_verbose_matches_compact_scores_and_statuses():
    compact = _dag(DagJudge([(True, True), (True, False), (False, False)]))
    verbose = _dag(DagJudge([(True, True), (True, False), (False, False)]), verbose=True)
    assert compact.score == verbose.score and compact.label == verbose.label
    assert [i["status"] for i in verbose.details["items"]] == ["covered", "partial", "missing"]


def test_geval_verbose_matches_compact_scores_and_statuses():
    compact = _geval(GEvalJudge(("a", True, True), ("b", True, False), ("c", False, False)))
    verbose = _geval(
        GEvalJudge(("a", True, True, ""), ("b", True, False, "why"), ("c", False, False, "why")),
        verbose=True,
    )
    assert compact.score == verbose.score and compact.label == verbose.label
    assert verbose.details["items"][1]["reason"] == "why"
