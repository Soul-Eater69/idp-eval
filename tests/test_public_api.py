"""Public package surface tests."""

import idp_eval


def test_common_public_api_is_exported():
    expected = {
        "EvaluationCase",
        "EvaluationResult",
        "EvaluationFramework",
        "Evaluator",
        "CoverageEvaluator",
        "FaithfulnessEvaluator",
        "InstructionAdherenceEvaluator",
        "RelevanceAtKEvaluator",
        "NDCGAtKEvaluator",
        "create_gateway_judge",
        "create_azure_judge",
        "register_tracing",
        "PersistenceError",
    }
    assert expected <= set(idp_eval.__all__)
    assert all(hasattr(idp_eval, name) for name in expected)


def test_legacy_and_internal_names_are_not_exported():
    removed = {
        "create_judge",
        "JudgeConfig",
        "SourceCoverageEvaluator",
        "TaskCoverageEvaluator",
        "Judge",
        "GatewayJudgeConfig",
        "AzureJudgeConfig",
        "ANNOTATOR_KINDS",
        "EvaluationRecord",
        "EvaluationWriter",
        "ExcelEvaluationWriter",
        "PhoenixEvaluationWriter",
    }
    assert removed.isdisjoint(idp_eval.__all__)
    assert all(not hasattr(idp_eval, name) for name in removed)
