"""Unit tests for deterministic structured-value rendering (no LLM)."""

import pytest

from idp_eval.rendering import (
    is_empty_value,
    render_value,
    validate_structured_value,
)


# --- scalars ----------------------------------------------------------------


def test_string_passthrough():
    assert render_value("Improve onboarding") == "Improve onboarding"
    assert render_value("multi\nline") == "multi\nline"  # unchanged


def test_integer_and_float():
    assert render_value(25) == "25"
    assert render_value(1.5) == "1.5"


def test_bool_renders_predictably():
    assert render_value(True) == "true"
    assert render_value(False) == "false"


def test_none_renders_empty():
    assert render_value(None) == ""


# --- dicts ------------------------------------------------------------------


def test_flat_dict_with_snake_case_labels():
    rendered = render_value(
        {
            "description": "Improve onboarding",
            "business_needs": [
                "Reduce onboarding time by 25%",
                "Retain existing IdP",
            ],
        }
    )
    assert rendered == (
        "Description: Improve onboarding\n"
        "\n"
        "Business Needs:\n"
        "- Reduce onboarding time by 25%\n"
        "- Retain existing IdP"
    )


def test_snake_case_label_and_acronym_preserved():
    # Scalar dict values render inline as ``Label: value``.
    assert render_value({"success_criteria": "x"}) == "Success Criteria: x"
    # Uppercase acronym tokens in keys are preserved.
    assert render_value({"SSO": "on"}) == "SSO: on"
    # Value text (with acronyms) is never altered.
    assert "IdP" in render_value({"note": "Keep IdP"})


def test_nested_dict_is_hierarchical():
    assert render_value({"metadata": {"priority": "high"}}) == (
        "Metadata:\n  Priority: high"
    )


# --- lists ------------------------------------------------------------------


def test_list_of_strings_is_bullet_list():
    assert render_value(["Step 1", "Step 2", "Step 3"]) == (
        "- Step 1\n- Step 2\n- Step 3"
    )


def test_nested_list_deterministic():
    assert render_value([["a", "b"], ["c"]]) == (
        "- - a\n  - b\n- - c"
    )


def test_dict_containing_list():
    assert render_value({"constraints": ["Keep IdP", "Support SSO"]}) == (
        "Constraints:\n- Keep IdP\n- Support SSO"
    )


def test_list_containing_dict():
    # Compact: bullet carries the first (inline) key/value line.
    assert render_value([{"a": "1"}, {"b": "2"}]) == "- A: 1\n- B: 2"


def test_list_of_multi_key_dict_aligns_under_bullet():
    # Later keys keep their indent so they stay grouped under the bullet.
    assert render_value([{"title": "Epic", "priority": "high"}]) == (
        "- Title: Epic\n\n  Priority: high"
    )


def test_dict_with_nested_dict_containing_list():
    assert render_value({"plan": {"steps": ["one", "two"]}}) == (
        "Plan:\n  Steps:\n  - one\n  - two"
    )


def test_deeply_nested_mixed_generic_data():
    value = {
        "service_limits": {
            "max_latency": "2 seconds",
            "regions": ["US", "EU"],
            "controls": [
                {"enabled": True, "threshold": 3},
                {"enabled": False, "note": None},
            ],
        }
    }

    rendered = render_value(value)

    assert "Service Limits:" in rendered
    assert "Max Latency: 2 seconds" in rendered
    assert "Regions:" in rendered
    assert "- US" in rendered and "- EU" in rendered
    assert "Enabled: true" in rendered
    assert "Threshold: 3" in rendered
    assert "Enabled: false" in rendered


# --- order & determinism ----------------------------------------------------


def test_dict_insertion_order_preserved():
    rendered = render_value({"zulu": "1", "alpha": "2", "mike": "3"})
    assert rendered == "Zulu: 1\n\nAlpha: 2\n\nMike: 3"


def test_rendering_is_deterministic():
    value = {
        "title": "Epic",
        "success_criteria": ["a", "b"],
        "metadata": {"priority": "high"},
    }
    assert render_value(value) == render_value(value)


# --- empty structures -------------------------------------------------------


def test_empty_dict_and_list():
    assert render_value({}) == ""
    assert render_value([]) == ""
    # An empty container under a label still shows the label deterministically.
    assert render_value({"items": []}) == "Items:"


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize("bad", [{"a", "b"}, ("x",), b"bytes", object()])
def test_unsupported_object_raises_useful_error(bad):
    with pytest.raises(TypeError, match="Unsupported value type"):
        render_value(bad)


def test_non_string_dict_key_raises():
    with pytest.raises(TypeError, match="keys must be strings"):
        validate_structured_value({1: "a"})


def test_error_names_the_path():
    with pytest.raises(TypeError, match=r"context\.items\[0\]"):
        validate_structured_value({"items": [set()]}, "context")


# --- is_empty_value ---------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   \n\t", {}, []])
def test_empty_values(value):
    assert is_empty_value(value) is True


@pytest.mark.parametrize("value", [0, False, "x", {"a": 1}, [1], 0.0])
def test_present_values(value):
    assert is_empty_value(value) is False
