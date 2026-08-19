"""Small configuration helpers shared by the two judge backends."""

from __future__ import annotations

import os
from typing import Any


def load_yaml_section(
    config_path: str | None,
    *,
    config_env: str,
    section: str,
) -> dict[str, Any]:
    """Loads one optional YAML mapping."""
    path = config_path or os.environ.get(config_env)
    if not path:
        return {}

    import yaml

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    values = data.get(section, {}) or {}
    if not isinstance(values, dict):
        raise ValueError(f"YAML {section!r} section must be a mapping.")
    return values


def resolve_required_strings(
    *,
    explicit: dict[str, str | None],
    env_vars: dict[str, str],
    yaml_values: dict[str, Any],
    config_name: str,
) -> dict[str, str]:
    """Resolves required strings with explicit > environment > YAML precedence."""
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for field_name, env_name in env_vars.items():
        value = explicit.get(field_name)
        if value is None:
            value = os.environ.get(env_name)
        if value is None:
            value = yaml_values.get(field_name)
        if not isinstance(value, str) or not value:
            missing.append(field_name)
        else:
            resolved[field_name] = value

    if missing:
        raise ValueError(
            f"Missing required {config_name} configuration: " + ", ".join(missing)
        )
    return resolved


def resolve_bool(
    explicit: bool | None,
    *,
    env_name: str,
    yaml_values: dict[str, Any],
    field_name: str,
    default: bool,
) -> bool:
    """Resolves a boolean setting without truthiness surprises."""
    value: Any = explicit
    if value is None:
        value = os.environ.get(env_name)
    if value is None:
        value = yaml_values.get(field_name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value.")


def resolve_positive_float(
    explicit: float | None,
    *,
    env_name: str,
    yaml_values: dict[str, Any],
    field_name: str,
    default: float,
) -> float:
    """Resolves and validates a positive finite timeout."""
    value: Any = explicit
    if value is None:
        value = os.environ.get(env_name)
    if value is None:
        value = yaml_values.get(field_name)
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive number.")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a positive number.") from None
    if result <= 0 or result == float("inf") or result != result:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return result
