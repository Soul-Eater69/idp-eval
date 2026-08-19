"""Compatibility imports for the renamed corporate gateway constructor."""

from __future__ import annotations

import warnings

from idp_eval.judges.gateway import (
    GatewayJudgeConfig,
    create_gateway_judge,
    resolve_gateway_judge_config,
)

# Temporary compatibility names for existing callers.
JudgeConfig = GatewayJudgeConfig
resolve_judge_config = resolve_gateway_judge_config


def create_judge(**kwargs):
    """Compatibility alias for :func:`create_gateway_judge`."""
    warnings.warn(
        "create_judge() is deprecated; use create_gateway_judge() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_gateway_judge(**kwargs)


__all__ = ["JudgeConfig", "create_judge", "resolve_judge_config"]
