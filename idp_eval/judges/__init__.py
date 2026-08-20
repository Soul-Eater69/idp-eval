"""Explicit judge backend constructors."""

from __future__ import annotations

from idp_eval.judges.azure import AzureJudgeConfig, create_azure_judge
from idp_eval.judges.gateway import GatewayJudgeConfig, create_gateway_judge

__all__ = [
    "AzureJudgeConfig",
    "GatewayJudgeConfig",
    "create_azure_judge",
    "create_gateway_judge",
]
