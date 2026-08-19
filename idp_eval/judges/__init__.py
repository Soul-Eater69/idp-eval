"""Public judge backends and their shared structural protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from idp_eval.judges.azure import (
    AzureJudge,
    AzureJudgeConfig,
    create_azure_judge,
    resolve_azure_judge_config,
)
from idp_eval.judges.gateway import (
    GatewayJudge,
    GatewayJudgeConfig,
    create_gateway_judge,
    resolve_gateway_judge_config,
)


@runtime_checkable
class Judge(Protocol):
    """Methods used by custom and Phoenix-backed evaluators."""

    model: str

    def generate_object(self, prompt, schema, **kwargs) -> dict[str, Any]: ...

    async def async_generate_object(
        self, prompt, schema, **kwargs
    ) -> dict[str, Any]: ...

    def generate_classification(
        self, prompt, labels, **kwargs
    ) -> dict[str, Any]: ...

    async def async_generate_classification(
        self, prompt, labels, **kwargs
    ) -> dict[str, Any]: ...


__all__ = [
    "AzureJudge",
    "AzureJudgeConfig",
    "GatewayJudgeConfig",
    "GatewayJudge",
    "Judge",
    "create_azure_judge",
    "create_gateway_judge",
    "resolve_azure_judge_config",
    "resolve_gateway_judge_config",
]
