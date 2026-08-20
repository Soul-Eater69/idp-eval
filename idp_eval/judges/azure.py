"""Direct Azure OpenAI judge backend using Azure AD authentication."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Any

from idp_eval.judges._config import (
    load_yaml_section,
    resolve_bool,
    resolve_positive_float,
    resolve_required_strings,
)

_AZURE_SCOPE = "https://cognitiveservices.azure.com/.default"
_CONFIG_ENV = "IDP_EVAL_CONFIG"
_VERIFY_SSL_ENV = "IDP_EVAL_AZURE_VERIFY_SSL"
_TIMEOUT_ENV = "IDP_EVAL_AZURE_TIMEOUT"
_PROXY_ENV = "IDP_EVAL_AZURE_PROXY_URL"
_REASONING_EFFORT_ENV = "IDP_EVAL_AZURE_REASONING_EFFORT"
_ENV_VARS = {
    "model": "IDP_EVAL_AZURE_MODEL",
    "azure_endpoint": "IDP_EVAL_AZURE_ENDPOINT",
    "tenant_id": "IDP_EVAL_AZURE_TENANT_ID",
    "client_id": "IDP_EVAL_AZURE_CLIENT_ID",
    "client_secret": "IDP_EVAL_AZURE_CLIENT_SECRET",
    "api_version": "IDP_EVAL_AZURE_API_VERSION",
}


@dataclass(frozen=True)
class AzureJudgeConfig:
    """Resolved direct Azure OpenAI configuration."""

    model: str
    azure_endpoint: str
    tenant_id: str
    client_id: str
    client_secret: str = field(repr=False)
    api_version: str
    timeout: float = 180.0
    proxy_url: str | None = None
    verify_ssl: bool = True
    reasoning_effort: str | None = None


def _resolve_optional_string(
    explicit: str | None,
    *,
    env_name: str,
    yaml_values: dict[str, Any],
    field_name: str,
) -> str | None:
    value: Any = explicit
    if value is None:
        value = os.environ.get(env_name)
    if value is None:
        value = yaml_values.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when set.")
    return value


def resolve_azure_judge_config(
    *,
    model: str | None = None,
    azure_endpoint: str | None = None,
    tenant_id: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    api_version: str | None = None,
    timeout: float | None = None,
    proxy_url: str | None = None,
    verify_ssl: bool | None = None,
    reasoning_effort: str | None = None,
    config_path: str | None = None,
) -> AzureJudgeConfig:
    """Resolves Azure values using explicit > environment > YAML precedence."""
    yaml_values = load_yaml_section(
        config_path,
        config_env=_CONFIG_ENV,
        section="azure_judge",
    )
    resolved = resolve_required_strings(
        explicit={
            "model": model,
            "azure_endpoint": azure_endpoint,
            "tenant_id": tenant_id,
            "client_id": client_id,
            "client_secret": client_secret,
            "api_version": api_version,
        },
        env_vars=_ENV_VARS,
        yaml_values=yaml_values,
        config_name="Azure judge",
    )
    return AzureJudgeConfig(
        **resolved,
        timeout=resolve_positive_float(
            timeout,
            env_name=_TIMEOUT_ENV,
            yaml_values=yaml_values,
            field_name="timeout",
            default=180.0,
        ),
        proxy_url=_resolve_optional_string(
            proxy_url,
            env_name=_PROXY_ENV,
            yaml_values=yaml_values,
            field_name="proxy_url",
        ),
        verify_ssl=resolve_bool(
            verify_ssl,
            env_name=_VERIFY_SSL_ENV,
            yaml_values=yaml_values,
            field_name="verify_ssl",
            default=True,
        ),
        reasoning_effort=_resolve_optional_string(
            reasoning_effort,
            env_name=_REASONING_EFFORT_ENV,
            yaml_values=yaml_values,
            field_name="reasoning_effort",
        ),
    )


class AzureJudge:
    """Phoenix-compatible judge with Azure request options and cleanup."""

    def __init__(self, llm, credential, reasoning_effort: str | None):
        self._llm = llm
        self._credential = credential
        self._reasoning_effort = reasoning_effort
        self._closed = False

    @property
    def model(self):
        return self._llm.model

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        request_kwargs = dict(kwargs)
        if self._reasoning_effort is not None:
            request_kwargs.setdefault(
                "reasoning_effort", self._reasoning_effort
            )
        return request_kwargs

    def generate_object(self, prompt, schema, **kwargs):
        return self._llm.generate_object(
            prompt,
            schema,
            **self._request_kwargs(kwargs),
        )

    async def async_generate_object(self, prompt, schema, **kwargs):
        return await self._llm.async_generate_object(
            prompt,
            schema,
            **self._request_kwargs(kwargs),
        )

    def generate_classification(self, prompt, labels, **kwargs):
        return self._llm.generate_classification(
            prompt,
            labels,
            **self._request_kwargs(kwargs),
        )

    async def async_generate_classification(self, prompt, labels, **kwargs):
        return await self._llm.async_generate_classification(
            prompt,
            labels,
            **self._request_kwargs(kwargs),
        )

    def close(self) -> None:
        """Closes the synchronous OpenAI client and Azure credential."""
        if self._closed:
            return
        sync_client = getattr(self._llm, "_sync_client", None)
        close = getattr(sync_client, "close", None)
        if callable(close):
            close()
        credential_close = getattr(self._credential, "close", None)
        if callable(credential_close):
            credential_close()
        self._closed = True

    async def aclose(self) -> None:
        """Closes both Phoenix-managed OpenAI clients and the credential."""
        if self._closed:
            return
        async_client = getattr(self._llm, "_async_client", None)
        async_close = getattr(async_client, "close", None)
        if callable(async_close):
            result = async_close()
            if inspect.isawaitable(result):
                await result
        sync_client = getattr(self._llm, "_sync_client", None)
        close = getattr(sync_client, "close", None)
        if callable(close):
            close()
        credential_close = getattr(self._credential, "close", None)
        if callable(credential_close):
            credential_close()
        self._closed = True

    def __getattr__(self, name: str):
        return getattr(self._llm, name)


def create_azure_judge(
    *,
    config: AzureJudgeConfig | None = None,
    model: str | None = None,
    azure_endpoint: str | None = None,
    tenant_id: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    api_version: str | None = None,
    timeout: float | None = None,
    proxy_url: str | None = None,
    verify_ssl: bool | None = None,
    reasoning_effort: str | None = None,
    config_path: str | None = None,
) -> AzureJudge:
    """Create a direct Azure OpenAI judge authenticated with Azure AD.

    Pass either a fully resolved ``config`` or individual configuration values.
    """
    individual_values = (
        model,
        azure_endpoint,
        tenant_id,
        client_id,
        client_secret,
        api_version,
        timeout,
        proxy_url,
        verify_ssl,
        reasoning_effort,
        config_path,
    )
    if config is not None:
        if any(value is not None for value in individual_values):
            raise ValueError(
                "Pass either `config` or individual judge configuration "
                "arguments, not both."
            )
    else:
        config = resolve_azure_judge_config(
            model=model,
            azure_endpoint=azure_endpoint,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            api_version=api_version,
            timeout=timeout,
            proxy_url=proxy_url,
            verify_ssl=verify_ssl,
            reasoning_effort=reasoning_effort,
            config_path=config_path,
        )

    from azure.identity import (
        ClientSecretCredential,
        get_bearer_token_provider,
    )
    from openai import DefaultAsyncHttpxClient, DefaultHttpxClient
    from phoenix.evals import LLM

    credential = ClientSecretCredential(
        tenant_id=config.tenant_id,
        client_id=config.client_id,
        client_secret=config.client_secret,
    )
    token_provider = get_bearer_token_provider(credential, _AZURE_SCOPE)

    sync_client_kwargs: dict[str, Any] = {}
    async_client_kwargs: dict[str, Any] = {}
    if config.proxy_url is not None or not config.verify_ssl:
        sync_client_kwargs["http_client"] = DefaultHttpxClient(
            proxy=config.proxy_url,
            verify=config.verify_ssl,
            timeout=config.timeout,
        )
        async_client_kwargs["http_client"] = DefaultAsyncHttpxClient(
            proxy=config.proxy_url,
            verify=config.verify_ssl,
            timeout=config.timeout,
        )

    llm = LLM(
        provider="azure",
        client="openai",
        model=config.model,
        azure_endpoint=config.azure_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=config.api_version,
        timeout=config.timeout,
        sync_client_kwargs=sync_client_kwargs,
        async_client_kwargs=async_client_kwargs,
    )
    return AzureJudge(llm, credential, config.reasoning_effort)


__all__ = [
    "AzureJudge",
    "AzureJudgeConfig",
    "create_azure_judge",
    "resolve_azure_judge_config",
]
