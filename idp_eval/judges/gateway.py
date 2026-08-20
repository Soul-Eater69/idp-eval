"""Corporate IDP/Mule gateway judge backend."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field

import httpx

from idp_eval.judges._config import (
    load_yaml_section,
    resolve_bool,
    resolve_positive_float,
    resolve_required_strings,
)

_logger = logging.getLogger(__name__)

_API_VERSION = "2024-04-01-preview"
_IDP_SCOPE = "profile openid roles permissions"
_CONFIG_ENV = "IDP_EVAL_CONFIG"
_VERIFY_SSL_ENV = "IDP_EVAL_VERIFY_SSL"
_TIMEOUT_ENV = "IDP_EVAL_GATEWAY_TIMEOUT"
_ENV_VARS = {
    "model": "IDP_EVAL_MODEL",
    "base_url": "IDP_EVAL_BASE_URL",
    "app_id": "IDP_EVAL_APP_ID",
    "idp_auth_url": "IDP_EVAL_AUTH_URL",
    "idp_client_id": "IDP_EVAL_CLIENT_ID",
    "idp_client_secret": "IDP_EVAL_CLIENT_SECRET",
    "idp_user": "IDP_EVAL_USER",
    "idp_password": "IDP_EVAL_PASSWORD",
}


@dataclass(frozen=True)
class GatewayJudgeConfig:
    """Resolved corporate gateway configuration."""

    model: str
    base_url: str
    app_id: str
    idp_auth_url: str
    idp_client_id: str
    idp_client_secret: str = field(repr=False)
    idp_user: str
    idp_password: str = field(repr=False)
    verify_ssl: bool = True
    timeout: float = 90.0


def resolve_gateway_judge_config(
    *,
    model: str | None = None,
    base_url: str | None = None,
    app_id: str | None = None,
    idp_auth_url: str | None = None,
    idp_client_id: str | None = None,
    idp_client_secret: str | None = None,
    idp_user: str | None = None,
    idp_password: str | None = None,
    config_path: str | None = None,
    verify_ssl: bool | None = None,
    timeout: float | None = None,
) -> GatewayJudgeConfig:
    """Resolves gateway values using explicit > environment > YAML precedence."""
    yaml_values = load_yaml_section(
        config_path,
        config_env=_CONFIG_ENV,
        section="judge",
    )
    resolved = resolve_required_strings(
        explicit={
            "model": model,
            "base_url": base_url,
            "app_id": app_id,
            "idp_auth_url": idp_auth_url,
            "idp_client_id": idp_client_id,
            "idp_client_secret": idp_client_secret,
            "idp_user": idp_user,
            "idp_password": idp_password,
        },
        env_vars=_ENV_VARS,
        yaml_values=yaml_values,
        config_name="gateway judge",
    )
    return GatewayJudgeConfig(
        **resolved,
        verify_ssl=resolve_bool(
            verify_ssl,
            env_name=_VERIFY_SSL_ENV,
            yaml_values=yaml_values,
            field_name="verify_ssl",
            default=True,
        ),
        timeout=resolve_positive_float(
            timeout,
            env_name=_TIMEOUT_ENV,
            yaml_values=yaml_values,
            field_name="timeout",
            default=90.0,
        ),
    )


def _get_idp_token(config: GatewayJudgeConfig) -> str:
    """Obtains the gateway JWT using the established IDP contract."""
    headers = {
        "Accept": "*/*",
        "ClientId": config.idp_client_id,
        "ClientSecret": config.idp_client_secret,
        "scope": _IDP_SCOPE,
    }
    body = {"username": config.idp_user, "password": config.idp_password}

    with httpx.Client(verify=config.verify_ssl, timeout=30.0) as client:
        response = client.post(config.idp_auth_url, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()

    token = payload.get("jwt_token")
    if not token:
        raise RuntimeError(
            "IDP authentication response did not contain jwt_token."
        )
    return token


class _GatewayHTTPClient(httpx.Client):
    """Routes OpenAI SDK requests through the corporate gateway."""

    def __init__(self, config: GatewayJudgeConfig):
        super().__init__(timeout=config.timeout)
        self._config = config
        self._gateway_client = httpx.Client(
            verify=config.verify_ssl,
            timeout=config.timeout,
        )
        self._token = _get_idp_token(config)
        self._gateway_url = (
            config.base_url.rstrip("/") + "/api/v1/chatcompletions"
        )

    def _gateway_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "app-id": self._config.app_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _call_gateway(self, payload: dict) -> httpx.Response:
        gateway_payload = dict(payload)
        gateway_payload["model"] = self._config.model
        gateway_payload["api_version"] = _API_VERSION

        response = self._gateway_client.post(
            self._gateway_url,
            headers=self._gateway_headers(),
            json=gateway_payload,
        )
        if response.status_code == 401:
            self._token = _get_idp_token(self._config)
            response = self._gateway_client.post(
                self._gateway_url,
                headers=self._gateway_headers(),
                json=gateway_payload,
            )
        return response

    def send(self, request: httpx.Request, *args, **kwargs) -> httpx.Response:
        try:
            payload = json.loads(request.content.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - safe chained error
            raise RuntimeError("Could not parse OpenAI request body") from exc

        response = self._call_gateway(payload)
        _logger.debug("Gateway responded with status %s", response.status_code)

        try:
            gateway_payload = response.json()
        except Exception:  # noqa: BLE001 - preserve non-JSON gateway body
            return httpx.Response(
                status_code=response.status_code,
                content=response.content,
                headers={
                    "content-type": response.headers.get(
                        "content-type", "text/plain"
                    )
                },
                request=request,
            )

        if "choice" in gateway_payload and "choices" not in gateway_payload:
            gateway_payload["choices"] = [gateway_payload.pop("choice")]

        return httpx.Response(
            status_code=response.status_code,
            json=gateway_payload,
            headers={"content-type": "application/json"},
            request=request,
        )

    def close(self) -> None:
        if not self._gateway_client.is_closed:
            self._gateway_client.close()
        if not self.is_closed:
            super().close()


class GatewayJudge:
    """Phoenix-compatible judge that keeps all calls on the gateway transport.

    The established gateway interception client is synchronous. Phoenix also
    constructs an asynchronous OpenAI client, but that client does not use the
    custom gateway transport. These async methods therefore bridge to the same
    synchronous Phoenix LLM in a worker thread instead of bypassing the gateway.
    """

    def __init__(self, llm, http_client: _GatewayHTTPClient):
        self._llm = llm
        self._http_client = http_client
        self._closed = False

    @property
    def model(self):
        return self._llm.model

    def generate_object(self, prompt, schema, **kwargs):
        return self._llm.generate_object(prompt, schema, **kwargs)

    async def async_generate_object(self, prompt, schema, **kwargs):
        return await asyncio.to_thread(
            self._llm.generate_object,
            prompt,
            schema,
            **kwargs,
        )

    def generate_classification(self, prompt, labels, **kwargs):
        return self._llm.generate_classification(prompt, labels, **kwargs)

    async def async_generate_classification(self, prompt, labels, **kwargs):
        return await asyncio.to_thread(
            self._llm.generate_classification,
            prompt,
            labels,
            **kwargs,
        )

    def _close_sync_resources(self) -> None:
        sync_client = getattr(self._llm, "_sync_client", None)
        close = getattr(sync_client, "close", None)
        if callable(close):
            close()
        if not self._http_client.is_closed:
            self._http_client.close()

    def close(self) -> None:
        """Close synchronous Phoenix and gateway resources."""
        if self._closed:
            return
        self._close_sync_resources()
        self._closed = True

    async def aclose(self) -> None:
        """Close asynchronous and synchronous resources."""
        if self._closed:
            return
        async_client = getattr(self._llm, "_async_client", None)
        close = getattr(async_client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        self._close_sync_resources()
        self._closed = True

    def __getattr__(self, name: str):
        return getattr(self._llm, name)


def create_gateway_judge(
    *,
    config: GatewayJudgeConfig | None = None,
    model: str | None = None,
    base_url: str | None = None,
    app_id: str | None = None,
    idp_auth_url: str | None = None,
    idp_client_id: str | None = None,
    idp_client_secret: str | None = None,
    idp_user: str | None = None,
    idp_password: str | None = None,
    config_path: str | None = None,
    verify_ssl: bool | None = None,
    timeout: float | None = None,
) -> GatewayJudge:
    """Create a Phoenix judge backed by the corporate IDP/Mule gateway.

    Pass either a fully resolved ``config`` or individual configuration values.
    ``timeout`` is the client-side timeout. It cannot override a shorter timeout
    enforced by an upstream gateway.
    """
    individual_values = (
        model,
        base_url,
        app_id,
        idp_auth_url,
        idp_client_id,
        idp_client_secret,
        idp_user,
        idp_password,
        config_path,
        verify_ssl,
        timeout,
    )
    if config is not None:
        if any(value is not None for value in individual_values):
            raise ValueError(
                "Pass either `config` or individual judge configuration "
                "arguments, not both."
            )
    else:
        config = resolve_gateway_judge_config(
            model=model,
            base_url=base_url,
            app_id=app_id,
            idp_auth_url=idp_auth_url,
            idp_client_id=idp_client_id,
            idp_client_secret=idp_client_secret,
            idp_user=idp_user,
            idp_password=idp_password,
            config_path=config_path,
            verify_ssl=verify_ssl,
            timeout=timeout,
        )
    gateway_http_client = _GatewayHTTPClient(config)

    from phoenix.evals import LLM

    try:
        llm = LLM(
            provider="openai",
            client="openai",
            model=config.model,
            api_key="unused",
            base_url=config.base_url.rstrip("/") + "/api/v1",
            sync_client_kwargs={"http_client": gateway_http_client},
        )
    except Exception:
        gateway_http_client.close()
        raise
    return GatewayJudge(llm, gateway_http_client)


__all__ = [
    "GatewayJudge",
    "GatewayJudgeConfig",
    "create_gateway_judge",
    "resolve_gateway_judge_config",
]
