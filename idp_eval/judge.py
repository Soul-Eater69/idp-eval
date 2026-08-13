"""Reusable judge creation for the corporate IDP LLM gateway.

Callers configure the judge once via :func:`create_judge` and never touch the
gateway HTTP client, JWT acquisition/refresh, Phoenix ``LLM`` wiring, or response
normalization. Configuration is resolved from explicit arguments, then
environment variables, then an optional YAML file (in that precedence order).

Authentication and the request/response contract mirror the existing working
gateway implementation: the IDP token endpoint, the ``chatcompletions`` gateway
path, the required headers, the fixed ``api_version``, the single 401 retry, and
the singular-``choice`` normalization are all preserved exactly.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

_logger = logging.getLogger(__name__)

# Fixed by the gateway contract; callers do not supply this.
_API_VERSION = "2024-04-01-preview"

# Scope required by the IDP token endpoint (from the working implementation).
_IDP_SCOPE = "profile openid roles permissions"

# Required judge configuration fields and their preferred environment variables.
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

# Environment variable that may point at a YAML config file.
_CONFIG_ENV = "IDP_EVAL_CONFIG"
_VERIFY_SSL_ENV = "IDP_EVAL_VERIFY_SSL"


@dataclass(frozen=True)
class JudgeConfig:
    """Resolved configuration required to build a gateway-backed judge.

    All string fields are required. ``verify_ssl`` is a safety setting, not a
    credential, and defaults to ``True``; set it to ``False`` only for local
    testing against a gateway with a self-signed certificate.

    Attributes:
        model: Gateway model name (injected into every request payload).
        base_url: Gateway base URL; ``/api/v1/...`` paths are derived from it.
        app_id: Value for the gateway ``app-id`` header.
        idp_auth_url: IDP token endpoint.
        idp_client_id: IDP client id (``ClientId`` header).
        idp_client_secret: IDP client secret (``ClientSecret`` header).
        idp_user: IDP username (request body).
        idp_password: IDP password (request body).
        verify_ssl: Whether TLS certificates are verified. Defaults to ``True``.
    """

    model: str
    base_url: str
    app_id: str
    idp_auth_url: str
    idp_client_id: str
    idp_client_secret: str
    idp_user: str
    idp_password: str
    verify_ssl: bool = True


_REQUIRED_FIELDS = tuple(_ENV_VARS)


def _load_yaml_judge_section(config_path: str | None) -> dict[str, Any]:
    """Loads the ``judge`` section from an optional YAML config file.

    Args:
        config_path: Explicit path, or ``None`` to fall back to the
            ``IDP_EVAL_CONFIG`` environment variable. When neither is set, no
            file is read.

    Returns:
        The ``judge`` mapping, or an empty dict when no config is available.
    """
    path = config_path or os.environ.get(_CONFIG_ENV)
    if not path:
        return {}

    import yaml  # Imported lazily; only needed when a config file is used.

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    section = data.get("judge", {}) or {}
    if not isinstance(section, dict):
        raise ValueError("YAML 'judge' section must be a mapping.")
    return section


def _resolve_verify_ssl(
    verify_ssl: bool | None, yaml_values: dict[str, Any]
) -> bool:
    """Resolves ``verify_ssl`` from explicit arg, env, YAML, then default True."""
    if verify_ssl is not None:
        return verify_ssl
    env_value = os.environ.get(_VERIFY_SSL_ENV)
    if env_value is not None:
        return env_value.strip().lower() not in {"false", "0", "no"}
    if "verify_ssl" in yaml_values:
        return bool(yaml_values["verify_ssl"])
    return True


def resolve_judge_config(
    *,
    config_path: str | None = None,
    verify_ssl: bool | None = None,
    **overrides: str,
) -> JudgeConfig:
    """Resolves judge configuration with explicit > env > YAML precedence.

    Args:
        config_path: Optional path to a YAML config file. Falls back to
            ``IDP_EVAL_CONFIG`` when not given.
        verify_ssl: Optional TLS verification override.
        **overrides: Explicit values for any :class:`JudgeConfig` string field.

    Returns:
        A fully populated :class:`JudgeConfig`.

    Raises:
        TypeError: If an unknown override field name is supplied.
        ValueError: If required fields remain unset after resolution. The error
            lists the missing field names only; secret values are never shown.
    """
    unknown = set(overrides) - set(_REQUIRED_FIELDS)
    if unknown:
        raise TypeError(
            f"Unknown judge configuration field(s): {sorted(unknown)}"
        )

    yaml_values = _load_yaml_judge_section(config_path)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for field_name in _REQUIRED_FIELDS:
        value = overrides.get(field_name)
        if value is None:
            value = os.environ.get(_ENV_VARS[field_name])
        if value is None:
            value = yaml_values.get(field_name)
        if value is None or value == "":
            missing.append(field_name)
        else:
            resolved[field_name] = value

    if missing:
        raise ValueError(
            "Missing required judge configuration: " + ", ".join(missing)
        )

    return JudgeConfig(
        verify_ssl=_resolve_verify_ssl(verify_ssl, yaml_values),
        **resolved,
    )


def _get_idp_token(config: JudgeConfig) -> str:
    """Obtains a JWT from the IDP token endpoint.

    Preserves the exact working request semantics: ``ClientId`` /
    ``ClientSecret`` / ``scope`` headers, a ``{username, password}`` JSON body,
    and a ``jwt_token`` field in the response.

    Args:
        config: Resolved judge configuration.

    Returns:
        The JWT string.

    Raises:
        RuntimeError: If the response does not contain ``jwt_token``. The error
            deliberately omits the response payload, which may be sensitive.
    """
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
    """Internal ``httpx.Client`` that routes OpenAI SDK calls to the gateway.

    The OpenAI SDK builds a normal chat-completions request; this client parses
    that request's JSON body, forwards only the body (never the incoming
    headers) to the corporate gateway with the minimal required headers, and
    returns an OpenAI-shaped response. Not part of the public API.
    """

    def __init__(self, config: JudgeConfig):
        """Builds the client and acquires the initial JWT.

        Args:
            config: Resolved judge configuration (read once, stored on self).
        """
        super().__init__(timeout=90.0)
        self._config = config
        self._gateway_client = httpx.Client(
            verify=config.verify_ssl, timeout=90.0
        )
        self._token = _get_idp_token(config)
        self._gateway_url = (
            config.base_url.rstrip("/") + "/api/v1/chatcompletions"
        )

    def _gateway_headers(self) -> dict[str, str]:
        """Returns only the headers the gateway requires."""
        return {
            "Authorization": f"Bearer {self._token}",
            "app-id": self._config.app_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _call_gateway(self, payload: dict) -> httpx.Response:
        """Posts a shallow copy of ``payload`` to the gateway, retrying once on 401.

        Args:
            payload: Parsed OpenAI request body. Not mutated.

        Returns:
            The raw gateway response.
        """
        gateway_payload = dict(payload)
        gateway_payload["model"] = self._config.model
        gateway_payload["api_version"] = _API_VERSION

        response = self._gateway_client.post(
            self._gateway_url,
            headers=self._gateway_headers(),
            json=gateway_payload,
        )

        if response.status_code == 401:
            # Refresh the JWT exactly once and retry.
            self._token = _get_idp_token(self._config)
            response = self._gateway_client.post(
                self._gateway_url,
                headers=self._gateway_headers(),
                json=gateway_payload,
            )

        return response

    def send(self, request: httpx.Request, *args, **kwargs) -> httpx.Response:
        """Intercepts the OpenAI SDK request and proxies it to the gateway."""
        try:
            payload = json.loads(request.content.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - re-raised with a safe message
            raise RuntimeError("Could not parse OpenAI request body") from exc

        response = self._call_gateway(payload)
        _logger.debug("Gateway responded with status %s", response.status_code)

        try:
            gateway_payload = response.json()
        except Exception:  # noqa: BLE001 - non-JSON body preserved verbatim
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

        # Normalize a singular gateway "choice" into OpenAI's "choices" list.
        if "choice" in gateway_payload and "choices" not in gateway_payload:
            gateway_payload["choices"] = [gateway_payload.pop("choice")]

        return httpx.Response(
            status_code=response.status_code,
            json=gateway_payload,
            headers={"content-type": "application/json"},
            request=request,
        )

    def close(self) -> None:
        """Closes the nested gateway client and the superclass client."""
        self._gateway_client.close()
        super().close()


def create_judge(
    *,
    config_path: str | None = None,
    verify_ssl: bool | None = None,
    **overrides: str,
):
    """Creates a Phoenix judge ``LLM`` backed by the corporate IDP gateway.

    Resolves configuration (explicit > env > YAML), builds the internal gateway
    HTTP client (which acquires the JWT), and wires it into a Phoenix ``LLM``
    using the native OpenAI client. Tracing is intentionally NOT started here;
    call :func:`idp_eval.register_tracing` separately.

    Args:
        config_path: Optional YAML config path (falls back to ``IDP_EVAL_CONFIG``).
        verify_ssl: Optional TLS verification override (defaults to ``True``).
        **overrides: Explicit values for any :class:`JudgeConfig` string field,
            e.g. ``model="gpt-5-idp-test"``.

    Returns:
        A Phoenix ``LLM`` ready to pass to evaluators.
    """
    config = resolve_judge_config(
        config_path=config_path, verify_ssl=verify_ssl, **overrides
    )

    gateway_http_client = _GatewayHTTPClient(config)

    from phoenix.evals import LLM

    return LLM(
        provider="openai",
        client="openai",
        model=config.model,
        api_key="unused",
        base_url=config.base_url.rstrip("/") + "/api/v1",
        sync_client_kwargs={"http_client": gateway_http_client},
    )


# Kept importable for tests/introspection; not part of the public surface.
__all__ = ["JudgeConfig", "create_judge", "resolve_judge_config"]
