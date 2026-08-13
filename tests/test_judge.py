"""Tests for judge configuration, the IDP gateway client, and create_judge.

No real Phoenix or IDP systems are contacted; the token helper and gateway HTTP
transport are mocked.
"""

import json
import sys
import types

import httpx
import pytest

from idp_eval import judge as judge_mod
from idp_eval.judge import (
    JudgeConfig,
    _GatewayHTTPClient,
    _get_idp_token,
    create_judge,
    resolve_judge_config,
)

FULL_CONFIG = {
    "model": "gpt-5-idp",
    "base_url": "https://gateway.example",
    "app_id": "app-123",
    "idp_auth_url": "https://auth.example/token",
    "idp_client_id": "client-abc",
    "idp_client_secret": "secret-xyz",
    "idp_user": "svc-user",
    "idp_password": "svc-pass",
}

ENV_MAP = {
    "model": "IDP_EVAL_MODEL",
    "base_url": "IDP_EVAL_BASE_URL",
    "app_id": "IDP_EVAL_APP_ID",
    "idp_auth_url": "IDP_EVAL_AUTH_URL",
    "idp_client_id": "IDP_EVAL_CLIENT_ID",
    "idp_client_secret": "IDP_EVAL_CLIENT_SECRET",
    "idp_user": "IDP_EVAL_USER",
    "idp_password": "IDP_EVAL_PASSWORD",
}


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensures a clean environment for every test."""
    for env_name in list(ENV_MAP.values()) + ["IDP_EVAL_CONFIG", "IDP_EVAL_VERIFY_SSL"]:
        monkeypatch.delenv(env_name, raising=False)


def _config(**overrides) -> JudgeConfig:
    values = {**FULL_CONFIG, **overrides}
    return JudgeConfig(**values)


# --- configuration resolution -----------------------------------------------


def test_resolve_from_explicit_args():
    config = resolve_judge_config(**FULL_CONFIG)
    assert config.model == "gpt-5-idp"
    assert config.verify_ssl is True


def test_resolve_from_environment(monkeypatch):
    for field, env_name in ENV_MAP.items():
        monkeypatch.setenv(env_name, FULL_CONFIG[field])
    config = resolve_judge_config()
    assert config.app_id == "app-123"


def test_resolve_from_yaml(tmp_path, monkeypatch):
    for field, env_name in ENV_MAP.items():
        monkeypatch.setenv(env_name, FULL_CONFIG[field])
    # Secrets come from env; non-secret fields from YAML.
    for secret in ("idp_client_secret", "idp_password"):
        monkeypatch.delenv(ENV_MAP[secret])
        monkeypatch.setenv(ENV_MAP[secret], "from-env")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "judge:\n  model: yaml-model\n  base_url: https://yaml\n"
        "  app_id: yaml-app\n  idp_auth_url: https://yaml-auth\n",
        encoding="utf-8",
    )
    # Clear the env for the YAML-provided fields so YAML supplies them.
    for field in ("model", "base_url", "app_id", "idp_auth_url"):
        monkeypatch.delenv(ENV_MAP[field])
    config = resolve_judge_config(config_path=str(yaml_path))
    assert config.model == "yaml-model"
    assert config.base_url == "https://yaml"


def test_explicit_beats_env(monkeypatch):
    for field, env_name in ENV_MAP.items():
        monkeypatch.setenv(env_name, FULL_CONFIG[field])
    monkeypatch.setenv("IDP_EVAL_MODEL", "env-model")
    config = resolve_judge_config(model="explicit-model")
    assert config.model == "explicit-model"


def test_env_beats_yaml(tmp_path, monkeypatch):
    for field, env_name in ENV_MAP.items():
        monkeypatch.setenv(env_name, FULL_CONFIG[field])
    monkeypatch.setenv("IDP_EVAL_MODEL", "env-model")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("judge:\n  model: yaml-model\n", encoding="utf-8")
    config = resolve_judge_config(config_path=str(yaml_path))
    assert config.model == "env-model"


def test_missing_configuration_lists_fields():
    with pytest.raises(ValueError) as exc:
        resolve_judge_config(model="only-model")
    message = str(exc.value)
    assert "base_url" in message and "idp_password" in message


def test_missing_error_does_not_leak_secret_values(monkeypatch):
    # Provide everything except password; supply a real-looking secret value.
    for field, env_name in ENV_MAP.items():
        if field != "idp_password":
            monkeypatch.setenv(env_name, FULL_CONFIG[field])
    with pytest.raises(ValueError) as exc:
        resolve_judge_config()
    message = str(exc.value)
    # The secret *value* must never appear; only the field name may.
    assert "secret-xyz" not in message
    assert "idp_password" in message


def test_unknown_override_field_raises():
    with pytest.raises(TypeError):
        resolve_judge_config(nonsense="x", **FULL_CONFIG)


def test_verify_ssl_from_env(monkeypatch):
    for field, env_name in ENV_MAP.items():
        monkeypatch.setenv(env_name, FULL_CONFIG[field])
    monkeypatch.setenv("IDP_EVAL_VERIFY_SSL", "false")
    assert resolve_judge_config().verify_ssl is False


# --- token helper (auth contract) -------------------------------------------


def _auth_transport(capture: dict, payload: dict, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        capture["url"] = str(request.url)
        capture["headers"] = request.headers
        capture["body"] = json.loads(request.content)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def test_get_idp_token_contract(monkeypatch):
    capture: dict = {}
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        return real_client(transport=_auth_transport(capture, {"jwt_token": "JWT123"}))

    monkeypatch.setattr(httpx, "Client", fake_client)
    token = _get_idp_token(_config())

    assert token == "JWT123"
    assert capture["url"] == "https://auth.example/token"
    assert capture["headers"]["ClientId"] == "client-abc"
    assert capture["headers"]["ClientSecret"] == "secret-xyz"
    assert capture["headers"]["scope"] == "profile openid roles permissions"
    assert capture["body"] == {"username": "svc-user", "password": "svc-pass"}


def test_get_idp_token_missing_jwt_is_safe(monkeypatch):
    capture: dict = {}
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        return real_client(
            transport=_auth_transport(capture, {"error": "nope", "secret": "s"})
        )

    monkeypatch.setattr(httpx, "Client", fake_client)
    with pytest.raises(RuntimeError) as exc:
        _get_idp_token(_config())
    message = str(exc.value)
    assert message == "IDP authentication response did not contain jwt_token."
    assert "secret" not in message  # full payload not leaked


# --- gateway client ---------------------------------------------------------


def _build_client(monkeypatch, handler, config=None, token="TESTTOKEN"):
    tokens = iter([token, token + "-refreshed", token + "-3"])
    monkeypatch.setattr(judge_mod, "_get_idp_token", lambda cfg: next(tokens))
    client = _GatewayHTTPClient(config or _config())
    client._gateway_client.close()
    client._gateway_client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def _openai_request(body: dict, headers: dict | None = None) -> httpx.Request:
    return httpx.Request(
        "POST",
        "https://api.openai.local/v1/chat/completions",
        json=body,
        headers=headers or {},
    )


def test_gateway_headers_and_url(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = _build_client(monkeypatch, handler)
    try:
        client.send(_openai_request({"messages": [{"role": "user", "content": "hi"}]}))
    finally:
        client.close()

    assert seen["url"] == "https://gateway.example/api/v1/chatcompletions"
    assert seen["headers"]["Authorization"] == "Bearer TESTTOKEN"
    assert seen["headers"]["app-id"] == "app-123"
    assert seen["headers"]["Content-Type"] == "application/json"
    assert seen["headers"]["Accept"] == "application/json"
    # Gateway payload carries the injected model + api_version.
    assert seen["body"]["model"] == "gpt-5-idp"
    assert seen["body"]["api_version"] == "2024-04-01-preview"
    # Original OpenAI fields preserved.
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_incoming_headers_not_forwarded(monkeypatch):
    """Header-size regression: large/arbitrary OpenAI headers must not be sent."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        return httpx.Response(200, json={"choices": []})

    client = _build_client(monkeypatch, handler)
    huge_headers = {
        "authorization": "Bearer unused",
        "x-openai-huge": "A" * 20000,
        "x-stainless-arch": "arm64",
        "cookie": "session=" + "b" * 8000,
    }
    try:
        client.send(_openai_request({"messages": []}, headers=huge_headers))
    finally:
        client.close()

    forwarded = seen["headers"]
    # None of the oversized/arbitrary incoming headers reach the gateway.
    assert "x-openai-huge" not in forwarded
    assert "x-stainless-arch" not in forwarded
    assert "cookie" not in forwarded
    # Authorization is our JWT, not the incoming "Bearer unused".
    assert forwarded["Authorization"] == "Bearer TESTTOKEN"


def test_payload_not_mutated(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _build_client(monkeypatch, handler)
    original = {"messages": [{"role": "user", "content": "hi"}]}
    try:
        client._call_gateway(original)
    finally:
        client.close()
    # The gateway-specific keys are added to a copy, not the original dict.
    assert "model" not in original
    assert "api_version" not in original


def test_temperature_not_injected(monkeypatch):
    """Regression: the outgoing gateway payload must not contain temperature."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": []})

    client = _build_client(monkeypatch, handler)
    try:
        client.send(_openai_request({"messages": [{"role": "user", "content": "hi"}]}))
    finally:
        client.close()
    assert "temperature" not in seen["body"]


def test_single_retry_after_401(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = _build_client(monkeypatch, handler)
    try:
        response = client.send(_openai_request({"messages": []}))
    finally:
        client.close()
    assert calls["n"] == 2
    assert response.status_code == 200


def test_no_second_retry_after_repeated_401(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "expired"})

    client = _build_client(monkeypatch, handler)
    try:
        response = client.send(_openai_request({"messages": []}))
    finally:
        client.close()
    assert calls["n"] == 2  # initial + exactly one retry
    assert response.status_code == 401


def test_choice_normalized_to_choices(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choice": {"message": {"content": "ok"}}})

    client = _build_client(monkeypatch, handler)
    try:
        response = client.send(_openai_request({"messages": []}))
    finally:
        client.close()
    body = response.json()
    assert body["choices"] == [{"message": {"content": "ok"}}]
    assert "choice" not in body


def test_non_json_response_preserved(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, content=b"gateway down", headers={"content-type": "text/plain"}
        )

    client = _build_client(monkeypatch, handler)
    try:
        response = client.send(_openai_request({"messages": []}))
    finally:
        client.close()
    assert response.status_code == 503
    assert response.content == b"gateway down"
    assert response.headers["content-type"] == "text/plain"


def test_synthetic_response_has_minimal_headers(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": []},
            headers={"content-type": "application/json", "x-gw-secret": "leak"},
        )

    client = _build_client(monkeypatch, handler)
    try:
        response = client.send(_openai_request({"messages": []}))
    finally:
        client.close()
    # Gateway-specific headers are not copied into the OpenAI-facing response.
    assert "x-gw-secret" not in response.headers
    assert response.headers["content-type"] == "application/json"


def test_bad_request_body_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _build_client(monkeypatch, handler)
    bad = httpx.Request(
        "POST", "https://api.openai.local/v1/chat/completions", content=b"not json"
    )
    try:
        with pytest.raises(RuntimeError, match="Could not parse OpenAI request body"):
            client.send(bad)
    finally:
        client.close()


# --- create_judge -----------------------------------------------------------


def test_create_judge_builds_phoenix_llm(monkeypatch):
    captured: dict = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_evals = types.ModuleType("phoenix.evals")
    fake_evals.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "phoenix", types.ModuleType("phoenix"))
    monkeypatch.setitem(sys.modules, "phoenix.evals", fake_evals)
    monkeypatch.setattr(judge_mod, "_get_idp_token", lambda cfg: "JWT")

    create_judge(**FULL_CONFIG)

    assert captured["provider"] == "openai"
    assert captured["client"] == "openai"
    assert captured["model"] == "gpt-5-idp"
    assert captured["api_key"] == "unused"
    assert captured["base_url"] == "https://gateway.example/api/v1"
    assert "temperature" not in captured
    http_client = captured["sync_client_kwargs"]["http_client"]
    assert isinstance(http_client, _GatewayHTTPClient)
    http_client.close()
