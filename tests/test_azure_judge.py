"""Offline tests for the direct Azure OpenAI judge backend."""

import asyncio
import copy

import pytest

from idp_eval.judges.azure import (
    AzureJudge,
    AzureJudgeConfig,
    create_azure_judge,
    resolve_azure_judge_config,
)


FULL_CONFIG = {
    "model": "deployment-name",
    "azure_endpoint": "https://azure.example",
    "tenant_id": "tenant-id",
    "client_id": "client-id",
    "client_secret": "secret-value",
    "api_version": "2024-12-01-preview",
}

ENV_MAP = {
    "model": "IDP_EVAL_AZURE_MODEL",
    "azure_endpoint": "IDP_EVAL_AZURE_ENDPOINT",
    "tenant_id": "IDP_EVAL_AZURE_TENANT_ID",
    "client_id": "IDP_EVAL_AZURE_CLIENT_ID",
    "client_secret": "IDP_EVAL_AZURE_CLIENT_SECRET",
    "api_version": "IDP_EVAL_AZURE_API_VERSION",
}


@pytest.fixture(autouse=True)
def clear_azure_env(monkeypatch):
    for name in [
        *ENV_MAP.values(),
        "IDP_EVAL_CONFIG",
        "IDP_EVAL_AZURE_TIMEOUT",
        "IDP_EVAL_AZURE_PROXY_URL",
        "IDP_EVAL_AZURE_VERIFY_SSL",
        "IDP_EVAL_AZURE_REASONING_EFFORT",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_explicit_azure_config_and_safe_repr():
    config = resolve_azure_judge_config(**FULL_CONFIG)
    assert config.model == "deployment-name"
    assert config.timeout == 180.0
    assert config.verify_ssl is True
    assert "secret-value" not in repr(config)


def test_azure_config_from_environment(monkeypatch):
    for field, env_name in ENV_MAP.items():
        monkeypatch.setenv(env_name, FULL_CONFIG[field])
    monkeypatch.setenv("IDP_EVAL_AZURE_TIMEOUT", "240")
    monkeypatch.setenv("IDP_EVAL_AZURE_VERIFY_SSL", "false")
    monkeypatch.setenv("IDP_EVAL_AZURE_PROXY_URL", "http://proxy.example")
    config = resolve_azure_judge_config()
    assert config.timeout == 240.0
    assert config.verify_ssl is False
    assert config.proxy_url == "http://proxy.example"


def test_azure_explicit_beats_env_and_env_beats_yaml(tmp_path, monkeypatch):
    for field, env_name in ENV_MAP.items():
        monkeypatch.setenv(env_name, FULL_CONFIG[field])
    monkeypatch.setenv("IDP_EVAL_AZURE_MODEL", "env-model")
    path = tmp_path / "config.yaml"
    path.write_text(
        "azure_judge:\n  model: yaml-model\n  timeout: 300\n",
        encoding="utf-8",
    )
    assert (
        resolve_azure_judge_config(config_path=str(path)).model == "env-model"
    )
    assert (
        resolve_azure_judge_config(
            config_path=str(path), model="explicit-model"
        ).model
        == "explicit-model"
    )


def test_missing_azure_config_is_safe():
    with pytest.raises(ValueError) as exc:
        resolve_azure_judge_config(model="only-model")
    message = str(exc.value)
    assert "azure_endpoint" in message
    assert "client_secret" in message
    assert "secret-value" not in message


class _Closer:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class _AsyncCloser:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


class _FakeCredential(_Closer):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        type(self).calls.append(kwargs)


class _FakeLLM:
    constructed = []

    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.kwargs = kwargs
        self.calls = []
        self._sync_client = _Closer()
        self._async_client = _AsyncCloser()
        type(self).constructed.append(self)

    def generate_object(self, prompt, schema, **kwargs):
        self.calls.append(("object", prompt, schema, kwargs))
        return {"items": []}

    async def async_generate_object(self, prompt, schema, **kwargs):
        self.calls.append(("async_object", prompt, schema, kwargs))
        return {"items": []}

    def generate_classification(self, prompt, labels, **kwargs):
        self.calls.append(("classification", prompt, labels, kwargs))
        return {"label": "yes"}

    async def async_generate_classification(self, prompt, labels, **kwargs):
        self.calls.append(("async_classification", prompt, labels, kwargs))
        return {"label": "yes"}


def _patch_azure_construction(monkeypatch):
    _FakeCredential.calls.clear()
    _FakeLLM.constructed.clear()
    token = {}

    def fake_token_provider(credential, scope):
        token.update(credential=credential, scope=scope)
        return lambda: "token"

    monkeypatch.setattr("azure.identity.ClientSecretCredential", _FakeCredential)
    monkeypatch.setattr(
        "azure.identity.get_bearer_token_provider", fake_token_provider
    )
    monkeypatch.setattr("phoenix.evals.LLM", _FakeLLM)
    return token


def test_create_azure_judge_uses_azure_ad_and_phoenix_native_provider(monkeypatch):
    token = _patch_azure_construction(monkeypatch)
    judge = create_azure_judge(**FULL_CONFIG)
    llm = _FakeLLM.constructed[0]

    assert _FakeCredential.calls == [
        {
            "tenant_id": "tenant-id",
            "client_id": "client-id",
            "client_secret": "secret-value",
        }
    ]
    assert token["scope"] == "https://cognitiveservices.azure.com/.default"
    assert llm.kwargs["provider"] == "azure"
    assert llm.kwargs["client"] == "openai"
    assert llm.kwargs["model"] == "deployment-name"
    assert llm.kwargs["azure_endpoint"] == "https://azure.example"
    assert llm.kwargs["api_version"] == "2024-12-01-preview"
    assert llm.kwargs["azure_ad_token_provider"]() == "token"
    assert llm.kwargs["timeout"] == 180.0
    assert "api_key" not in llm.kwargs
    assert "temperature" not in llm.kwargs
    assert judge.model == "deployment-name"


def test_proxy_ssl_and_timeout_use_official_openai_http_clients(monkeypatch):
    _patch_azure_construction(monkeypatch)
    sync_calls = []
    async_calls = []

    class SyncHTTP:
        def __init__(self, **kwargs):
            sync_calls.append(kwargs)

    class AsyncHTTP:
        def __init__(self, **kwargs):
            async_calls.append(kwargs)

    monkeypatch.setattr("openai.DefaultHttpxClient", SyncHTTP)
    monkeypatch.setattr("openai.DefaultAsyncHttpxClient", AsyncHTTP)
    create_azure_judge(
        **FULL_CONFIG,
        proxy_url="http://proxy.example",
        verify_ssl=False,
        timeout=210,
    )
    llm = _FakeLLM.constructed[0]
    expected = {
        "proxy": "http://proxy.example",
        "verify": False,
        "timeout": 210.0,
    }
    assert sync_calls == [expected]
    assert async_calls == [expected]
    assert isinstance(llm.kwargs["sync_client_kwargs"]["http_client"], SyncHTTP)
    assert isinstance(
        llm.kwargs["async_client_kwargs"]["http_client"], AsyncHTTP
    )


def test_reasoning_effort_is_request_only_and_temperature_is_never_added():
    llm = _FakeLLM(model="deployment", provider="azure")
    judge = AzureJudge(llm, _FakeCredential(), "minimal")
    schema = {"type": "object"}

    assert judge.generate_object("prompt", schema) == {"items": []}
    asyncio.run(judge.async_generate_object("prompt", schema))
    judge.generate_classification("prompt", ["yes", "no"])
    asyncio.run(judge.async_generate_classification("prompt", ["yes", "no"]))

    for call in llm.calls:
        kwargs = call[-1]
        assert kwargs["reasoning_effort"] == "minimal"
        assert "temperature" not in kwargs


def test_reasoning_effort_omitted_when_unconfigured():
    llm = _FakeLLM(model="deployment", provider="azure")
    judge = AzureJudge(llm, _FakeCredential(), None)
    judge.generate_object("prompt", {"type": "object"})
    assert llm.calls[0][-1] == {}


def test_structured_schema_is_forwarded_without_runtime_mutation():
    llm = _FakeLLM(model="deployment", provider="azure")
    judge = AzureJudge(llm, _FakeCredential(), None)
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array"}},
        "required": ["items"],
        "additionalProperties": False,
    }
    before = copy.deepcopy(schema)
    judge.generate_object("prompt", schema)
    assert llm.calls[0][2] is schema
    assert schema == before


def test_timeout_error_propagates_from_async_judge():
    class TimeoutLLM(_FakeLLM):
        async def async_generate_object(self, prompt, schema, **kwargs):
            raise TimeoutError("request timed out")

    judge = AzureJudge(
        TimeoutLLM(model="deployment", provider="azure"),
        _FakeCredential(),
        None,
    )
    with pytest.raises(TimeoutError, match="request timed out"):
        asyncio.run(judge.async_generate_object("prompt", {}))


def test_close_and_aclose_release_owned_resources():
    sync_llm = _FakeLLM(model="deployment", provider="azure")
    sync_credential = _FakeCredential()
    sync_judge = AzureJudge(sync_llm, sync_credential, None)
    sync_judge.close()
    sync_judge.close()
    assert sync_llm._sync_client.closed == 1
    assert sync_credential.closed == 1

    async_llm = _FakeLLM(model="deployment", provider="azure")
    async_credential = _FakeCredential()
    async_judge = AzureJudge(async_llm, async_credential, None)
    asyncio.run(async_judge.aclose())
    asyncio.run(async_judge.aclose())
    assert async_llm._sync_client.closed == 1
    assert async_llm._async_client.closed == 1
    assert async_credential.closed == 1


def test_azure_config_rejects_invalid_optional_values():
    with pytest.raises(ValueError, match="timeout"):
        resolve_azure_judge_config(**FULL_CONFIG, timeout=0)
    with pytest.raises(ValueError, match="verify_ssl"):
        resolve_azure_judge_config(**FULL_CONFIG, verify_ssl="maybe")


def test_config_dataclass_secret_is_not_printed():
    config = AzureJudgeConfig(**FULL_CONFIG)
    assert "secret-value" not in repr(config)
