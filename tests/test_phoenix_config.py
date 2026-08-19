"""Offline tests for Phoenix endpoint/API-key configuration.

No Phoenix server and no network: the Phoenix SDK entry points are monkeypatched
with recorders. Verifies explicit config is forwarded, env resolution is left to
the SDK (no env copying), injected clients win, and secrets never leak.
"""

import pytest

pytest.importorskip("phoenix.otel")
pytest.importorskip("phoenix.client")

from idp_eval.output import (  # noqa: E402
    EvaluationRecord,
    PhoenixEvaluationWriter,
)
from idp_eval.models import EvaluationResult  # noqa: E402
from idp_eval.phoenix_client import register_tracing  # noqa: E402


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return object()  # stand-in tracer provider / client


# --- register_tracing --------------------------------------------------------


def test_register_tracing_default_forwards(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("phoenix.otel.register", rec)
    register_tracing(project_name="abc")
    kw = rec.calls[0]["kwargs"]
    assert kw["project_name"] == "abc"
    assert kw["batch"] is False
    # endpoint/api_key forwarded as None -> Phoenix resolves env itself.
    assert kw["endpoint"] is None
    assert kw["api_key"] is None


def test_register_tracing_explicit_endpoint(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("phoenix.otel.register", rec)
    register_tracing(project_name="abc", endpoint="https://phoenix.example.com")
    assert rec.calls[0]["kwargs"]["endpoint"] == "https://phoenix.example.com"


def test_register_tracing_explicit_api_key_forwarded(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("phoenix.otel.register", rec)
    register_tracing(project_name="abc", api_key="secret-token")
    assert rec.calls[0]["kwargs"]["api_key"] == "secret-token"


def test_register_tracing_does_not_copy_env(monkeypatch):
    # Even with env vars set, we forward None (SDK does the resolution, not us).
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://env-phoenix")
    monkeypatch.setenv("PHOENIX_API_KEY", "env-key")
    rec = _Recorder()
    monkeypatch.setattr("phoenix.otel.register", rec)
    register_tracing(project_name="abc")
    kw = rec.calls[0]["kwargs"]
    assert kw["endpoint"] is None and kw["api_key"] is None


# --- PhoenixEvaluationWriter client construction -----------------------------


def test_writer_default_client_uses_env(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("phoenix.client.Client", rec)
    PhoenixEvaluationWriter()._get_client()
    # No explicit config -> Client() called with no kwargs (env-driven).
    assert rec.calls[0]["kwargs"] == {}


def test_writer_explicit_config(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("phoenix.client.Client", rec)
    PhoenixEvaluationWriter(
        base_url="https://phoenix.example.com", api_key="secret"
    )._get_client()
    kw = rec.calls[0]["kwargs"]
    assert kw == {"base_url": "https://phoenix.example.com", "api_key": "secret"}


def test_writer_partial_config_only_forwards_set_values(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("phoenix.client.Client", rec)
    PhoenixEvaluationWriter(base_url="https://phoenix.example.com")._get_client()
    assert rec.calls[0]["kwargs"] == {"base_url": "https://phoenix.example.com"}


def test_writer_injected_client_wins(monkeypatch):
    rec = _Recorder()  # would record if a Client were ever built
    monkeypatch.setattr("phoenix.client.Client", rec)
    fake = object()
    writer = PhoenixEvaluationWriter(
        client=fake, base_url="https://x", api_key="secret"
    )
    assert writer._get_client() is fake
    assert rec.calls == []  # no Client constructed


def test_api_key_never_in_payload():
    record = EvaluationRecord.from_result(
        EvaluationResult("coverage", 0.5, "incomplete", "why", {"k": "v"}),
        annotator_kind="LLM",
        span_id="abc",
    )
    payload = PhoenixEvaluationWriter._payload(record)
    flat = repr(payload)
    assert "api_key" not in flat and "Authorization" not in flat
    assert "secret" not in flat
