"""Single place that constructs the Phoenix judge LLM.

Gateway wiring lives here and nowhere else. Evaluators receive the judge object
via dependency injection; they never build it themselves.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Default judge model. Override via ``get_judge_llm(model=...)``.
DEFAULT_JUDGE_MODEL = "gpt-4o"


@lru_cache(maxsize=None)
def get_judge_llm(
    model: str = DEFAULT_JUDGE_MODEL,
    gateway_http_client: Any | None = None,
):
    """Builds (and caches) the Phoenix judge LLM.

    The corporate gateway is injected through ``sync_client_kwargs`` so all
    judge traffic flows through it. The result is cached so the whole process
    shares one judge object.

    Args:
        model: Judge model name understood by the gateway.
        gateway_http_client: Pre-configured HTTP client pointing at the
            corporate LLM gateway. When ``None``, the provider's default client
            is used (useful for local development).

    Returns:
        A Phoenix ``LLM`` instance.
    """
    from phoenix.evals import LLM

    sync_client_kwargs = {}
    if gateway_http_client is not None:
        sync_client_kwargs["http_client"] = gateway_http_client

    return LLM(
        provider="openai",
        client="openai",
        model=model,
        sync_client_kwargs=sync_client_kwargs,
    )


def register_tracing(
    project_name: str = "idp-eval",
    batch: bool = False,
) -> None:
    """Registers Phoenix OpenTelemetry tracing at application startup.

    Call this once, early. No tracing code belongs in metric scoring logic.

    Args:
        project_name: Phoenix project to record spans under.
        batch: Whether to batch span export.
    """
    from phoenix.otel import register

    register(project_name=project_name, batch=batch)
