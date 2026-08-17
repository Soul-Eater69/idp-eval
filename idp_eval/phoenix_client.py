"""Phoenix tracing registration.

Judge construction lives in :mod:`idp_eval.judge` (see ``create_judge``). This
module only owns Phoenix OpenTelemetry tracing, which is a separate concern from
building the judge.
"""

from __future__ import annotations


def register_tracing(
    project_name: str = "idp-eval",
    endpoint: str | None = None,
    api_key: str | None = None,
    batch: bool = False,
) -> None:
    """Registers Phoenix OpenTelemetry tracing at application startup.

    Call this once, early, and separately from ``create_judge``. No tracing code
    belongs in metric scoring logic.

    Configuration resolution is delegated to Phoenix's ``register`` (verified
    against ``arize-phoenix-otel`` 0.17.1), so precedence is: explicit argument →
    Phoenix environment variable → Phoenix SDK default (local). When ``endpoint``
    or ``api_key`` are ``None``, Phoenix reads them from
    ``PHOENIX_COLLECTOR_ENDPOINT`` and ``PHOENIX_API_KEY`` respectively; local
    unauthenticated Phoenix needs no API key. Environment variables are never
    copied into arguments here.

    Args:
        project_name: Phoenix project to record spans under (Phoenix also honors
            ``PHOENIX_PROJECT_NAME``).
        endpoint: OTLP collector endpoint for trace export. ``None`` →
            ``PHOENIX_COLLECTOR_ENDPOINT`` / local default.
        api_key: API key for authenticated Phoenix. ``None`` → ``PHOENIX_API_KEY``
            / none for local. Never logged.
        batch: Whether to batch span export.
    """
    from phoenix.otel import register

    register(
        project_name=project_name,
        endpoint=endpoint,
        api_key=api_key,
        batch=batch,
    )
