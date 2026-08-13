"""Phoenix tracing registration.

Judge construction lives in :mod:`idp_eval.judge` (see ``create_judge``). This
module only owns Phoenix OpenTelemetry tracing, which is a separate concern from
building the judge.
"""

from __future__ import annotations


def register_tracing(
    project_name: str = "idp-eval",
    batch: bool = False,
) -> None:
    """Registers Phoenix OpenTelemetry tracing at application startup.

    Call this once, early, and separately from ``create_judge``. No tracing code
    belongs in metric scoring logic.

    Args:
        project_name: Phoenix project to record spans under.
        batch: Whether to batch span export.
    """
    from phoenix.otel import register

    register(project_name=project_name, batch=batch)
