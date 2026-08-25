"""Central classification and safe reporting of transient operational errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import httpx


_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_SENSITIVE_MARKERS = (
    "authorization",
    "bearer ",
    "api key",
    "api_key",
    "client_secret",
    "password",
    "jwt",
    "token",
)


@dataclass(frozen=True)
class OperationalErrorInfo:
    """Compact, persistence-safe facts about one operational failure."""

    error_type: str
    explanation: str
    retryable: bool
    provider: str | None = None
    status_code: int | None = None
    request_id: str | None = None
    retry_after_seconds: float | None = None
    current_rate_tokens_per_sec: float | None = None
    initial_rate_tokens_per_sec: float | None = None
    enforcement_window_seconds: float | None = None

    def details(self) -> dict[str, Any]:
        """Returns compact JSON-safe error metadata, omitting unknown values."""
        values = {
            "status": "error",
            "error_type": self.error_type,
            "retryable": self.retryable,
            "provider": self.provider,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "retry_after_seconds": self.retry_after_seconds,
            "current_rate_tokens_per_sec": self.current_rate_tokens_per_sec,
            "initial_rate_tokens_per_sec": self.initial_rate_tokens_per_sec,
            "enforcement_window_seconds": self.enforcement_window_seconds,
        }
        return {key: value for key, value in values.items() if value is not None}


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yields an exception and its explicit/implicit causes without looping."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _headers(exc: BaseException):
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None)


def _request_id(exc: BaseException) -> str | None:
    value = getattr(exc, "request_id", None)
    if isinstance(value, str) and value:
        return value
    headers = _headers(exc)
    if headers is not None:
        value = headers.get("x-request-id") or headers.get("request-id")
        if isinstance(value, str) and value:
            return value
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = _headers(exc)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _safe_message(exc: BaseException) -> str:
    """Returns one bounded line, suppressing messages likely to contain secrets."""
    error_type = type(exc).__name__
    message = " ".join(str(exc).split())
    if not message or any(marker in message.lower() for marker in _SENSITIVE_MARKERS):
        return f"{error_type}: operational provider failure"
    if len(message) > 240:
        message = f"{message[:237]}..."
    return f"{error_type}: {message}"


def _provider(exc: BaseException) -> str | None:
    module = type(exc).__module__
    if module.startswith("openai"):
        return "openai"
    if module.startswith("phoenix"):
        return "phoenix"
    if module.startswith("httpx") or module.startswith("httpcore"):
        return "http"
    return None


def _is_openai_operational(exc: BaseException) -> bool:
    try:
        import openai

        classes = tuple(
            cls
            for cls in (
                getattr(openai, "RateLimitError", None),
                getattr(openai, "APIConnectionError", None),
                getattr(openai, "APITimeoutError", None),
                getattr(openai, "InternalServerError", None),
            )
            if isinstance(cls, type)
        )
        return isinstance(exc, classes)
    except ImportError:
        return False


def _is_phoenix_rate_limit(exc: BaseException) -> bool:
    try:
        from phoenix.evals.rate_limiters import RateLimitError

        return isinstance(exc, RateLimitError)
    except ImportError:
        return (
            type(exc).__name__ == "RateLimitError"
            and type(exc).__module__.startswith("phoenix")
        )


def _is_operational(exc: BaseException) -> bool:
    if _is_phoenix_rate_limit(exc) or _is_openai_operational(exc):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if _status_code(exc) not in _RETRYABLE_STATUS_CODES:
        return False
    # A numeric ``status_code`` alone is not enough: arbitrary evaluator bugs
    # may expose similarly named fields. Require a known provider/HTTP module or
    # an actual response object before treating the status as operational.
    return (
        _provider(exc) is not None
        or getattr(exc, "response", None) is not None
    )


def classify_operational_error(exc: BaseException) -> OperationalErrorInfo | None:
    """Classifies known provider/transport failures across an exception chain.

    Validation errors, type errors, arbitrary runtime errors, schema failures,
    and persistence failures intentionally return ``None`` and continue to raise.
    No retry or delay is performed here.
    """
    for candidate in _exception_chain(exc):
        if not _is_operational(candidate):
            continue
        return OperationalErrorInfo(
            error_type=type(candidate).__name__,
            explanation=_safe_message(candidate),
            retryable=True,
            provider=_provider(candidate),
            status_code=_status_code(candidate),
            request_id=_request_id(candidate),
            retry_after_seconds=_retry_after_seconds(candidate),
            current_rate_tokens_per_sec=getattr(
                candidate, "current_rate_tokens_per_sec", None
            ),
            initial_rate_tokens_per_sec=getattr(
                candidate, "initial_rate_tokens_per_sec", None
            ),
            enforcement_window_seconds=getattr(
                candidate, "enforcement_window_seconds", None
            ),
        )
    return None
