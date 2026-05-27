"""Retry helpers for transient errors from the Prime API and httpx.

The Prime SDK collapses HTTP errors into `APIError("HTTP <code>: ...")`. We
sniff the message for 429/5xx and retry with exponential backoff. Timeouts
and transport errors retry unconditionally.
"""
from __future__ import annotations

import random
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")

# Match the "HTTP 429: ..." / "HTTP 503: ..." prefix the Prime SDK builds.
_HTTP_CODE_RE = re.compile(r"\bHTTP\s+(\d{3})\b")
_RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})


def _http_status_of(exc: BaseException) -> int | None:
    msg = str(exc)
    m = _HTTP_CODE_RE.search(msg)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def is_retryable(exc: BaseException) -> bool:
    """True if the exception looks transient enough to retry."""
    # Lazy imports — the Prime SDK and httpx are only needed at runtime; in
    # tests we want to import this module without pulling them in.
    try:
        from prime_cli.core.client import APIError, APITimeoutError
    except Exception:  # noqa: BLE001
        APIError = ()  # type: ignore[assignment]
        APITimeoutError = ()  # type: ignore[assignment]
    try:
        import httpx
    except Exception:  # noqa: BLE001
        httpx = None  # type: ignore[assignment]

    if APITimeoutError and isinstance(exc, APITimeoutError):
        return True
    if httpx is not None:
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
    if APIError and isinstance(exc, APIError):
        status = _http_status_of(exc)
        return status is not None and status in _RETRYABLE_HTTP
    return False


def with_retry(
    fn: Callable[[], T],
    *,
    retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    on_retry: Callable[[int, int, float, BaseException], None] | None = None,
) -> T:
    """Call `fn()` with exponential-backoff retry on transient errors.

    Total wait worst-case: base_delay * (1 + 2 + 4 + 8) ≈ 15s for retries=4.
    Caller-visible delay includes per-call API latency.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as e:  # noqa: BLE001 — re-raised when non-retryable
            if attempt > retries or not is_retryable(e):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            # Jitter so concurrent runs don't synchronize their retries.
            delay = delay * (0.5 + random.random() * 0.5)
            if on_retry is not None:
                try:
                    on_retry(attempt, retries, delay, e)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(delay)
