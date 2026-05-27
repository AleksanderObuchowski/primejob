"""Tests for _retry.with_retry — Prime API transient error handling."""
from __future__ import annotations

import pytest

from primejob._retry import is_retryable, with_retry


def test_with_retry_returns_value_on_first_success() -> None:
    calls = {"n": 0}

    def fn() -> int:
        calls["n"] += 1
        return 42

    assert with_retry(fn) == 42
    assert calls["n"] == 1


def test_with_retry_retries_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("primejob._retry.time.sleep", lambda _s: None)
    from prime_cli.core.client import APIError

    seq = iter([APIError("HTTP 503: backend down"), 7])

    def fn() -> int:
        v = next(seq)
        if isinstance(v, BaseException):
            raise v
        return v

    assert with_retry(fn, retries=3, base_delay=0.01) == 7


def test_with_retry_gives_up_after_exhaustion(monkeypatch) -> None:
    monkeypatch.setattr("primejob._retry.time.sleep", lambda _s: None)
    from prime_cli.core.client import APIError

    def fn() -> int:
        raise APIError("HTTP 429: too many")

    with pytest.raises(APIError):
        with_retry(fn, retries=2, base_delay=0.01)


def test_with_retry_does_not_retry_non_retryable() -> None:
    from prime_cli.core.client import APIError

    calls = {"n": 0}

    def fn() -> int:
        calls["n"] += 1
        raise APIError("HTTP 422: bad payload")

    with pytest.raises(APIError):
        with_retry(fn, retries=5, base_delay=0.01)
    assert calls["n"] == 1


def test_is_retryable_classification() -> None:
    from prime_cli.core.client import APIError, APITimeoutError

    assert is_retryable(APIError("HTTP 503: ..."))
    assert is_retryable(APIError("HTTP 429: ..."))
    assert is_retryable(APITimeoutError("slow"))
    assert not is_retryable(APIError("HTTP 401: nope"))
    assert not is_retryable(APIError("HTTP 422: bad"))
    assert not is_retryable(ValueError("unrelated"))
