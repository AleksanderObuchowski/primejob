"""Tests for CleanupGuard and main-thread signal registration."""
from __future__ import annotations

from primejob.runtime import (
    CleanupGuard,
    _dispatch_active_cleanup,
    register_active_cleanup,
)


def test_cleanup_guard_fires_once() -> None:
    calls: list[str] = []

    def cleanup() -> None:
        calls.append("ok")

    with CleanupGuard(cleanup):
        pass
    assert calls == ["ok"]

    guard = CleanupGuard(cleanup)
    guard.fire()
    guard.fire()
    assert calls.count("ok") == 2


def test_register_active_cleanup_dispatched() -> None:
    calls: list[str] = []

    def cleanup() -> None:
        calls.append("active")

    register_active_cleanup(cleanup)
    try:
        _dispatch_active_cleanup()
        assert calls == ["active"]
    finally:
        register_active_cleanup(None)
