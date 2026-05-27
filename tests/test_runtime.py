"""Tests for CleanupGuard, cost tracker, and main-thread signal registration."""
from __future__ import annotations

import time

from primejob.runtime import (
    CleanupGuard,
    CostTracker,
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


def test_cost_tracker_segments_across_rate_changes(monkeypatch) -> None:
    """Provider fallback mid-run must keep prior seconds priced at the old rate."""
    clock = {"now": 1000.0}
    monkeypatch.setattr("primejob.runtime.time.monotonic", lambda: clock["now"])

    tracker = CostTracker(rate_per_hr=2.0)
    # 100 seconds at $2/h
    clock["now"] += 100.0
    assert tracker.spent() == (2.0 / 3600.0) * 100.0

    tracker.update_rate(4.0)
    # 50 more seconds at $4/h
    clock["now"] += 50.0
    expected = (2.0 / 3600.0) * 100.0 + (4.0 / 3600.0) * 50.0
    assert abs(tracker.spent() - expected) < 1e-9


def test_cost_tracker_ignores_noop_rate_update(monkeypatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("primejob.runtime.time.monotonic", lambda: clock["now"])
    tracker = CostTracker(rate_per_hr=1.5)
    tracker.update_rate(1.5)  # no-op
    tracker.update_rate(1.5)  # no-op
    assert len(tracker._segments) == 1
