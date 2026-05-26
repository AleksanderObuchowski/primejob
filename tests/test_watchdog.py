"""Tests for lease heartbeat and watchdog stale detection."""
from __future__ import annotations

import importlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from primejob.watchdog import (
    DEFAULT_STALE_GRACE_S,
    create_lease,
    heartbeat,
    lease_is_stale,
    read_lease,
    release_lease,
)


def test_lease_heartbeat_and_release(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    import primejob.state as state_mod
    import primejob.watchdog as watchdog_mod

    importlib.reload(state_mod)
    importlib.reload(watchdog_mod)

    lease = watchdog_mod.create_lease("run-a", "pod-1", parent_pid=12345)
    assert lease.path.exists()

    time.sleep(0.05)
    watchdog_mod.heartbeat("run-a")
    updated = watchdog_mod.read_lease("run-a")
    assert updated is not None
    assert updated.heartbeat_at >= lease.heartbeat_at

    watchdog_mod.release_lease("run-a")
    released = watchdog_mod.read_lease("run-a")
    assert released is not None
    assert released.released is True
    assert not watchdog_mod.lease_is_stale(released)


def test_lease_is_stale_when_heartbeat_old(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    import primejob.state as state_mod
    import primejob.watchdog as watchdog_mod

    importlib.reload(state_mod)
    importlib.reload(watchdog_mod)

    lease = watchdog_mod.create_lease("run-b", "pod-2")
    lease.heartbeat_at = time.time() - DEFAULT_STALE_GRACE_S - 10
    lease.path.write_text(json.dumps(asdict(lease), indent=2))
    stale = watchdog_mod.read_lease("run-b")
    assert stale is not None
    assert watchdog_mod.lease_is_stale(stale)
