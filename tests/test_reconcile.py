"""Tests for stale run detection."""
from __future__ import annotations

import importlib
import time
from pathlib import Path

from primejob.reconcile import assess_run
from primejob.state import RunRecord


def test_assess_run_orphan_marker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    import primejob.state as state_mod
    import primejob.reconcile as reconcile_mod

    importlib.reload(state_mod)
    importlib.reload(reconcile_mod)

    record = RunRecord(
        run_id="run-orphan",
        pod_id="pod-x",
        gpu_type="L40S",
        gpu_count=1,
        country="US",
        provider="massedcompute",
        rate_per_hr=0.82,
        script="train.py",
        status="running",
    )
    record.save()
    (record.dir / "orphaned.txt").write_text("pod-x\n")

    health = reconcile_mod.assess_run(record)
    assert health.stale is True
    assert health.has_orphan_marker is True


def test_assess_run_finished_not_stale(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    import primejob.state as state_mod

    importlib.reload(state_mod)

    record = RunRecord(
        run_id="run-done",
        pod_id="pod-y",
        gpu_type="L40S",
        gpu_count=1,
        country="US",
        provider="massedcompute",
        rate_per_hr=0.82,
        script="train.py",
        status="finished",
        ended_at=time.time(),
    )
    record.save()

    health = assess_run(record)
    assert health.stale is False
