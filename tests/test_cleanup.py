"""Tests for shared terminate_run_pod."""
from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

from primejob.backend.pods import TerminateResult
from primejob.cleanup import terminate_run_pod
from primejob.state import RunRecord


def test_terminate_run_pod_updates_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    import primejob.state as state_mod

    importlib.reload(state_mod)

    record = RunRecord(
        run_id="run-term",
        pod_id="pod-99",
        gpu_type="L40S",
        gpu_count=1,
        country="US",
        provider="test",
        rate_per_hr=1.0,
        script="train.py",
        status="running",
    )
    record.save()

    client = MagicMock()
    messages: list[str] = []

    def fake_terminate_pod(_client, pod_id: str) -> TerminateResult:
        assert pod_id == "pod-99"
        return TerminateResult(success=True)

    monkeypatch.setattr("primejob.cleanup.terminate_pod", fake_terminate_pod)
    monkeypatch.setattr("primejob.cleanup.release_lease", lambda _rid: None)

    terminate_run_pod(
        client,
        record,
        on_status=messages.append,
        cleanup_note="test",
        total_cost=1.23,
    )

    assert record.status == "terminated"
    assert record.ended_at is not None
    assert record.total_cost == 1.23
    assert record.cleanup_note == "test"
    assert any("Terminating pod" in m for m in messages)
