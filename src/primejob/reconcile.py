"""Detect stale local runs and optionally reconcile with remote pod state."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from prime_cli.api.client import APIClient

from primejob.backend.pods import FAILED_STATES, RUNNING_STATES, get_status
from primejob.state import RUNS_DIR, RunRecord
from primejob.watchdog import DEFAULT_STALE_GRACE_S, lease_is_stale, read_lease

ORPHAN_MARKER = "orphaned.txt"


@dataclass
class RunHealth:
    run_id: str
    local_status: str
    stale: bool
    reason: str | None
    remote_status: str | None = None
    remote_active: bool | None = None
    has_orphan_marker: bool = False
    lease_stale: bool = False
    pod_id: str | None = None


def _orphan_marker_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / ORPHAN_MARKER


def assess_run(
    record: RunRecord,
    client: APIClient | None = None,
    *,
    check_remote: bool = False,
    stale_grace_s: float = DEFAULT_STALE_GRACE_S,
) -> RunHealth:
    """Classify whether a local run record looks stale or still billing remotely."""
    orphan = _orphan_marker_path(record.run_id).exists()
    lease = read_lease(record.run_id)
    lease_stale = lease is not None and lease_is_stale(lease, grace_s=stale_grace_s)

    remote_status: str | None = None
    remote_active: bool | None = None
    if check_remote and client is not None and record.pod_id:
        try:
            live = get_status(client, record.pod_id)
            remote_status = (live.status or "").lower() or None
            remote_active = remote_status in RUNNING_STATES
        except Exception:  # noqa: BLE001
            remote_active = None

    stale = False
    reason: str | None = None

    if record.status != "running" or record.ended_at is not None:
        return RunHealth(
            run_id=record.run_id,
            local_status=record.status,
            stale=False,
            reason=None,
            remote_status=remote_status,
            remote_active=remote_active,
            has_orphan_marker=orphan,
            lease_stale=lease_stale,
            pod_id=record.pod_id,
        )

    if orphan:
        stale = True
        reason = "orphan marker (force-quit without cleanup)"
    elif lease_stale:
        stale = True
        reason = "lease heartbeat stale (local process likely dead)"
    elif check_remote and remote_active is True:
        stale = True
        reason = f"remote pod still {remote_status or 'active'} while local run is running"
    elif check_remote and remote_active is False and remote_status in FAILED_STATES:
        stale = True
        reason = f"local run still running but remote pod is {remote_status}"

    # Long-running manifest with empty log often means abrupt client death.
    if not stale and record.log_path.exists():
        if record.log_path.stat().st_size == 0 and (time.time() - record.started_at) > 600:
            stale = True
            reason = "no log output after 10+ minutes"

    return RunHealth(
        run_id=record.run_id,
        local_status=record.status,
        stale=stale,
        reason=reason,
        remote_status=remote_status,
        remote_active=remote_active,
        has_orphan_marker=orphan,
        lease_stale=lease_stale,
        pod_id=record.pod_id,
    )


def format_status_label(health: RunHealth) -> str:
    if not health.stale:
        return health.local_status
    if health.remote_active:
        return f"{health.local_status} [red](billing?)[/red]"
    return f"{health.local_status} [yellow](stale)[/yellow]"
