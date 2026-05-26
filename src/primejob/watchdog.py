"""Detached watchdog: terminate pods when the controlling process dies."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from primejob.state import RUNS_DIR, RunRecord, load_run

LEASE_FILENAME = "lease.json"
DEFAULT_STALE_GRACE_S = 45.0
WATCHDOG_POLL_S = 5.0


@dataclass
class Lease:
    run_id: str
    pod_id: str
    parent_pid: int
    heartbeat_at: float
    released: bool = False
    released_at: float | None = None

    @property
    def path(self) -> Path:
        return RUNS_DIR / self.run_id / LEASE_FILENAME


def _lease_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / LEASE_FILENAME


def read_lease(run_id: str) -> Lease | None:
    path = _lease_path(run_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return Lease(**data)
    except Exception:  # noqa: BLE001
        return None


def _write_lease(lease: Lease) -> None:
    lease.path.parent.mkdir(parents=True, exist_ok=True)
    lease.path.write_text(json.dumps(asdict(lease), indent=2))


def create_lease(run_id: str, pod_id: str, *, parent_pid: int | None = None) -> Lease:
    lease = Lease(
        run_id=run_id,
        pod_id=pod_id,
        parent_pid=parent_pid if parent_pid is not None else os.getpid(),
        heartbeat_at=time.time(),
    )
    _write_lease(lease)
    return lease


def heartbeat(run_id: str) -> None:
    """Refresh lease heartbeat (no-op if lease missing or released)."""
    lease = read_lease(run_id)
    if lease is None or lease.released:
        return
    lease.heartbeat_at = time.time()
    _write_lease(lease)


def release_lease(run_id: str) -> None:
    """Mark lease released so the watchdog exits without terminating the pod."""
    lease = read_lease(run_id)
    if lease is None:
        return
    lease.released = True
    lease.released_at = time.time()
    _write_lease(lease)


def lease_is_stale(lease: Lease, *, grace_s: float = DEFAULT_STALE_GRACE_S) -> bool:
    if lease.released:
        return False
    return (time.time() - lease.heartbeat_at) > grace_s


def _parent_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_watchdog(run_id: str, pod_id: str) -> None:
    """Spawn a detached watchdog process for this run."""
    parent_pid = os.getpid()
    cmd = [
        sys.executable,
        "-m",
        "primejob.watchdog",
        run_id,
        pod_id,
        str(parent_pid),
    ]
    subprocess.Popen(
        cmd,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def _finalize_run_terminated(
    run_id: str,
    *,
    cleanup_note: str,
    total_cost: float | None = None,
) -> None:
    try:
        record = load_run(run_id)
    except FileNotFoundError:
        return
    if record.ended_at is None:
        record.ended_at = time.time()
    if record.status == "running":
        record.status = "terminated"
    record.cleanup_note = cleanup_note
    if total_cost is not None:
        record.total_cost = total_cost
    record.save()


def watchdog_cleanup(
    run_id: str,
    pod_id: str,
    *,
    reason: str,
) -> None:
    """Terminate pod and update local manifest (idempotent)."""
    release_lease(run_id)
    from primejob.auth import get_client
    from primejob.backend.pods import terminate_pod

    try:
        client = get_client()
        result = terminate_pod(client, pod_id)
        note = f"watchdog:{reason}"
        if not result.success:
            note = f"{note}; terminate failed: {result.error}"
        _finalize_run_terminated(run_id, cleanup_note=note)
    except Exception as e:  # noqa: BLE001
        _finalize_run_terminated(
            run_id,
            cleanup_note=f"watchdog:{reason}; client error: {e}",
        )


def run_watchdog_loop(
    run_id: str,
    pod_id: str,
    parent_pid: int,
    *,
    stale_grace_s: float = DEFAULT_STALE_GRACE_S,
) -> int:
    """Poll until parent dies, heartbeat goes stale, or lease is released."""
    while True:
        lease = read_lease(run_id)
        if lease is None:
            return 0
        if lease.released:
            return 0
        if not _parent_alive(parent_pid):
            watchdog_cleanup(run_id, pod_id, reason="parent_dead")
            return 0
        if lease_is_stale(lease, grace_s=stale_grace_s):
            watchdog_cleanup(run_id, pod_id, reason="stale_heartbeat")
            return 0
        time.sleep(WATCHDOG_POLL_S)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 3:
        print("usage: primejob.watchdog <run_id> <pod_id> <parent_pid>", file=sys.stderr)
        return 2
    run_id, pod_id, parent_pid_s = args[0], args[1], args[2]
    return run_watchdog_loop(run_id, pod_id, int(parent_pid_s))


if __name__ == "__main__":
    raise SystemExit(main())
