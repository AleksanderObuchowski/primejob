"""Shared pod termination and manifest finalization for runs."""
from __future__ import annotations

import time
from typing import Callable

from prime_cli.api.client import APIClient

from primejob.backend.pods import TerminateResult, terminate_pod
from primejob.state import RunRecord
from primejob.watchdog import release_lease


def terminate_run_pod(
    client: APIClient,
    record: RunRecord,
    *,
    on_status: Callable[[str], None] | None = None,
    cleanup_note: str | None = None,
    total_cost: float | None = None,
    final_status: str = "terminated",
) -> TerminateResult | None:
    """Terminate remote pod, release lease, and finalize local manifest."""
    release_lease(record.run_id)

    result: TerminateResult | None = None
    if record.pod_id:
        if on_status:
            on_status(f"Terminating pod {record.pod_id}...")
        result = terminate_pod(client, record.pod_id)
        if on_status:
            if result.success:
                on_status(f"  → pod {record.pod_id} terminate requested")
            else:
                on_status(f"  → terminate failed: {result.error}")

    if record.ended_at is None:
        record.ended_at = time.time()
    if total_cost is not None:
        record.total_cost = total_cost
    if record.status == "running":
        record.status = final_status
    if cleanup_note:
        record.cleanup_note = cleanup_note
    elif result is not None and not result.success:
        record.cleanup_note = f"terminate failed: {result.error}"
    record.save()
    return result


def force_terminate_run(
    client: APIClient,
    run_id: str,
    *,
    on_status: Callable[[str], None] | None = None,
) -> RunRecord:
    """Terminate pod for a run_id and mark manifest terminated (TUI/CLI recovery)."""
    from primejob.state import load_run

    record = load_run(run_id)
    terminate_run_pod(
        client,
        record,
        on_status=on_status,
        cleanup_note="force terminate",
    )
    return record
