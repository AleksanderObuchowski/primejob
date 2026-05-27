"""Local persistence for run history under ~/.primejob/runs/."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from primejob._atomic import atomic_write_text


STATE_ROOT = Path.home() / ".primejob"
RUNS_DIR = STATE_ROOT / "runs"


@dataclass
class RunRecord:
    run_id: str
    pod_id: str | None
    gpu_type: str
    gpu_count: int
    country: str | None
    provider: str | None
    rate_per_hr: float
    script: str
    args: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    exit_code: int | None = None
    total_cost: float | None = None
    disk_name: str | None = None
    status: str = "running"  # running | finished | failed | terminated
    cleanup_note: str | None = None

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def log_path(self) -> Path:
        return self.dir / "log.txt"

    def ensure_dir(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def save(self) -> None:
        self.ensure_dir()
        atomic_write_text(
            self.manifest_path,
            json.dumps(asdict(self), indent=2, default=str),
            mode=0o600,
        )


def load_run(run_id: str) -> RunRecord:
    path = RUNS_DIR / run_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No run record for {run_id}")
    data = json.loads(path.read_text())
    return RunRecord(**data)


def list_runs(limit: int = 50) -> list[RunRecord]:
    if not RUNS_DIR.exists():
        return []
    records: list[RunRecord] = []
    for child in sorted(RUNS_DIR.iterdir(), reverse=True):
        manifest = child / "manifest.json"
        if not manifest.exists():
            continue
        try:
            records.append(RunRecord(**json.loads(manifest.read_text())))
        except Exception:  # noqa: BLE001 — skip corrupt records
            continue
        if len(records) >= limit:
            break
    return records


def new_run_id() -> str:
    """ULID-like sortable timestamp (ms-precision) + short random."""
    import secrets

    # datetime.utcnow() is deprecated in 3.12; use timezone-aware now().
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
    return f"{ts}-{secrets.token_hex(3)}"
