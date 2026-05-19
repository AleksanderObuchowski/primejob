"""Model objects for the TUI dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    PREFLIGHT = "preflight"
    PROVISION = "provision"
    INSTALL = "install"
    UPLOAD = "upload"
    RUNNING = "running"
    WRAP = "wrap"
    DONE = "done"
    FAILED = "failed"


PHASE_ORDER: list[Phase] = [
    Phase.PREFLIGHT,
    Phase.PROVISION,
    Phase.INSTALL,
    Phase.UPLOAD,
    Phase.RUNNING,
    Phase.WRAP,
]


PHASE_LABELS: dict[Phase, str] = {
    Phase.PREFLIGHT: "preflight",
    Phase.PROVISION: "provision",
    Phase.INSTALL: "install",
    Phase.UPLOAD: "upload",
    Phase.RUNNING: "running",
    Phase.WRAP: "wrap",
}


@dataclass
class GpuMetric:
    index: int
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    temp_c: float
    power_w: float
    throttle: str = ""  # human-readable state ("", "SW throttle", "HW slowdown", etc.)


@dataclass
class RunMeta:
    run_id: str
    script: str
    args: list[str] = field(default_factory=list)
    gpu_type: str = ""
    gpu_count: int = 1
    country: str | None = None
    provider: str | None = None
    pod_id: str | None = None


@dataclass
class FinalSummary:
    exit_code: int | None
    status: str  # finished | failed | terminated
    elapsed_s: float
    total_cost: float
    outputs_path: str | None
    last_error: list[str] = field(default_factory=list)  # grep'd error lines


def script_label(meta: RunMeta) -> str:
    """Display string: 'train.py --epochs 10' (truncated if long)."""
    if not meta.script:
        return "(no script)"
    parts = [meta.script, *meta.args]
    joined = " ".join(parts)
    return joined if len(joined) <= 60 else joined[:57] + "..."


def gpu_badge(meta: RunMeta) -> str:
    """Display string: '[H100×1 US datacrunch]'."""
    if not meta.gpu_type:
        return ""
    bits = [f"{meta.gpu_type}×{meta.gpu_count}"]
    if meta.country:
        bits.append(meta.country)
    if meta.provider:
        bits.append(meta.provider)
    return "[" + " ".join(bits) + "]"
