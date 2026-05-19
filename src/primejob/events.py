"""EventSink protocol — the surface `run_training` uses to talk to the UI.

Two implementations:
  - `ConsoleSink` (default) preserves the original plain-mode behavior: prints
    status lines, streams remote stdout/stderr to local stdout + log file,
    drives a threaded status bar with cost.
  - `TuiEventSink` (in primejob.tui.app) bridges events into a running Textual
    app via `call_from_thread`.

`run_training` constructs no UI of its own — it just calls sink methods.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from primejob.backend.ssh import SshEndpoint
    from primejob.tui.state import FinalSummary, Phase, RunMeta


@dataclass
class ConfirmRequest:
    prompt: str  # one-liner suitable for plain mode `input()`
    gpu_type: str
    gpu_count: int
    rate_per_hr: float
    provider: str | None
    country: str | None


class EventSink(Protocol):
    """Everything `run_training` reports during a run. Default impls are no-ops
    so a sink can override only what it cares about."""

    def status(self, msg: str) -> None: ...
    """Primejob status message (e.g. 'Picking cheapest H100...')."""

    def status_note(self, note: str) -> None: ...
    """Short transient status (e.g. 'pod=provisioning install=12% rate=$0.00/h')."""

    def log_line(self, stream: str, line: str) -> None: ...
    """A line of stdout/stderr from the remote training script."""

    def phase(self, phase: "Phase", *, failed: bool = False) -> None: ...
    """Lifecycle phase transition."""

    def meta(self, meta: "RunMeta") -> None: ...
    """Run metadata once known (run_id, gpu, provider, country, pod_id...)."""

    def cost(self, *, started_at: float, rate_per_hr: float, spent: float) -> None: ...
    """Latest cost snapshot — emitted periodically."""

    def ssh_ready(self, endpoint: "SshEndpoint") -> None: ...
    """Pod is reachable over SSH — TUI may spawn nvidia-smi poller."""

    def confirm(self, request: ConfirmRequest) -> bool: ...
    """Block until the user accepts/rejects spawning the pod."""

    def finish(self, summary: "FinalSummary") -> None: ...
    """Run is over. TUI shows summary screen, plain prints a one-line recap."""


class ConsoleSink:
    """Plain-mode sink. Drop-in replacement for the previous in-line behavior."""

    def __init__(self, *, log_file: Path | None = None, yes: bool = False) -> None:
        from rich.console import Console
        self._console = Console()
        self._log_file = log_file
        self._fh = None
        self._yes = yes
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._fh = log_file.open("a", encoding="utf-8")

    # ------------------------------------------------------------ EventSink

    def status(self, msg: str) -> None:
        self._console.print(f"[bold]›[/bold] {msg}")

    def status_note(self, note: str) -> None:
        sys.stderr.write(f"\033[2m{note}\033[0m\n")
        sys.stderr.flush()

    def log_line(self, stream: str, line: str) -> None:
        prefix = "" if stream == "stdout" else "[stderr] "
        sys.stdout.write(prefix + line + "\n")
        sys.stdout.flush()
        if self._fh is not None:
            self._fh.write(prefix + line + "\n")
            self._fh.flush()

    def phase(self, phase, *, failed: bool = False) -> None:
        # Plain mode doesn't visualize the stepper; status() messages cover it.
        pass

    def meta(self, meta) -> None:
        pass

    def cost(self, *, started_at: float, rate_per_hr: float, spent: float) -> None:
        # ConsoleSink ignores per-tick cost; the StatusBar thread in runtime.py
        # publishes a digest line every 30s instead (legacy behavior).
        pass

    def ssh_ready(self, endpoint) -> None:
        pass

    def confirm(self, request: ConfirmRequest) -> bool:
        if self._yes:
            return True
        try:
            answer = input(request.prompt).strip().lower()
        except EOFError:
            return False
        return answer in {"y", "yes"}

    def finish(self, summary) -> None:
        if summary.last_error:
            self._console.print("[bold red]Last error:[/bold red]")
            for line in summary.last_error[-10:]:
                self._console.print(f"  {line}")

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None
