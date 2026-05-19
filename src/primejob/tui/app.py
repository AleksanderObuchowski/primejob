"""PrimejobApp — the Textual dashboard wrapped around `run_training`.

Two entry points:
  - run_dashboard(client, opts): live dashboard for `primejob run`. Spawns
    run_training in a worker thread; sink events marshal to the UI via
    call_from_thread.
  - attach_dashboard(run_id): view-only dashboard for an existing run, tailing
    its log file and (if still running) polling pod status + nvidia-smi."""
from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer

from primejob.events import ConfirmRequest, EventSink
from primejob.runtime import fmt_elapsed
from primejob.state import RunRecord, load_run
from primejob.tui.screens.help import HelpScreen
from primejob.tui.screens.preflight import PreflightModal
from primejob.tui.screens.summary import SummaryScreen
from primejob.tui.screens.terminate import TerminateModal
from primejob.tui.state import (
    FinalSummary,
    GpuMetric,
    PHASE_ORDER,
    Phase,
    RunMeta,
)
from primejob.tui.theme import PRIME_THEME, THEME_CYCLE
from primejob.tui.widgets.gpus import GpuTable
from primejob.tui.widgets.header import Header
from primejob.tui.widgets.log import LogView
from primejob.tui.widgets.meta import MetaLine
from primejob.tui.widgets.stepper import Stepper
from primejob.tui.workers.log_tail import LogTailer
from primejob.tui.workers.nvidia import NvidiaWorker

if TYPE_CHECKING:
    from prime_cli.api.client import APIClient
    from primejob.backend.ssh import SshEndpoint
    from primejob.run import RunOptions


_TUI_DIR = Path(__file__).parent


class PrimejobApp(App):
    """The dashboard. Has two modes: live (drives run_training) and attach."""

    CSS_PATH = _TUI_DIR / "styles.tcss"
    TITLE = "primejob"

    BINDINGS = [
        Binding("q", "quit_ask", "quit"),
        Binding("ctrl+c", "terminate", "terminate"),
        Binding("slash", "search", "search"),
        Binding("p", "pause", "pause"),
        Binding("question_mark", "help", "help"),
        Binding("e", "edit_log", show=False),
        Binding("o", "open_outputs", show=False),
        Binding("t", "cycle_theme", show=False),
        Binding("g", "toggle_gpus", show=False),
        Binding("s", "copy_ssh", show=False),
        Binding("k", "copy_run_id", show=False),
    ]

    def __init__(
        self,
        *,
        client: "APIClient | None" = None,
        opts: "RunOptions | None" = None,
        record: RunRecord | None = None,
        attach: bool = False,
        exit_on_finish: bool = False,
    ) -> None:
        super().__init__()
        self._client = client
        self._opts = opts
        self._record = record
        self._attach = attach
        self._exit_on_finish = exit_on_finish

        # Initial meta — filled in once run_training reports it.
        if record is not None:
            self._meta = RunMeta(
                run_id=record.run_id,
                script=record.script,
                args=list(record.args),
                gpu_type=record.gpu_type,
                gpu_count=record.gpu_count,
                country=record.country,
                provider=record.provider,
                pod_id=record.pod_id,
            )
        else:
            self._meta = RunMeta(
                run_id="(pending)",
                script=opts.script if opts else "",
                args=list(opts.args) if opts else [],
                gpu_type=opts.gpu or "" if opts else "",
                gpu_count=opts.count if opts else 1,
            )

        self._theme_idx = 0
        self._summary: FinalSummary | None = None
        self._nvidia: NvidiaWorker | None = None
        self._tailer: LogTailer | None = None
        self._terminate_pending = False
        self._run_done = threading.Event()

    # ------------------------------------------------------------- compose

    def compose(self) -> ComposeResult:
        yield Header(self._meta, attach_mode=self._attach)
        yield Stepper()
        yield MetaLine()
        yield GpuTable()
        yield LogView()
        yield Footer()

    def on_mount(self) -> None:
        # Register and apply our custom Prime-Intellect-flavored theme.
        self.register_theme(PRIME_THEME)
        self.theme = THEME_CYCLE[0]
        self._theme_idx = 0

        # GPU panel is hidden by default; revealed only when nvidia-smi reports
        # real metrics (live mode). Attach mode never gets data.
        try:
            self.query_one(GpuTable).add_class("-hidden")
        except Exception:  # noqa: BLE001
            pass

        if self._attach:
            self._start_attach_mode()
        else:
            self._start_live_mode()

    # ------------------------------------------------------------- modes

    def _start_live_mode(self) -> None:
        """Spawn run_training in a worker thread; events marshal to the UI."""
        assert self._client is not None and self._opts is not None
        sink = TuiEventSink(self)
        self.run_worker(
            self._run_training_thread(sink),
            thread=True,
            exclusive=True,
            name="primejob-run",
        )

    def _start_attach_mode(self) -> None:
        assert self._record is not None
        rec = self._record

        # Replay the existing log file.
        log_widget = self.query_one(LogView)
        if rec.log_path.exists():
            for line in rec.log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                stream = "stderr" if line.startswith("[stderr] ") else "stdout"
                payload = line[len("[stderr] "):] if stream == "stderr" else line
                log_widget.append(stream, payload)

        # Mark phase from the record.
        stepper = self.query_one(Stepper)
        if rec.status == "finished":
            stepper.set_phase(Phase.DONE)
        elif rec.status == "failed":
            stepper.set_phase(Phase.RUNNING, failed=True)
        elif rec.status == "terminated":
            stepper.set_phase(Phase.WRAP, failed=True)
        elif rec.status == "running":
            stepper.set_phase(Phase.RUNNING)

        # Meta line.
        meta = self.query_one(MetaLine)
        if rec.status == "running":
            # Live: derive monotonic anchor from wall-clock difference.
            anchor = time.monotonic() - (time.time() - rec.started_at)
            meta.set_cost(
                started_at=anchor,
                rate_per_hr=rec.rate_per_hr,
                spent=rec.total_cost or 0.0,
            )
            meta.set_note("attached (view-only)")
        else:
            # Finished: freeze to recorded elapsed; no live ticker.
            elapsed = (rec.ended_at or rec.started_at) - rec.started_at
            meta.freeze(
                elapsed_s=elapsed,
                rate_per_hr=rec.rate_per_hr,
                spent=rec.total_cost or 0.0,
            )
            meta.set_note("view-only")

        # For still-running runs, keep tailing.
        if rec.status == "running":
            self._tailer = LogTailer(
                rec.log_path,
                on_line=lambda s, l: self.call_from_thread(self._append_log, s, l),
            )
            # Seek to end of file we already replayed — set offset via stop+start trick:
            # Easier: just point tailer at the file; replay above already showed history,
            # but tailer will start from offset 0. We'll re-read what we already showed.
            # For MVP we accept duplicates from tail starting at 0 — refine later by
            # peeking at file size and seeking past it.
            self._tailer.start()

        # Note: nvidia poller requires a live SSH endpoint. We don't store it in
        # RunRecord today; deferred (#3 in PLAN's open questions).

    # ------------------------------------------------------------- worker

    def _run_training_thread(self, sink: "TuiEventSink"):
        # Textual's thread workers want a callable that returns a value when done.
        def target():
            from primejob.run import RunAborted, run_training
            try:
                run_training(self._client, self._opts, sink=sink)
            except RunAborted:
                # User declined cost — exit immediately.
                self.call_from_thread(self.exit, 130)
                return
            except Exception as e:  # noqa: BLE001
                self.call_from_thread(self._show_error_and_finish, str(e))
                return
            finally:
                self._run_done.set()
        return target

    # ------------------------------------------------------------- sink callbacks (main thread)

    def _set_meta(self, meta: RunMeta) -> None:
        self._meta = meta
        self.query_one(Header).refresh_meta(meta)

    def _set_phase(self, phase: Phase, *, failed: bool = False) -> None:
        self.query_one(Stepper).set_phase(phase, failed=failed)

    def _set_cost(self, *, started_at: float, rate_per_hr: float, spent: float) -> None:
        self.query_one(MetaLine).set_cost(
            started_at=started_at, rate_per_hr=rate_per_hr, spent=spent,
        )

    def _set_status_note(self, note: str) -> None:
        self.query_one(MetaLine).set_note(note)

    def _append_log(self, stream: str, line: str) -> None:
        self.query_one(LogView).append(stream, line)

    def _spawn_nvidia(self, endpoint: "SshEndpoint") -> None:
        if self._nvidia is not None:
            return
        gpu_widget = self.query_one(GpuTable)

        def push_metrics(metrics: list[GpuMetric]) -> None:
            def apply():
                gpu_widget.update_metrics(metrics)
                if metrics:
                    gpu_widget.remove_class("-hidden")
            self.call_from_thread(apply)

        def push_error(msg: str) -> None:
            # On any error we just hide the panel.
            self.call_from_thread(gpu_widget.add_class, "-hidden")

        self._nvidia = NvidiaWorker(
            endpoint,
            interval=2.0,
            on_metrics=push_metrics,
            on_error=push_error,
        )
        self._nvidia.start()

    def _show_summary(self, summary: FinalSummary) -> None:
        self._summary = summary
        # Stop the elapsed counter — pin to the final value.
        try:
            self.query_one(MetaLine).freeze(
                elapsed_s=summary.elapsed_s,
                rate_per_hr=0.0,
                spent=summary.total_cost,
            )
        except Exception:  # noqa: BLE001
            pass
        if self._exit_on_finish:
            self.exit(summary.exit_code or 0)
            return
        self.push_screen(SummaryScreen(summary), self._on_summary_dismiss)

    def _on_summary_dismiss(self, _result) -> None:
        self.exit(self._summary.exit_code if self._summary else 0)

    def _show_error_and_finish(self, msg: str) -> None:
        self._append_log("stderr", f"primejob: {msg}")
        summary = FinalSummary(
            exit_code=1,
            status="failed",
            elapsed_s=0.0,
            total_cost=0.0,
            outputs_path=None,
            last_error=[msg],
        )
        self._show_summary(summary)

    # ------------------------------------------------------------- actions

    def action_quit_ask(self) -> None:
        # If the run is finished, `q` quits immediately.
        if self._run_done.is_set() or self._attach:
            self.exit(self._summary.exit_code if self._summary else 0)
            return
        # Otherwise treat like terminate.
        self.action_terminate()

    def action_terminate(self) -> None:
        if self._attach:
            # Attach is view-only; just exit.
            self.exit(0)
            return
        if self._terminate_pending:
            return
        self._terminate_pending = True
        spent = 0.0
        rate = 0.0
        try:
            meta_widget = self.query_one(MetaLine)
            rate = meta_widget._rate_per_hr
            spent = meta_widget._spent
        except Exception:  # noqa: BLE001
            pass
        modal = TerminateModal(
            run_id=self._meta.run_id,
            pod_id=self._meta.pod_id,
            spent=spent,
            rate=rate,
        )
        self.push_screen(modal, self._on_terminate_result)

    def _on_terminate_result(self, result: bool | None) -> None:
        self._terminate_pending = False
        if result is None:
            # Force quit. Mark orphan, exit immediately.
            self._mark_orphan_pod()
            self.exit(130)
        elif result:
            # Graceful terminate: kick the pod, let run_training wind down.
            self._kick_pod_terminate()
        # else: cancelled — back to dashboard

    def _mark_orphan_pod(self) -> None:
        if not self._meta.pod_id or not self._meta.run_id:
            return
        try:
            from primejob.state import RUNS_DIR
            (RUNS_DIR / self._meta.run_id).mkdir(parents=True, exist_ok=True)
            (RUNS_DIR / self._meta.run_id / "orphaned.txt").write_text(
                f"{self._meta.pod_id}\n"
            )
        except Exception:  # noqa: BLE001
            pass

    def _kick_pod_terminate(self) -> None:
        if not self._meta.pod_id or self._client is None:
            return
        try:
            from primejob.backend.pods import terminate as kill_pod
            kill_pod(self._client, self._meta.pod_id)
            self._append_log("stdout", f"primejob: terminate requested for pod {self._meta.pod_id}")
        except Exception as e:  # noqa: BLE001
            self._append_log("stderr", f"primejob: terminate failed: {e}")

    def action_search(self) -> None:
        self.query_one(LogView).open_search()

    def action_pause(self) -> None:
        self.query_one(LogView).toggle_pause()

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_edit_log(self) -> None:
        if not self._meta.run_id:
            return
        log_path = Path.home() / ".primejob" / "runs" / self._meta.run_id / "log.txt"
        if not log_path.exists():
            return
        editor = os.environ.get("EDITOR") or "vi"
        with self.suspend():
            subprocess.run([editor, str(log_path)])

    def action_open_outputs(self) -> None:
        if not self._meta.run_id:
            return
        outputs = Path.cwd() / "outputs" / self._meta.run_id
        if not outputs.exists():
            return
        # Cross-platform open
        if os.name == "nt":
            os.startfile(str(outputs))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(
                ["open" if os.uname().sysname == "Darwin" else "xdg-open", str(outputs)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def action_cycle_theme(self) -> None:
        self._theme_idx = (self._theme_idx + 1) % len(THEME_CYCLE)
        try:
            self.theme = THEME_CYCLE[self._theme_idx]
        except Exception:  # noqa: BLE001
            # Skip themes that aren't installed in this textual version.
            pass

    def action_toggle_gpus(self) -> None:
        widget = self.query_one(GpuTable)
        if widget.has_class("-hidden"):
            widget.remove_class("-hidden")
        else:
            widget.add_class("-hidden")

    def action_copy_ssh(self) -> None:
        # Stub — proper clipboard support depends on platform tools (pbcopy / xclip).
        # Just print the SSH command into the log so the user can copy by selection.
        if not self._meta.pod_id:
            return
        self._append_log("stdout", f"primejob: pod_id={self._meta.pod_id} (use `prime pods ssh {self._meta.pod_id}`)")

    def action_copy_run_id(self) -> None:
        self._append_log("stdout", f"primejob: run_id={self._meta.run_id}")

    # ------------------------------------------------------------- shutdown

    async def on_unmount(self) -> None:
        if self._nvidia is not None:
            self._nvidia.stop()
        if self._tailer is not None:
            self._tailer.stop()


class TuiEventSink(EventSink):
    """Marshals run_training events onto the Textual main thread."""

    def __init__(self, app: PrimejobApp) -> None:
        self._app = app
        self._log_fh = None

    # ------------------------------------------------------------ status

    def status(self, msg: str) -> None:
        self._app.call_from_thread(self._app._append_log, "stdout", f"› {msg}")

    def status_note(self, note: str) -> None:
        self._app.call_from_thread(self._app._set_status_note, note.strip())

    def log_line(self, stream: str, line: str) -> None:
        self._app.call_from_thread(self._app._append_log, stream, line)
        if self._log_fh is not None:
            try:
                prefix = "" if stream == "stdout" else "[stderr] "
                self._log_fh.write(prefix + line + "\n")
                self._log_fh.flush()
            except Exception:  # noqa: BLE001
                pass

    def phase(self, phase, *, failed: bool = False) -> None:
        self._app.call_from_thread(self._app._set_phase, phase, failed=failed)

    def meta(self, meta) -> None:
        # Re-open log file once we know run_id (first meta call has it).
        if self._log_fh is None and meta.run_id and meta.run_id != "(pending)":
            try:
                path = Path.home() / ".primejob" / "runs" / meta.run_id / "log.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                self._log_fh = path.open("a", encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        self._app.call_from_thread(self._app._set_meta, meta)

    def cost(self, *, started_at: float, rate_per_hr: float, spent: float) -> None:
        self._app.call_from_thread(
            self._app._set_cost,
            started_at=started_at,
            rate_per_hr=rate_per_hr,
            spent=spent,
        )

    def ssh_ready(self, endpoint) -> None:
        self._app.call_from_thread(self._app._spawn_nvidia, endpoint)

    def confirm(self, request: ConfirmRequest) -> bool:
        event = threading.Event()
        result: list[bool] = []

        def on_dismiss(answer):
            result.append(bool(answer))
            event.set()

        def push():
            self._app.push_screen(
                PreflightModal(
                    gpu=request.gpu_type,
                    count=request.gpu_count,
                    rate_per_hr=request.rate_per_hr,
                    provider=request.provider,
                    country=request.country,
                ),
                on_dismiss,
            )
        self._app.call_from_thread(push)
        event.wait()
        return result[0] if result else False

    def finish(self, summary) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_fh = None
        self._app.call_from_thread(self._app._show_summary, summary)


# ====================================================================== entry points


def run_dashboard(client, opts, *, exit_on_finish: bool = False) -> int:
    """Open the live dashboard for a `primejob run`. Returns the run's exit code."""
    app = PrimejobApp(client=client, opts=opts, exit_on_finish=exit_on_finish)
    result = app.run()
    if isinstance(result, int):
        return result
    return 0


def attach_dashboard(run_id: str) -> int:
    """Open the dashboard in view-only mode for an existing run_id."""
    record = load_run(run_id)
    app = PrimejobApp(record=record, attach=True)
    result = app.run()
    if isinstance(result, int):
        return result
    return 0
