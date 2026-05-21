"""Visual demo of the primejob TUI — no Prime API calls, no SSH, no cost.

Fakes a training run end-to-end:
  preflight → provision → install → upload → running → wrap → finished

Pumps simulated nvidia-smi metrics + log lines through the real widgets so you
can see the GPU table, stepper animation, meta-line, log scrolling, and
keybindings. Run with:

    uv run python examples/demo_tui.py
    uv run python examples/demo_tui.py --fast   # shorter phases (for README GIF)
"""
from __future__ import annotations

import argparse
import random
import threading
import time
from datetime import datetime, timezone

from primejob.state import RUNS_DIR, RunRecord
from primejob.tui.app import PrimejobApp
from primejob.tui.state import GpuMetric, Phase
from primejob.tui.widgets.gpus import GpuTable
from primejob.tui.widgets.log import LogView
from primejob.tui.widgets.meta import MetaLine
from primejob.tui.widgets.stepper import Stepper


PHASES = [Phase.PREFLIGHT, Phase.PROVISION, Phase.INSTALL, Phase.UPLOAD, Phase.RUNNING, Phase.WRAP]
PHASE_DURATIONS_S = {
    Phase.PREFLIGHT: 2.0,
    Phase.PROVISION: 4.0,
    Phase.INSTALL: 4.0,
    Phase.UPLOAD: 3.0,
    Phase.RUNNING: 60.0,   # the bulk of the demo
    Phase.WRAP: 3.0,
}

FAKE_LOG_LINES = [
    ("stdout", "› Picking cheapest H100 ×4 (country=US)..."),
    ("stdout", "›   → H100_80GB ×4 @ datacrunch (US, FIN-01) $9.7200/h"),
    ("stdout", "› Creating pod (run_id=20260520T000000-demo01)..."),
    ("stdout", "› Waiting for pod to become running..."),
    ("stdout", "› SSH up at root@65.108.33.93:22"),
    ("stdout", "› Uploading src tarball..."),
    ("stdout", "›   → 142 files, 4.7 MB"),
    ("stdout", "› Installing uv on the pod..."),
    ("stdout", "› Running `uv sync`..."),
    ("stderr", "Resolved 87 packages in 1.23s"),
    ("stderr", "Downloaded torch"),
    ("stderr", "Downloaded transformers"),
    ("stderr", "Downloaded accelerate"),
    ("stderr", "Downloaded peft"),
    ("stderr", "Installed 87 packages in 6.42s"),
    ("stdout", "› Running: train.py --epochs 10 --lr 3e-4"),
    ("stdout", "[2026-05-20 00:00:14] config loaded: bs=16 lr=3e-4 epochs=10"),
    ("stdout", "loading checkpoint from /mnt/dataset/base/model.safetensors"),
    ("stdout", "model parameters: 7.24B (trainable: 1.92M)"),
    ("stdout", "epoch 1/10  step    1/250  loss=2.4123  lr=3.0e-04  gpu_mem=58.2GB"),
    ("stdout", "epoch 1/10  step   50/250  loss=1.8421  lr=3.0e-04  gpu_mem=63.1GB"),
    ("stdout", "epoch 1/10  step  100/250  loss=1.5093  lr=3.0e-04  gpu_mem=64.7GB"),
    ("stdout", "epoch 1/10  step  150/250  loss=1.3210  lr=3.0e-04  gpu_mem=65.0GB"),
    ("stderr", "Warning: gradient norm exceeded threshold (4.2 > 4.0), clipping"),
    ("stdout", "epoch 1/10  step  200/250  loss=1.1820  lr=3.0e-04  gpu_mem=65.1GB"),
    ("stdout", "epoch 1/10  step  250/250  loss=1.0420  lr=3.0e-04  gpu_mem=65.1GB"),
    ("stdout", "epoch 1/10  val_loss=0.9821  val_acc=0.6420"),
    ("stdout", "checkpoint saved to outputs/epoch_1.pt"),
    ("stdout", "epoch 2/10  step    1/250  loss=0.9876  lr=2.7e-04  gpu_mem=65.3GB"),
    ("stdout", "epoch 2/10  step   50/250  loss=0.8521  lr=2.7e-04  gpu_mem=65.2GB"),
    ("stdout", "epoch 2/10  step  100/250  loss=0.7820  lr=2.7e-04  gpu_mem=65.4GB"),
    ("stdout", "epoch 2/10  step  150/250  loss=0.7321  lr=2.7e-04  gpu_mem=65.3GB"),
    ("stdout", "epoch 2/10  step  200/250  loss=0.6921  lr=2.7e-04  gpu_mem=65.4GB"),
    ("stdout", "epoch 2/10  step  250/250  loss=0.6620  lr=2.7e-04  gpu_mem=65.4GB"),
    ("stdout", "epoch 2/10  val_loss=0.7120  val_acc=0.7820"),
    ("stdout", "epoch 3/10  step  100/250  loss=0.5821  lr=2.4e-04  gpu_mem=65.5GB"),
    ("stdout", "epoch 3/10  step  200/250  loss=0.5320  lr=2.4e-04  gpu_mem=65.4GB"),
    ("stdout", "epoch 3/10  val_loss=0.6020  val_acc=0.8210"),
]


def _fake_gpu_metric(idx: int, t: float) -> GpuMetric:
    """Generate a plausible H100 metric with some noise."""
    # Each GPU slightly different baseline; util oscillates with a sine, mem stable.
    base_util = 82 + idx * 2
    util = max(0.0, min(100.0, base_util + 10 * random.random() - 5 + 4 * (1 if int(t * 0.5) % 4 == idx else 0)))
    mem_total = 81559.0  # 80GB
    mem_used = 64000 + idx * 800 + random.randint(-300, 300)
    temp = 68 + idx * 2 + random.uniform(-2, 2)
    power = 380 + idx * 8 + random.uniform(-15, 15)
    throttle = ""
    # Occasional throttle for visual interest
    if idx == 2 and int(t) % 23 == 0:
        throttle = "SW thermal"
    return GpuMetric(
        index=idx,
        util_pct=util,
        mem_used_mb=mem_used,
        mem_total_mb=mem_total,
        temp_c=temp,
        power_w=power,
        throttle=throttle,
    )


class DemoApp(PrimejobApp):
    """PrimejobApp wired up with fakes — no API client, no SSH, no costs."""

    def __init__(self, *, fast: bool = False) -> None:
        if fast:
            _apply_fast_durations()
        record = self._make_fake_record()
        # Keep attach=True so PrimejobApp.on_mount goes the attach path, which
        # we then neuter by overriding `_start_attach_mode` below. (Textual
        # walks the MRO and calls every on_mount it finds.)
        super().__init__(record=record, attach=True)
        self._stop_event = threading.Event()
        self._demo_threads: list[threading.Thread] = []

    def _start_attach_mode(self) -> None:  # type: ignore[override]
        """Override parent: don't replay the (nonexistent) log file or freeze meta."""
        return

    @staticmethod
    def _make_fake_record() -> RunRecord:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return RunRecord(
            run_id=f"{ts}-demo01",
            pod_id="pod_demo_123",
            gpu_type="H100_80GB",
            gpu_count=4,
            country="US",
            provider="datacrunch",
            rate_per_hr=9.72,
            script="train.py",
            args=["--epochs", "10", "--lr", "3e-4"],
            status="running",
        )

    # --- override on_mount: skip both live (no client) and attach paths

    def on_mount(self) -> None:  # type: ignore[override]
        # Register theme + apply default (same as parent).
        from primejob.tui.theme import PRIME_THEME, THEME_CYCLE
        self.register_theme(PRIME_THEME)
        self.theme = THEME_CYCLE[0]
        self._theme_idx = 0

        # Hide GPU panel — will be revealed once running phase starts.
        self.query_one(GpuTable).add_class("-hidden")

        # Spin up fake workers.
        for fn in (self._phase_driver, self._log_driver, self._gpu_driver, self._cost_driver):
            t = threading.Thread(target=fn, daemon=True, name=f"demo-{fn.__name__}")
            t.start()
            self._demo_threads.append(t)

    # --- workers (run in threads, marshal via call_from_thread)

    def _phase_driver(self) -> None:
        t0 = time.time()
        for phase in PHASES:
            if self._stop_event.is_set():
                return
            self.call_from_thread(self.query_one(Stepper).set_phase, phase)
            self.call_from_thread(
                self.query_one(MetaLine).set_note,
                f"phase: {phase.value}",
            )
            if self._stop_event.wait(PHASE_DURATIONS_S[phase]):
                return
        # Done — freeze meta line.
        elapsed = time.time() - t0
        self.call_from_thread(
            self.query_one(MetaLine).freeze,
            elapsed_s=elapsed,
            rate_per_hr=0.0,
            spent=9.72 * elapsed / 3600,
        )
        self.call_from_thread(self.query_one(Stepper).set_phase, Phase.DONE)

    def _log_driver(self) -> None:
        log = self.query_one(LogView)
        for stream, line in FAKE_LOG_LINES:
            if self._stop_event.is_set():
                return
            self.call_from_thread(log.append, stream, line)
            time.sleep(0.35 + random.random() * 0.5)
        # Keep pushing fake "training" lines so the log keeps moving.
        epoch = 4
        while not self._stop_event.is_set():
            for step in range(50, 251, 50):
                if self._stop_event.is_set():
                    return
                line = (
                    f"epoch {epoch}/10  step  {step:3d}/250  "
                    f"loss={0.45 - epoch * 0.04 + random.random() * 0.02:.4f}  "
                    f"lr={(3e-4) * (0.9 ** epoch):.2e}  "
                    f"gpu_mem={65 + random.random() * 0.5:.1f}GB"
                )
                self.call_from_thread(log.append, "stdout", line)
                time.sleep(0.6 + random.random() * 0.4)
            epoch += 1
            if epoch > 10:
                break

    def _gpu_driver(self) -> None:
        # Wait until RUNNING phase begins so we don't show GPU data prematurely.
        delay = sum(PHASE_DURATIONS_S[p] for p in [Phase.PREFLIGHT, Phase.PROVISION, Phase.INSTALL, Phase.UPLOAD])
        if self._stop_event.wait(delay):
            return
        gpu_widget = self.query_one(GpuTable)
        self.call_from_thread(gpu_widget.remove_class, "-hidden")
        t0 = time.time()
        while not self._stop_event.is_set():
            t = time.time() - t0
            metrics = [_fake_gpu_metric(i, t) for i in range(4)]
            self.call_from_thread(gpu_widget.update_metrics, metrics)
            if self._stop_event.wait(2.0):
                return

    def _cost_driver(self) -> None:
        # Bill only during install + upload + running + wrap (~4+3+60+3 = 70s).
        billable_delay = PHASE_DURATIONS_S[Phase.PREFLIGHT] + PHASE_DURATIONS_S[Phase.PROVISION]
        if self._stop_event.wait(billable_delay):
            return
        anchor = time.monotonic()
        rate = 9.72
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - anchor
            spent = rate * elapsed / 3600
            self.call_from_thread(
                self.query_one(MetaLine).set_cost,
                started_at=anchor,
                rate_per_hr=rate,
                spent=spent,
            )
            if self._stop_event.wait(1.0):
                return

    async def on_unmount(self) -> None:  # type: ignore[override]
        self._stop_event.set()
        await super().on_unmount()


def _apply_fast_durations() -> None:
    """Shrink phase timers so tapegif/README captures finish in ~30s."""
    PHASE_DURATIONS_S[Phase.PREFLIGHT] = 1.5
    PHASE_DURATIONS_S[Phase.PROVISION] = 2.5
    PHASE_DURATIONS_S[Phase.INSTALL] = 2.5
    PHASE_DURATIONS_S[Phase.UPLOAD] = 2.0
    PHASE_DURATIONS_S[Phase.RUNNING] = 14.0
    PHASE_DURATIONS_S[Phase.WRAP] = 2.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="primejob TUI visual demo (no API/SSH)")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="shorter phase timers for GIF recording (docs/assets/demo.gif)",
    )
    args = parser.parse_args()
    if args.fast:
        _apply_fast_durations()
    DemoApp().run()
