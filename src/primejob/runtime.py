"""Cost tracking, live status display, signal handling for `primejob run`."""
from __future__ import annotations

import atexit
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable


@dataclass
class CostTracker:
    rate_per_hr: float
    started_at: float = field(default_factory=time.monotonic)

    def update_rate(self, new_rate: float) -> None:
        # Rough: if rate changes mid-flight, we just swap; for MVP this is fine.
        self.rate_per_hr = new_rate

    def elapsed(self) -> timedelta:
        return timedelta(seconds=time.monotonic() - self.started_at)

    def spent(self) -> float:
        return (self.rate_per_hr / 3600.0) * (time.monotonic() - self.started_at)


def fmt_elapsed(td: timedelta) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


class StatusBar:
    """Periodically prints `[run_id] 12m34s | $2.43/h | spent $0.51` to a console.

    Optional `on_tick` lets a sink also receive cost updates on the same cadence
    (without parsing the rendered string)."""

    def __init__(
        self,
        run_id: str,
        tracker: CostTracker,
        printer: Callable[[str], None],
        interval: float = 30.0,
        *,
        on_tick: Callable[[], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.tracker = tracker
        self.printer = printer
        self.interval = interval
        self.on_tick = on_tick
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="primejob-status")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def render(self) -> str:
        return (
            f"[{self.run_id}] elapsed={fmt_elapsed(self.tracker.elapsed())} "
            f"rate=${self.tracker.rate_per_hr:.4f}/h spent=${self.tracker.spent():.4f}"
        )

    def _loop(self) -> None:
        # Skip first tick — give the job a few seconds to print initial output.
        if self._stop.wait(self.interval):
            return
        while not self._stop.is_set():
            try:
                self.printer(self.render())
            except Exception:  # noqa: BLE001 — never let status thread kill main flow
                pass
            if self.on_tick is not None:
                try:
                    self.on_tick()
                except Exception:  # noqa: BLE001
                    pass
            if self._stop.wait(self.interval):
                return


# Process-wide active cleanup for runs where signal handlers cannot register
# (e.g. Textual worker thread). Main thread SIGINT/SIGTERM still fire this.
_active_cleanup: Callable[[], None] | None = None
_active_lock = threading.Lock()


def register_active_cleanup(cleanup: Callable[[], None] | None) -> None:
    """Register the current run's cleanup for main-thread signal delivery."""
    global _active_cleanup
    with _active_lock:
        _active_cleanup = cleanup


def _dispatch_active_cleanup() -> None:
    with _active_lock:
        cb = _active_cleanup
    if cb is not None:
        try:
            cb()
        except Exception:  # noqa: BLE001
            pass


def _main_thread_signal_handler(signum, frame) -> None:
    _dispatch_active_cleanup()
    raise KeyboardInterrupt(f"signal {signum}")


_main_signals_installed = False


def _ensure_main_thread_signals() -> None:
    global _main_signals_installed
    if _main_signals_installed:
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _main_thread_signal_handler)
        except ValueError:
            pass
    try:
        signal.signal(signal.SIGHUP, _main_thread_signal_handler)
    except (ValueError, AttributeError):
        pass
    _main_signals_installed = True


class CleanupGuard:
    """Register a cleanup callback that runs on SIGINT/SIGTERM AND normal exit.

    The callback should be idempotent — we may call it multiple times.
    """

    _atexit_registered = False

    def __init__(self, cleanup: Callable[[], None]) -> None:
        self.cleanup = cleanup
        self._fired = False
        self._previous: dict[int, object] = {}

    def __enter__(self) -> "CleanupGuard":
        register_active_cleanup(self.cleanup)
        _ensure_main_thread_signals()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handler)
            except ValueError:
                # signal can only be set from main thread — best effort.
                pass
        try:
            self._previous[signal.SIGHUP] = signal.signal(signal.SIGHUP, self._handler)
        except (ValueError, AttributeError):
            pass
        if not CleanupGuard._atexit_registered:
            atexit.register(_dispatch_active_cleanup)
            CleanupGuard._atexit_registered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.fire()
        register_active_cleanup(None)
        for sig, prev in self._previous.items():
            try:
                signal.signal(sig, prev)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass

    def _handler(self, signum, frame) -> None:
        self.fire()
        # Re-raise to let normal SIGINT handling produce KeyboardInterrupt.
        raise KeyboardInterrupt(f"signal {signum}")

    def fire(self) -> None:
        if self._fired:
            return
        self._fired = True
        try:
            self.cleanup()
        except Exception:  # noqa: BLE001 — last-ditch cleanup, swallow
            pass
