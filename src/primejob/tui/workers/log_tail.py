"""Tail a local log file. Used by `primejob attach` for finished or running runs.

The main `primejob run` flow already streams log lines via the EventSink and
doesn't need this. Attach is the consumer: read whatever is on disk now, then
keep polling for new bytes."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable


class LogTailer:
    def __init__(
        self,
        path: Path,
        *,
        interval: float = 0.2,
        on_line: Callable[[str, str], None] | None = None,
        initial_text_seek: int | None = None,
    ) -> None:
        self.path = path
        self.interval = interval
        self.on_line = on_line
        self._initial_text_seek = initial_text_seek
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="primejob-logtail")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _loop(self) -> None:
        offset = 0 if self._initial_text_seek is None else self._initial_text_seek
        carry = ""
        while not self._stop.is_set():
            if self.path.exists():
                try:
                    with self.path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(offset)
                        chunk = f.read()
                        offset = f.tell()
                except OSError:
                    chunk = ""
                if chunk:
                    text = carry + chunk
                    lines = text.split("\n")
                    carry = lines.pop()  # last item is partial line (no trailing \n)
                    for line in lines:
                        if self.on_line:
                            stream = "stderr" if line.startswith("[stderr] ") else "stdout"
                            payload = line[len("[stderr] "):] if stream == "stderr" else line
                            self.on_line(stream, payload)
            if self._stop.wait(self.interval):
                # flush trailing partial line
                if carry and self.on_line:
                    self.on_line("stdout", carry)
                return
