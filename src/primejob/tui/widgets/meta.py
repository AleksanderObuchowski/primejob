"""Single-line meta panel: elapsed · rate · spent · provider · country."""
from __future__ import annotations

import time

from rich.text import Text
from textual.widgets import Static

from primejob.runtime import fmt_elapsed
from datetime import timedelta


class MetaLine(Static):
    DEFAULT_CSS = ""

    def __init__(self) -> None:
        super().__init__(id="meta")
        self._started_at: float | None = None
        self._rate_per_hr: float = 0.0
        self._spent: float = 0.0
        self._note: str = ""
        self._timer = None
        self._frozen_elapsed_s: float | None = None  # set on finish — stops the live timer

    def on_mount(self) -> None:
        self._timer = self.set_interval(1.0, self._redraw)
        self._redraw()

    def set_cost(self, *, started_at: float, rate_per_hr: float, spent: float) -> None:
        self._started_at = started_at
        self._rate_per_hr = rate_per_hr
        self._spent = spent
        self._redraw()

    def set_note(self, note: str) -> None:
        self._note = note
        self._redraw()

    def freeze(self, *, elapsed_s: float, rate_per_hr: float, spent: float) -> None:
        """Stop the live counter and pin to the final values."""
        self._frozen_elapsed_s = elapsed_s
        self._rate_per_hr = rate_per_hr
        self._spent = spent
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:  # noqa: BLE001
                pass
            self._timer = None
        self._redraw()

    def _redraw(self) -> None:
        text = Text()
        if self._frozen_elapsed_s is not None:
            elapsed = fmt_elapsed(timedelta(seconds=self._frozen_elapsed_s))
            text.append("elapsed ", style="dim")
            text.append(elapsed, style="bold")
            text.append("  ·  ", style="dim")
        elif self._started_at is None:
            text.append("elapsed —  ", style="dim")
        else:
            elapsed = fmt_elapsed(timedelta(seconds=time.monotonic() - self._started_at))
            text.append("elapsed ", style="dim")
            text.append(elapsed, style="bold")
            text.append("  ·  ", style="dim")
        text.append("rate ", style="dim")
        text.append(f"${self._rate_per_hr:.4f}/h", style="white")
        text.append("  ·  ", style="dim")
        text.append("spent ", style="dim")
        text.append(f"${self._spent:.4f}", style="bold #f47421")
        if self._note:
            text.append("  ·  ", style="dim")
            text.append(self._note, style="dim")
        self.update(text)
