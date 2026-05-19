"""Horizontal lifecycle stepper with pulsing active phase."""
from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from primejob.tui.state import PHASE_LABELS, PHASE_ORDER, Phase


# Reflect state: done / active / pending / failed
_GLYPHS = {
    "done": "●",
    "active": "◐",
    "pending": "○",
    "failed": "●",
}

_STYLES = {
    "done": "#00d4aa",       # success teal
    "active": "bold #f47421",  # Ubuntu orange — Prime Intellect accent
    "pending": "grey50",
    "failed": "bold #ff4444",
}


class Stepper(Static):
    DEFAULT_CSS = ""

    def __init__(self) -> None:
        super().__init__(id="stepper")
        self._current: Phase = Phase.PREFLIGHT
        self._failed: bool = False
        self._pulse: int = 0
        self._pulse_timer = None

    def on_mount(self) -> None:
        # Pulse the active glyph by toggling between ◐ and ◑ every 500ms.
        self._pulse_timer = self.set_interval(0.5, self._tick)
        self._redraw()

    def _tick(self) -> None:
        self._pulse ^= 1
        self._redraw()

    def set_phase(self, phase: Phase, *, failed: bool = False) -> None:
        self._current = phase
        self._failed = failed
        self._redraw()

    def _state_for(self, phase: Phase) -> str:
        if self._failed and phase == self._current:
            return "failed"
        try:
            cur_idx = PHASE_ORDER.index(self._current)
        except ValueError:
            # DONE/FAILED — everything done
            cur_idx = len(PHASE_ORDER)
        my_idx = PHASE_ORDER.index(phase)
        if my_idx < cur_idx:
            return "done"
        if my_idx == cur_idx:
            return "active"
        return "pending"

    def _glyph_for(self, state: str) -> str:
        if state == "active":
            return "◐" if self._pulse == 0 else "◑"
        return _GLYPHS[state]

    def _redraw(self) -> None:
        text = Text()
        for i, phase in enumerate(PHASE_ORDER):
            state = self._state_for(phase)
            glyph = self._glyph_for(state)
            text.append(glyph + " ", style=_STYLES[state])
            text.append(PHASE_LABELS[phase], style=_STYLES[state])
            if i != len(PHASE_ORDER) - 1:
                # Separator
                text.append("  ─  ", style="dim")
        self.update(text)
