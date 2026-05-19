"""Streaming log viewer: RichLog with autoscroll-pause, search, error highlight."""
from __future__ import annotations

import re

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, RichLog, Static


# Patterns we highlight as errors / warnings.
_ERROR_RE = re.compile(r"\b(Error|Exception|Traceback|FAILED|OOM|CUDA out of memory)\b", re.IGNORECASE)
_WARN_RE = re.compile(r"\b(Warning|WARN|DeprecationWarning)\b", re.IGNORECASE)


def style_log_line(stream: str, line: str) -> Text:
    """Apply error/warning/stderr styles to a raw log line."""
    base = Text()
    if stream == "stderr":
        base.append("[stderr] ", style="dim red")
    text = Text(line)
    if _ERROR_RE.search(line):
        text.stylize("bold red")
    elif _WARN_RE.search(line):
        text.stylize("yellow")
    elif stream == "stderr":
        text.stylize("red")
    base.append(text)
    return base


class LogView(Vertical):
    """Vertical layout: status row (top, hidden by default), log (1fr), search input (bottom, hidden by default)."""

    DEFAULT_CSS = """
    LogView {
        height: 1fr;
    }
    LogView > Static.status-row {
        height: 1;
        background: $panel;
        color: $text-muted;
    }
    LogView > Static.status-row.-hidden {
        display: none;
    }
    LogView > RichLog {
        height: 1fr;
    }
    LogView > Input.search-input {
        height: 3;
        background: $panel;
        border: round $accent;
    }
    LogView > Input.search-input.-hidden {
        display: none;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="log")
        self._log: RichLog | None = None
        self._search_input: Input | None = None
        self._status: Static | None = None
        self._paused = False
        self._search_term: str | None = None
        self._buffer: list[tuple[str, str]] = []  # (stream, line) — replayed on re-render

    def compose(self) -> ComposeResult:
        self._status = Static("", classes="status-row -hidden")
        yield self._status
        self._log = RichLog(highlight=False, markup=False, wrap=False, auto_scroll=True)
        yield self._log
        self._search_input = Input(placeholder="search…  enter=jump  esc=close", classes="search-input -hidden")
        yield self._search_input

    def append(self, stream: str, line: str) -> None:
        self._buffer.append((stream, line))
        if len(self._buffer) > 5000:
            # Cap the buffer; primejob writes log to a file too, full record there.
            self._buffer = self._buffer[-5000:]
        rendered = style_log_line(stream, line)
        if self._search_term and self._search_term.lower() in line.lower():
            rendered.stylize("on yellow")
        if self._log is not None:
            self._log.write(rendered, scroll_end=not self._paused)

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._log is not None:
            self._log.auto_scroll = not self._paused
        self._update_status()

    def open_search(self) -> None:
        if self._search_input is None:
            return
        self._search_input.remove_class("-hidden")
        self._search_input.focus()

    def close_search(self) -> None:
        if self._search_input is None:
            return
        self._search_input.add_class("-hidden")
        self._search_input.value = ""
        self._search_term = None
        self._update_status()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is not self._search_input:
            return
        self._search_term = event.value.strip() or None
        self._update_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is not self._search_input:
            return
        # On enter, repaint log buffer with the search highlight applied.
        if self._search_term:
            self._repaint_with_highlight(self._search_term)

    def on_key(self, event: events.Key) -> None:
        # Esc while search has focus → close it.
        if event.key == "escape" and self._search_input and self._search_input.has_focus:
            self.close_search()
            event.stop()

    def _repaint_with_highlight(self, term: str) -> None:
        if self._log is None:
            return
        self._log.clear()
        for stream, line in self._buffer:
            rendered = style_log_line(stream, line)
            if term.lower() in line.lower():
                rendered.stylize("on yellow")
            self._log.write(rendered, scroll_end=False)
        self._log.scroll_end()

    def _update_status(self) -> None:
        if self._status is None:
            return
        bits = []
        if self._paused:
            bits.append("⏸ paused")
        if self._search_term:
            bits.append(f"/ {self._search_term!r}")
        if bits:
            self._status.update("  ".join(bits))
            self._status.remove_class("-hidden")
        else:
            self._status.add_class("-hidden")
