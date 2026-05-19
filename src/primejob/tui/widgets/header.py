"""Top banner showing primejob + script + GPU badge."""
from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from primejob.tui.state import RunMeta, gpu_badge, script_label


class Header(Static):
    DEFAULT_CSS = ""

    def __init__(self, meta: RunMeta, *, attach_mode: bool = False) -> None:
        super().__init__(id="header")
        self._meta = meta
        self._attach_mode = attach_mode

    def on_mount(self) -> None:
        self.refresh_meta(self._meta)

    def refresh_meta(self, meta: RunMeta) -> None:
        self._meta = meta
        text = Text()
        text.append("primejob", style="bold cyan")
        if self._attach_mode:
            text.append("  attach", style="bold yellow")
        text.append("  •  ")
        text.append(script_label(meta), style="bold")
        badge = gpu_badge(meta)
        if badge:
            text.append("  •  ")
            text.append(badge, style="dim")
        if meta.run_id:
            text.append("  •  ")
            text.append(meta.run_id, style="dim italic")
        self.update(text)
