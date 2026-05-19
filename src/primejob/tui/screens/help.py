"""? overlay listing keybindings."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


HELP_ROWS = [
    ("q",      "quit (asks for confirm if run is still alive)"),
    ("^C",     "terminate run (graceful cleanup)"),
    ("/",      "search in log"),
    ("p",      "pause/resume log auto-scroll"),
    ("g",      "toggle GPU panel"),
    ("e",      "open log in $EDITOR"),
    ("o",      "open outputs/ folder"),
    ("s",      "copy SSH command to clipboard"),
    ("k",      "copy run_id to clipboard"),
    ("t",      "cycle theme (tokyo-night → nord → catppuccin → gruvbox → monokai)"),
    ("?",      "this help"),
]


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        ("escape", "close", "Close"),
        ("q", "close", "Close"),
        ("question_mark", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="help-box"):
            yield Static("primejob keybindings", classes="modal-title")
            for key, desc in HELP_ROWS:
                yield Static(f"  [bold cyan]{key:<6}[/bold cyan]  {desc}", classes="help-row")
            yield Static("\n[dim]esc or q to close[/dim]", classes="modal-hint")

    def action_close(self) -> None:
        self.dismiss(None)
