"""Confirm-terminate modal triggered by Ctrl+C.

Result semantics:
    True  → user confirmed, do graceful cleanup (download outputs + terminate pod)
    False → user cancelled, return to dashboard
    None  → force quit (second Ctrl+C); caller must warn that pod may still be running
"""
from __future__ import annotations

import time

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class TerminateModal(ModalScreen[bool | None]):
    BINDINGS = [
        ("y", "confirm", "Terminate"),
        ("Y", "confirm", "Terminate"),
        ("n", "cancel", "Cancel"),
        ("N", "cancel", "Cancel"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, *, run_id: str, pod_id: str | None, spent: float, rate: float) -> None:
        super().__init__()
        self._run_id = run_id
        self._pod_id = pod_id
        self._spent = spent
        self._rate = rate
        self._last_ctrl_c: float = 0.0

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box -danger"):
            yield Static("Terminate run?", classes="modal-title")
            yield Static(
                f"run_id:  {self._run_id}\n"
                f"pod_id:  {self._pod_id or '(not yet created)'}\n"
                f"spent:   ${self._spent:.4f}    rate: ${self._rate:.4f}/h\n"
                f"\n"
                f"This downloads outputs/ and tells Prime to delete the pod.",
                classes="modal-detail",
            )
            yield Static(
                "[y] confirm    [n / esc] cancel    [^C] force quit (leaves pod running!)",
                classes="modal-hint",
            )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_key(self, event: events.Key) -> None:
        # Force-quit semantics: ^C inside the modal = dismiss(None) (caller force exits).
        if event.key == "ctrl+c":
            self.dismiss(None)
            event.stop()
