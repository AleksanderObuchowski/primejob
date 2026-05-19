"""Modal asking the user to confirm pod cost before spinning it up."""
from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class PreflightModal(ModalScreen[bool]):
    BINDINGS = [
        ("y", "confirm", "Yes"),
        ("Y", "confirm", "Yes"),
        ("n", "cancel", "No"),
        ("N", "cancel", "No"),
        ("escape", "cancel", "Cancel"),
        ("enter", "cancel", "No (default)"),
    ]

    def __init__(self, *, gpu: str, count: int, rate_per_hr: float, provider: str | None, country: str | None) -> None:
        super().__init__()
        self._gpu = gpu
        self._count = count
        self._rate = rate_per_hr
        self._provider = provider
        self._country = country

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box"):
            yield Static("Confirm pod cost", classes="modal-title")
            location = " / ".join(x for x in (self._provider, self._country) if x) or "any"
            yield Static(
                f"GPU:      {self._gpu} ×{self._count}\n"
                f"Provider: {location}\n"
                f"Rate:     ${self._rate:.4f}/h",
                classes="modal-detail",
            )
            yield Static("[y] confirm and spawn pod    [n / esc] abort", classes="modal-hint")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
