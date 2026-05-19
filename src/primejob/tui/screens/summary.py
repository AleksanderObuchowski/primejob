"""Final summary screen: shown after the run finishes (success or failure)."""
from __future__ import annotations

from datetime import timedelta

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from primejob.runtime import fmt_elapsed
from primejob.tui.state import FinalSummary


class SummaryScreen(ModalScreen[None]):
    BINDINGS = [
        ("q", "close", "Close"),
        ("escape", "close", "Close"),
        ("enter", "close", "Close"),
    ]

    def __init__(self, summary: FinalSummary) -> None:
        super().__init__(id="summary")
        self._summary = summary

    def compose(self) -> ComposeResult:
        s = self._summary
        ok = s.status == "finished" and (s.exit_code == 0)
        title_cls = "-ok" if ok else "-fail"
        card_cls = "" if ok else "-failed"
        with Vertical(classes=f"summary-card {card_cls}"):
            title = "✓ Run finished" if ok else (
                "⚠ Run terminated" if s.status == "terminated" else "✗ Run failed"
            )
            yield Static(title, classes=f"summary-title {title_cls}")

            elapsed = fmt_elapsed(timedelta(seconds=s.elapsed_s))
            exit_str = "—" if s.exit_code is None else str(s.exit_code)
            outputs = s.outputs_path or "—"
            yield Static(
                Text.from_markup(
                    f"[dim]status[/dim]    {s.status}\n"
                    f"[dim]exit[/dim]      {exit_str}\n"
                    f"[dim]elapsed[/dim]   {elapsed}\n"
                    f"[dim]cost[/dim]      ${s.total_cost:.4f}\n"
                    f"[dim]outputs[/dim]   {outputs}"
                ),
                classes="summary-row",
            )

            if s.last_error:
                yield Static("Last error", classes="summary-error")
                snippet = Text("\n".join(s.last_error[-10:]))
                yield Static(snippet, classes="summary-error-line")

            yield Static("\nPress [bold]q[/bold] or [bold]enter[/bold] to close.", classes="modal-hint")

    def action_close(self) -> None:
        self.dismiss(None)
