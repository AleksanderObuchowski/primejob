"""DataTable of per-GPU nvidia-smi metrics."""
from __future__ import annotations

from rich.text import Text
from textual.widgets import DataTable

from primejob.tui.state import GpuMetric


def _util_bar(pct: float, width: int = 8) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = round(pct / 100.0 * width)
    return "█" * filled + "░" * (width - filled)


class GpuTable(DataTable):
    DEFAULT_CSS = ""

    def __init__(self) -> None:
        super().__init__(id="gpus", zebra_stripes=False, cursor_type="none", show_header=True)
        self._populated = False

    def on_mount(self) -> None:
        self.add_columns("#", "util", "mem", "temp", "power", "state")

    def update_metrics(self, metrics: list[GpuMetric]) -> None:
        self.clear()
        for m in metrics:
            bar = _util_bar(m.util_pct)
            util_text = Text()
            util_text.append(bar, style="cyan")
            util_text.append(f" {m.util_pct:5.1f}%", style="bold")

            mem_pct = (m.mem_used_mb / m.mem_total_mb * 100.0) if m.mem_total_mb else 0
            mem_text = Text(
                f"{m.mem_used_mb/1024:5.1f}/{m.mem_total_mb/1024:5.1f} GB",
                style="bold" if mem_pct > 90 else "",
            )

            temp_style = "red" if m.temp_c >= 80 else ("yellow" if m.temp_c >= 70 else "")
            temp_text = Text(f"{m.temp_c:5.1f} °C", style=temp_style)

            power_text = Text(f"{m.power_w:5.0f} W")

            if m.throttle:
                state_text = Text(m.throttle, style="bold yellow")
            else:
                state_text = Text("—", style="dim")

            self.add_row(
                str(m.index),
                util_text,
                mem_text,
                temp_text,
                power_text,
                state_text,
            )
        self._populated = bool(metrics)

    def has_data(self) -> bool:
        return self._populated
