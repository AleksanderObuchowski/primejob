"""Background poller: ssh `nvidia-smi` every 2s, parse → GpuMetric rows."""
from __future__ import annotations

import threading
import time
from typing import Callable

from primejob.backend.ssh import SshClient, SshEndpoint
from primejob.tui.state import GpuMetric


NVIDIA_SMI_CMD = (
    "nvidia-smi --query-gpu="
    "index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks_throttle_reasons.active"
    " --format=csv,noheader,nounits"
)


def parse_nvidia_smi(output: str) -> list[GpuMetric]:
    metrics: list[GpuMetric] = []
    for raw in output.strip().splitlines():
        cols = [c.strip() for c in raw.split(",")]
        if len(cols) < 6:
            continue
        try:
            index = int(cols[0])
            util = float(cols[1])
            mem_used = float(cols[2])
            mem_total = float(cols[3])
            temp = float(cols[4])
            power = float(cols[5]) if cols[5] not in {"[N/A]", "N/A"} else 0.0
        except ValueError:
            continue
        throttle = ""
        if len(cols) >= 7:
            throttle = _decode_throttle(cols[6])
        metrics.append(GpuMetric(
            index=index,
            util_pct=util,
            mem_used_mb=mem_used,
            mem_total_mb=mem_total,
            temp_c=temp,
            power_w=power,
            throttle=throttle,
        ))
    return metrics


# nvidia-smi returns throttle reasons as a hex bitmask in --format=csv when queried via
# clocks_throttle_reasons.active. Map a few common cases.
_THROTTLE_BITS = {
    0x1: "GPU idle",
    0x2: "App clk",
    0x4: "SW pwr cap",
    0x8: "HW slowdown",
    0x10: "Sync boost",
    0x20: "SW thermal",
    0x40: "HW thermal",
    0x80: "HW pwr brake",
    0x100: "Display clk",
}


def _decode_throttle(raw: str) -> str:
    raw = raw.strip()
    if not raw or raw.lower() in {"0", "0x0", "not_active"}:
        return ""
    try:
        bits = int(raw, 0)
    except ValueError:
        # Some drivers print "Not Active" / human strings — return as-is, truncated.
        return raw if raw.lower() != "not active" else ""
    reasons = [name for bit, name in _THROTTLE_BITS.items() if bits & bit]
    return ", ".join(reasons) if reasons else ""


class NvidiaWorker:
    """Threaded poller; calls on_metrics(list[GpuMetric]) and on_error(str)."""

    def __init__(
        self,
        endpoint: SshEndpoint,
        *,
        interval: float = 2.0,
        on_metrics: Callable[[list[GpuMetric]], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.interval = interval
        self.on_metrics = on_metrics
        self.on_error = on_error
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: SshClient | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="primejob-nvidia")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def _loop(self) -> None:
        # Hold a persistent SSH connection to amortize handshake cost across polls.
        try:
            self._client = SshClient(self.endpoint, retries=2, retry_delay=1.0)
            self._client.connect()
        except Exception as e:  # noqa: BLE001
            if self.on_error:
                self.on_error(f"nvidia ssh connect failed: {e}")
            return

        while not self._stop.is_set():
            try:
                result = self._client.exec(NVIDIA_SMI_CMD)
                if result.exit_code != 0:
                    # No GPU on this pod (CPU node) or driver missing — give up quietly.
                    if self.on_error:
                        self.on_error("nvidia-smi unavailable on pod")
                    return
                metrics = parse_nvidia_smi(result.stdout)
                if self.on_metrics:
                    self.on_metrics(metrics)
            except Exception as e:  # noqa: BLE001
                if self.on_error:
                    self.on_error(f"nvidia poll error: {e}")
                # Try to reconnect on next loop.
                try:
                    self._client.close()
                    self._client = SshClient(self.endpoint, retries=2, retry_delay=1.0)
                    self._client.connect()
                except Exception as conn_e:  # noqa: BLE001
                    if self.on_error:
                        self.on_error(f"nvidia reconnect failed: {conn_e}")
                    return
            if self._stop.wait(self.interval):
                return
