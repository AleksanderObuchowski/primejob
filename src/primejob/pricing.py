"""GPU availability + cheapest-pick logic.

Prime's API requires full GPU type names like H100_80GB. We expose user-friendly
short aliases (H100, H200, A100, etc.) and resolve them when querying.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from prime_cli.api.availability import AvailabilityClient
from prime_cli.api.client import APIClient


GPU_ALIASES: dict[str, str] = {
    "H100": "H100_80GB",
    "H200": "H200_141GB",
    "H200_80": "H200_96GB",
    "A100": "A100_80GB",
    "A40": "A40_48GB",
    "A6000": "A6000_48GB",
    "A5000": "A5000_24GB",
    "A10": "A10_24GB",
    "B200": "B200_180GB",
    "B300": "B300_262GB",
    "L40S": "L40S_48GB",
    "L40": "L40_48GB",
    "L4": "L4_24GB",
    "RTX4090": "RTX4090_24GB",
    "RTX5090": "RTX5090_32GB",
    "RTX3090": "RTX3090_24GB",
    "RTX_PRO_6000": "RTX_PRO_6000B_96GB",
    "CPU": "CPU_NODE",
}


def resolve_gpu_type(name: str) -> str:
    """Resolve an alias to the API gpu_type. Pass-through if already full name."""
    return GPU_ALIASES.get(name.upper(), name)


@dataclass
class GpuOption:
    gpu_type: str
    gpu_count: int
    country: str | None
    data_center: str | None
    cloud_id: str | None
    provider: str | None
    socket: str | None
    security: str | None
    gpu_memory: int | None
    vcpu_default: int | None
    memory_default: int | None
    disk_default: int | None
    disk_min: int | None
    disk_max: int | None
    images: list[str]
    on_demand_price: float | None
    community_price: float | None
    currency: str
    stock_status: str | None
    is_spot: bool

    @property
    def effective_price(self) -> float:
        """On-demand if available, else community/spot."""
        return self.on_demand_price or self.community_price or float("inf")

    def available(self) -> bool:
        if self.stock_status and self.stock_status.lower() in {"unavailable", "outofstock"}:
            return False
        return self.effective_price != float("inf")


def _from_availability(item) -> GpuOption:
    d = item.model_dump()
    prices = d.get("prices") or {}
    vcpu = d.get("vcpu") or {}
    mem = d.get("memory") or {}
    disk = d.get("disk") or {}
    return GpuOption(
        gpu_type=d.get("gpu_type"),
        gpu_count=d.get("gpu_count"),
        country=d.get("country"),
        data_center=d.get("data_center"),
        cloud_id=d.get("cloud_id"),
        provider=d.get("provider"),
        socket=d.get("socket"),
        security=d.get("security"),
        gpu_memory=d.get("gpu_memory"),
        vcpu_default=vcpu.get("default_count"),
        memory_default=mem.get("default_count"),
        disk_default=disk.get("default_count"),
        disk_min=disk.get("min_count"),
        disk_max=disk.get("max_count"),
        images=list(d.get("images") or []),
        on_demand_price=prices.get("on_demand"),
        community_price=prices.get("community_price"),
        currency=prices.get("currency") or "USD",
        stock_status=d.get("stock_status"),
        is_spot=bool(d.get("is_spot")),
    )


def list_gpus(
    client: APIClient,
    *,
    country: str | None = None,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    disks: list[str] | None = None,
) -> list[GpuOption]:
    av = AvailabilityClient(client)
    resolved = resolve_gpu_type(gpu_type) if gpu_type else None
    # Note: API's `regions` param wants macroregion names (united_states,
    # eu_west, etc.), not ISO country codes — we filter by ISO country
    # client-side after the response.
    raw = av.get(
        gpu_count=gpu_count,
        gpu_type=resolved,
        disks=disks,
    )
    out: list[GpuOption] = []
    for items in raw.values():
        for item in items:
            opt = _from_availability(item)
            if country and opt.country and opt.country.upper() != country.upper():
                continue
            out.append(opt)
    out.sort(key=lambda o: o.effective_price)
    return out


def pick_cheapest(
    client: APIClient,
    *,
    gpu_type: str,
    gpu_count: int = 1,
    country: str | None = None,
    disks: list[str] | None = None,
) -> GpuOption:
    options = list_gpus(
        client,
        country=country,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        disks=disks,
    )
    options = [o for o in options if o.available() and o.gpu_count == gpu_count]
    if not options:
        raise RuntimeError(
            f"No available offerings for gpu_type={gpu_type} count={gpu_count} country={country}"
        )
    return options[0]


def available_gpu_types(client: APIClient) -> list[str]:
    return AvailabilityClient(client).get_available_gpu_types()
