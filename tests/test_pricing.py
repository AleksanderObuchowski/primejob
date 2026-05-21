"""Pricing logic on synthetic fixtures (no network)."""
from __future__ import annotations

import pytest

from primejob.pricing import (
    GPU_ALIASES,
    GpuOption,
    _provider_excluded,
    normalize_provider_name,
    resolve_gpu_type,
)


def _opt(price=1.0, gpu="H100_80GB", country="US", count=1, stock="Available", community=None):
    return GpuOption(
        gpu_type=gpu,
        gpu_count=count,
        country=country,
        data_center="dc1",
        cloud_id="cloud",
        provider="runpod",
        socket="PCIe",
        security="secure_cloud",
        gpu_memory=80,
        vcpu_default=16,
        memory_default=128,
        disk_default=120,
        disk_min=80,
        disk_max=3000,
        images=["ubuntu_22_cuda_12"],
        on_demand_price=price,
        community_price=community,
        currency="USD",
        stock_status=stock,
        is_spot=False,
    )


def test_resolve_aliases() -> None:
    assert resolve_gpu_type("H100") == "H100_80GB"
    assert resolve_gpu_type("h200") == "H200_141GB"
    assert resolve_gpu_type("H100_80GB") == "H100_80GB"
    assert resolve_gpu_type("UNKNOWN_CARD") == "UNKNOWN_CARD"


def test_effective_price_prefers_on_demand() -> None:
    assert _opt(price=2.0, community=0.5).effective_price == 2.0
    assert _opt(price=None, community=0.5).effective_price == 0.5
    assert _opt(price=None, community=None).effective_price == float("inf")


def test_available_filters_oos() -> None:
    assert _opt(stock="Available").available()
    assert _opt(stock="Low").available()  # not unavailable
    assert not _opt(stock="OutOfStock").available()
    assert not _opt(price=None, community=None).available()


def test_alias_table_complete() -> None:
    # the aliases we advertise to users should resolve to API-acceptable formats
    for short, full in GPU_ALIASES.items():
        assert "_" in full or full == "CPU_NODE", f"{short} → {full} looks wrong"


def test_normalize_provider_name() -> None:
    assert normalize_provider_name("MassedCompute") == "massedcompute"
    assert normalize_provider_name("nebius") == "nebius"
    assert normalize_provider_name("crusoe-cloud") == "crusoecloud"


def test_provider_excluded_case_insensitive() -> None:
    exclude = {normalize_provider_name("MassedCompute")}
    assert _provider_excluded("massedcompute", exclude)
    assert _provider_excluded("MassedCompute", exclude)
    assert not _provider_excluded("nebius", exclude)
