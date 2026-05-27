"""Persistent disk lifecycle."""
from __future__ import annotations

import time
from dataclasses import dataclass

from prime_cli.api.availability import AvailabilityClient, DiskAvailability
from prime_cli.api.client import APIClient
from prime_cli.api.disks import Disk, DisksClient

from primejob._retry import with_retry


READY_STATES = {"active", "available", "ready"}
PENDING_STATES = {"creating", "pending", "provisioning"}
FAILED_STATES = {"failed", "error", "terminated"}


@dataclass
class DiskLocation:
    country: str
    cloud_id: str | None
    data_center_id: str | None
    price_per_gb_hr: float | None
    provider: str | None


def find_disk(client: APIClient, name: str) -> Disk | None:
    page = with_retry(lambda: DisksClient(client).list(limit=200))
    for d in page.data:
        if d.name == name:
            return d
    return None


def pick_disk_region(
    client: APIClient,
    *,
    country: str | None = None,
) -> DiskLocation:
    """Choose the cheapest disk region, optionally constrained to a country."""
    av = AvailabilityClient(client)
    options: list[DiskAvailability] = with_retry(av.get_disks)
    candidates = []
    for o in options:
        if country and (o.country or "").upper() != country.upper():
            continue
        price = (o.spec or {}).get("price_per_unit") if isinstance(o.spec, dict) else None
        if price is None and hasattr(o.spec, "price_per_unit"):
            price = o.spec.price_per_unit
        data_center = o.data_center
        data_center_id = getattr(o, "data_center_id", None)
        if data_center_id is None:
            data_center_id = getattr(data_center, "id", None) if data_center is not None else None
        candidates.append(
            DiskLocation(
                country=o.country or "",
                cloud_id=o.cloud_id,
                data_center_id=data_center_id,
                price_per_gb_hr=price,
                provider=o.provider,
            )
        )
    if not candidates:
        raise RuntimeError(
            f"No disk regions available (country={country!r}). "
            "Try a different country or omit the filter."
        )
    candidates.sort(key=lambda c: (c.price_per_gb_hr or float("inf")))
    return candidates[0]


def ensure_disk(
    client: APIClient,
    *,
    name: str,
    size_gb: int,
    country: str | None = None,
    wait: bool = True,
) -> Disk:
    """Find existing disk by name, else create a fresh one in the cheapest matching region."""
    existing = find_disk(client, name)
    if existing is not None:
        return existing

    loc = pick_disk_region(client, country=country)
    payload = build_disk_create_payload(name=name, size_gb=size_gb, loc=loc)
    disk = with_retry(lambda: DisksClient(client).create(payload))
    if wait:
        disk = wait_for_disk_ready(client, disk.id)
    return disk


def build_disk_create_payload(*, name: str, size_gb: int, loc: DiskLocation) -> dict:
    if not loc.provider:
        raise RuntimeError(f"Disk region has no provider: {loc}")
    disk = {
        "size": size_gb,
        "name": name,
        "country": loc.country,
        "cloudId": loc.cloud_id,
        "dataCenterId": loc.data_center_id,
    }
    return {
        "disk": {k: v for k, v in disk.items() if v is not None},
        "provider": {"type": loc.provider},
    }


def wait_for_disk_ready(
    client: APIClient,
    disk_id: str,
    *,
    timeout: int = 180,
    poll_interval: float = 3.0,
) -> Disk:
    start = time.monotonic()
    last_status = None
    while True:
        d = DisksClient(client).get(disk_id)
        status = (d.status or "").lower()
        if status in READY_STATES:
            return d
        if status in FAILED_STATES:
            raise RuntimeError(f"Disk {disk_id} entered terminal state '{status}'")
        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"Disk {disk_id} still '{status}' after {timeout}s (last seen: {last_status})"
            )
        last_status = status
        time.sleep(poll_interval)


def wait_for_disk_detached(
    client: APIClient,
    disk_id: str,
    *,
    timeout: int = 180,
    poll_interval: float = 3.0,
) -> Disk:
    """Wait until a disk is no longer attached to any pod or cluster."""
    start = time.monotonic()
    while True:
        d = DisksClient(client).get(disk_id)
        if not d.pods and not d.clusters:
            return d
        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"Disk {disk_id} still attached after {timeout}s "
                f"(pods={d.pods}, clusters={d.clusters})"
            )
        time.sleep(poll_interval)


def disk_location(disk: Disk) -> tuple[str | None, str | None, str | None]:
    """Extract (country, cloud_id, data_center_id) from a Disk's info field."""
    info = disk.info
    if info is None:
        return None, None, None
    if isinstance(info, dict):
        return info.get("country"), info.get("cloud_id"), info.get("data_center_id")
    return (
        getattr(info, "country", None),
        getattr(info, "cloud_id", None),
        getattr(info, "data_center_id", None),
    )
