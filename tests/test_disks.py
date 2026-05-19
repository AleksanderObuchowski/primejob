"""Persistent disk payload logic."""
from __future__ import annotations

import pytest

from primejob.backend.disks import DiskLocation, build_disk_create_payload


def test_build_disk_create_payload_matches_prime_cli_shape() -> None:
    payload = build_disk_create_payload(
        name="pj-audit",
        size_gb=50,
        loc=DiskLocation(
            country="FI",
            cloud_id="cloud-1",
            data_center_id="FIN-01",
            price_per_gb_hr=0.0001,
            provider="datacrunch",
        ),
    )

    assert payload == {
        "disk": {
            "size": 50,
            "name": "pj-audit",
            "country": "FI",
            "cloudId": "cloud-1",
            "dataCenterId": "FIN-01",
        },
        "provider": {"type": "datacrunch"},
    }


def test_build_disk_create_payload_omits_none_location_fields() -> None:
    payload = build_disk_create_payload(
        name="pj-audit",
        size_gb=50,
        loc=DiskLocation(
            country="US",
            cloud_id=None,
            data_center_id="US-CA-2",
            price_per_gb_hr=0.0001,
            provider="runpod",
        ),
    )

    assert payload["disk"] == {
        "size": 50,
        "name": "pj-audit",
        "country": "US",
        "dataCenterId": "US-CA-2",
    }


def test_build_disk_create_payload_requires_provider() -> None:
    with pytest.raises(RuntimeError, match="no provider"):
        build_disk_create_payload(
            name="pj-audit",
            size_gb=50,
            loc=DiskLocation(
                country="FI",
                cloud_id="cloud-1",
                data_center_id="FIN-01",
                price_per_gb_hr=0.0001,
                provider=None,
            ),
        )
