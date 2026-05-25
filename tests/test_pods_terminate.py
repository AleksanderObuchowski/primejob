"""Tests for terminate_pod result reporting."""
from __future__ import annotations

from unittest.mock import MagicMock

from primejob.backend.pods import TerminateResult, terminate_pod


def test_terminate_pod_success() -> None:
    client = MagicMock()
    pods_client = MagicMock()
    client.pods = pods_client

    import primejob.backend.pods as pods_mod

    original = pods_mod.PodsClient

    class FakePodsClient:
        def __init__(self, _client):
            pass

        def delete(self, pod_id: str) -> None:
            assert pod_id == "abc"

    pods_mod.PodsClient = FakePodsClient
    try:
        result = terminate_pod(client, "abc")
        assert result.success is True
        assert result.error is None
    finally:
        pods_mod.PodsClient = original


def test_terminate_pod_failure() -> None:
    client = MagicMock()

    import primejob.backend.pods as pods_mod

    original = pods_mod.PodsClient

    class FakePodsClient:
        def __init__(self, _client):
            pass

        def delete(self, _pod_id: str) -> None:
            raise RuntimeError("api down")

    pods_mod.PodsClient = FakePodsClient
    try:
        result = terminate_pod(client, "abc")
        assert result.success is False
        assert "api down" in (result.error or "")
    finally:
        pods_mod.PodsClient = original
