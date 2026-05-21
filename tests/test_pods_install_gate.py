"""Pod readiness waits for installation_progress when API reports it."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from primejob.backend import pods


def test_wait_for_running_waits_until_install_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStatus:
        def __init__(self, prog: int, ssh_connection: str = "root@host"):
            self.status = "active"
            self.ssh_connection = ssh_connection
            self.installation_progress = prog
            self.installation_failure = None

    seq = iter([FakeStatus(40), FakeStatus(99), FakeStatus(100)])

    def fake_get_status(client, pod_id: str):
        return next(seq)

    monkeypatch.setattr(pods, "get_status", fake_get_status)
    monkeypatch.setattr(pods.time, "sleep", lambda _x: None)

    out = pods.wait_for_running(MagicMock(), "pod-a", timeout=60)
    assert out.installation_progress == 100


def test_wait_for_running_without_install_field(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStatus:
        status = "active"
        ssh_connection = "root@host"
        installation_progress = None
        installation_failure = None

    monkeypatch.setattr(pods, "get_status", lambda _c, _p: FakeStatus())
    monkeypatch.setattr(pods.time, "sleep", lambda _x: None)

    out = pods.wait_for_running(MagicMock(), "pod-b", timeout=60)
    assert out.ssh_connection == "root@host"
