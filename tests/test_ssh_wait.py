"""SSH wait loop classification."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import paramiko
import pytest

from primejob.backend.ssh import SshEndpoint, wait_for_ssh_connect


@pytest.fixture
def endpoint() -> SshEndpoint:
    return SshEndpoint(host="pod.example", port=2222, user="root")


def test_wait_for_ssh_connect_retries_auth_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, endpoint: SshEndpoint
) -> None:
    monkeypatch.setattr("primejob.backend.ssh.time.sleep", lambda _s: None)

    mock_inst = MagicMock()
    mock_inst.connect.side_effect = [
        paramiko.AuthenticationException(),
        None,
    ]
    mock_cls = MagicMock(return_value=mock_inst)

    pod_ready = time.monotonic()

    with patch("primejob.backend.ssh.paramiko.SSHClient", mock_cls):
        out = wait_for_ssh_connect(
            endpoint,
            max_wait_s=5.0,
            retry_delay_s=0.01,
            pod_ready_monotonic=pod_ready,
            auth_warmup_s=300.0,
        )

    assert out is mock_inst
    assert mock_inst.connect.call_count == 2


def test_wait_for_ssh_connect_raises_after_auth_warm_window(
    monkeypatch: pytest.MonkeyPatch, endpoint: SshEndpoint
) -> None:
    monkeypatch.setattr("primejob.backend.ssh.time.sleep", lambda _s: None)
    monkeypatch.setattr("primejob.backend.ssh.time.monotonic", lambda: 10_000.0)

    mock_inst = MagicMock()
    mock_inst.connect.side_effect = paramiko.AuthenticationException()
    mock_cls = MagicMock(return_value=mock_inst)

    with (
        patch("primejob.backend.ssh.paramiko.SSHClient", mock_cls),
        pytest.raises(RuntimeError, match="authentication failed"),
    ):
        wait_for_ssh_connect(
            endpoint,
            max_wait_s=5.0,
            retry_delay_s=0.01,
            pod_ready_monotonic=1000.0,
            auth_warmup_s=300.0,
        )


def test_wait_for_ssh_connect_retries_transport(
    monkeypatch: pytest.MonkeyPatch, endpoint: SshEndpoint
) -> None:
    monkeypatch.setattr("primejob.backend.ssh.time.sleep", lambda _s: None)

    mock_inst = MagicMock()
    mock_inst.connect.side_effect = [
        paramiko.SSHException("banner"),
        None,
    ]
    mock_cls = MagicMock(return_value=mock_inst)

    pod_ready = time.monotonic()

    with patch("primejob.backend.ssh.paramiko.SSHClient", mock_cls):
        out = wait_for_ssh_connect(
            endpoint,
            max_wait_s=5.0,
            retry_delay_s=0.01,
            pod_ready_monotonic=pod_ready,
        )

    assert out is mock_inst
    assert mock_inst.connect.call_count == 2
