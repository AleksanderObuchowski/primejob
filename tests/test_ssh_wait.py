"""SSH wait loop classification."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import paramiko
import pytest

from primejob.backend.ssh import (
    SshAuthPropagationTimeout,
    SshEndpoint,
    _download_path_selected,
    wait_for_ssh_connect,
)


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
        pytest.raises(RuntimeError, match="exclude_providers|skip-provider"),
    ):
        wait_for_ssh_connect(
            endpoint,
            max_wait_s=5.0,
            retry_delay_s=0.01,
            pod_ready_monotonic=1000.0,
            auth_warmup_s=300.0,
            auth_failure_hint=(
                "Try `--skip-provider`, set `[tool.primejob].exclude_providers`, "
                "or pick a different `--country`."
            ),
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


def test_wait_for_ssh_connect_auth_timeout_is_distinct(
    monkeypatch: pytest.MonkeyPatch, endpoint: SshEndpoint
) -> None:
    monkeypatch.setattr("primejob.backend.ssh.time.sleep", lambda _s: None)
    monkeypatch.setattr("primejob.backend.ssh.time.monotonic", lambda: 200.0)

    mock_inst = MagicMock()
    mock_inst.connect.side_effect = paramiko.AuthenticationException()
    mock_cls = MagicMock(return_value=mock_inst)

    with (
        patch("primejob.backend.ssh.paramiko.SSHClient", mock_cls),
        pytest.raises(SshAuthPropagationTimeout),
    ):
        wait_for_ssh_connect(
            endpoint,
            max_wait_s=300.0,
            retry_delay_s=0.01,
            pod_ready_monotonic=100.0,
            auth_warmup_s=300.0,
            auth_timeout_s=90.0,
        )


def test_download_path_selected_defaults_to_all() -> None:
    assert _download_path_selected("outputs/checkpoint/model.pt", [], [])


def test_download_path_selected_include_only() -> None:
    assert _download_path_selected(
        "outputs/run/best/model.safetensors",
        ["outputs/**/best/**"],
        [],
    )
    assert not _download_path_selected(
        "outputs/run/checkpoint-1/model.safetensors",
        ["outputs/**/best/**"],
        [],
    )


def test_download_path_selected_exclude_only() -> None:
    assert not _download_path_selected(
        "outputs/run/checkpoint-1/optimizer.pt",
        [],
        ["outputs/**/checkpoint-*/*.pt"],
    )
    assert _download_path_selected("outputs/run/metrics.json", [], ["outputs/**/*.pt"])


def test_download_path_selected_include_then_exclude() -> None:
    assert not _download_path_selected(
        "outputs/run/checkpoint-1/model.safetensors",
        ["outputs/**/*.safetensors"],
        ["outputs/**/checkpoint-*/*"],
    )
    assert _download_path_selected(
        "outputs/run/best/model.safetensors",
        ["outputs/**/*.safetensors"],
        ["outputs/**/checkpoint-*/*"],
    )
