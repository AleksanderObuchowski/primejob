"""SSH key resolution and registration checks."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from primejob.auth import (
    _key_registered,
    _normalize_public_key,
    check_ssh_key,
    list_registered_ssh_keys,
    require_ssh_key,
)


def test_normalize_public_key_strips_comment() -> None:
    raw = "ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIComment me user@host"
    assert _normalize_public_key(raw).startswith("ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAI")


def test_key_registered_by_public_key() -> None:
    local = "ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIabc"
    registered = [{"public_key": f"{local} user@laptop"}]
    assert _key_registered(local, "SHA256:unused", registered)


def test_key_registered_by_fingerprint() -> None:
    registered = [{"fingerprint": "SHA256:abc123"}]
    assert _key_registered("ssh-ed25519 AAA", "SHA256:abc123", registered)


def test_list_registered_ssh_keys_list_response() -> None:
    client = MagicMock()
    client.request.return_value = [{"public_key": "ssh-ed25519 AAA"}]
    keys = list_registered_ssh_keys(client)
    assert len(keys) == 1


def test_list_registered_ssh_keys_wrapped_response() -> None:
    client = MagicMock()
    client.request.return_value = {"ssh_keys": [{"public_key": "ssh-ed25519 AAA"}]}
    keys = list_registered_ssh_keys(client)
    assert len(keys) == 1


def test_check_ssh_key_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing_key"
    monkeypatch.setenv("PRIME_SSH_KEY_PATH", str(missing))
    status = check_ssh_key()
    assert not status.ok
    assert "not found" in (status.error or "").lower()


def test_require_ssh_key_raises_when_unregistered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "id_ed25519"
    key_path.write_text("not-a-real-key")
    monkeypatch.setenv("PRIME_SSH_KEY_PATH", str(key_path))

    client = MagicMock()
    client.request.return_value = []

    with pytest.raises(RuntimeError, match="Could not read SSH key|not registered"):
        require_ssh_key(client)
