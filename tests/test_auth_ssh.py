"""SSH key resolution and registration checks."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from primejob.auth import (
    discover_ssh_keys,
    register_ssh_key,
    set_ssh_key_primary,
    _key_registered,
    _normalize_public_key,
    check_ssh_key,
    list_registered_ssh_keys,
    record_is_primary,
    require_ssh_key,
    ssh_auth_failure_hint,
)


def test_normalize_public_key_strips_comment() -> None:
    raw = "ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIComment me user@host"
    assert _normalize_public_key(raw).startswith("ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAI")


def test_key_registered_by_public_key_camel_case() -> None:
    local = "ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIabc"
    registered = [{"publicKey": f"{local} user@laptop"}]
    assert _key_registered(local, "SHA256:unused", registered)


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


def test_discover_ssh_keys_order(tmp_path: Path) -> None:
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "id_rsa").write_text("dummy")
    (ssh / "id_ed25519").write_text("dummy")
    keys = discover_ssh_keys(home=tmp_path)
    assert keys[0].name == "id_ed25519"
    assert keys[1].name == "id_rsa"


def test_register_ssh_key_posts_expected_payload() -> None:
    client = MagicMock()
    client.request.return_value = {"id": "k1"}
    register_ssh_key(client, name="laptop", public_key_line="ssh-ed25519 AAA x")
    client.request.assert_called_once_with(
        "POST",
        "/ssh_keys/",
        json={"name": "laptop", "publicKey": "ssh-ed25519 AAA x"},
    )


def test_set_ssh_key_primary_patch_payload() -> None:
    client = MagicMock()
    client.request.return_value = {"id": "k1", "isPrimary": True}
    set_ssh_key_primary(client, "k1")
    client.request.assert_called_once_with(
        "PATCH",
        "/ssh_keys/k1",
        json={"isPrimary": True},
    )


def test_record_is_primary_camel_and_snake() -> None:
    assert record_is_primary({"isPrimary": True})
    assert not record_is_primary({"isPrimary": False})
    assert record_is_primary({"is_primary": True})
    assert not record_is_primary({})


def test_check_ssh_key_primary_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import paramiko

    key_path = tmp_path / "id_rsa"
    pkey = paramiko.RSAKey.generate(2048)
    pkey.write_private_key_file(str(key_path))
    pub_line = f"{pkey.get_name()} {pkey.get_base64()} test@host"
    key_path.with_suffix(".pub").write_text(pub_line)

    monkeypatch.setenv("PRIME_SSH_KEY_PATH", str(key_path))

    client = MagicMock()
    client.request.return_value = [
        {
            "id": "k1",
            "fingerprint": None,
            "publicKey": pub_line,
            "isPrimary": True,
        }
    ]
    status = check_ssh_key(client)
    assert status.registered is True
    assert status.is_primary is True
    assert status.matched_key_id == "k1"


def test_ssh_auth_failure_hint_registered_primary() -> None:
    from primejob.auth import SshKeyStatus

    st = SshKeyStatus(
        key_path=Path("/tmp/k"),
        key_exists=True,
        key_readable=True,
        fingerprint="SHA256:abc",
        registered=True,
        is_primary=True,
    )
    hint = ssh_auth_failure_hint(st, provider="massedcompute")
    assert "massedcompute" in hint
    assert "exclude_providers" in hint


def test_ssh_auth_failure_hint_not_primary() -> None:
    from primejob.auth import SshKeyStatus

    st = SshKeyStatus(
        key_path=Path("/tmp/k"),
        key_exists=True,
        key_readable=True,
        fingerprint="SHA256:abc",
        registered=True,
        is_primary=False,
    )
    hint = ssh_auth_failure_hint(st)
    assert "primary" in hint.lower()
