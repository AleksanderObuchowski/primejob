"""Auth wrapper around prime_cli APIClient.

APIClient already reads PRIME_API_KEY env var, then falls back to ~/.prime/config
(written by `prime login`). We just centralize the import + a sanity probe.
"""
from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from prime_cli.api.client import APIClient


@dataclass
class AuthStatus:
    has_env_key: bool
    has_config_file: bool
    config_path: Path
    client_ok: bool
    error: str | None = None


@dataclass
class SshKeyStatus:
    key_path: Path | None
    key_exists: bool
    key_readable: bool
    fingerprint: str | None
    registered: bool | None  # None when the API check could not run
    error: str | None = None

    @property
    def ok(self) -> bool:
        if not self.key_exists:
            return False
        if not self.key_readable:
            return False
        if self.registered is False:
            return False
        return True


def get_client() -> APIClient:
    """Return an authenticated APIClient.

    Loads .env from cwd first so local PRIME_API_KEY overrides shell env.
    """
    load_dotenv(Path.cwd() / ".env", override=False)
    return APIClient()


def resolve_ssh_key_path() -> Path | None:
    """Return the SSH private key path from prime CLI config or env."""
    env_val = os.environ.get("PRIME_SSH_KEY_PATH")
    if env_val:
        path = Path(env_val).expanduser()
        return path if path.exists() else None

    try:
        from prime_cli.core.config import Config

        path = Path(Config().ssh_key_path).expanduser()
        return path if path.exists() else None
    except Exception:  # noqa: BLE001
        default = Path.home() / ".ssh" / "id_rsa"
        return default if default.exists() else None


def check_auth() -> AuthStatus:
    """Probe auth without raising — for the `doctor` command."""
    load_dotenv(Path.cwd() / ".env", override=False)
    config_path = Path.home() / ".prime" / "config.json"
    has_env_key = bool(os.environ.get("PRIME_API_KEY"))
    has_config_file = config_path.exists()
    try:
        client = APIClient()
        from prime_cli.api.wallet import WalletClient
        WalletClient(client).get(limit=1)
        return AuthStatus(has_env_key, has_config_file, config_path, client_ok=True)
    except Exception as e:  # noqa: BLE001
        return AuthStatus(
            has_env_key, has_config_file, config_path, client_ok=False, error=str(e)
        )


def _load_private_key(path: Path):
    import paramiko

    loaders = (
        paramiko.RSAKey.from_private_key_file,
        paramiko.Ed25519Key.from_private_key_file,
        paramiko.ECDSAKey.from_private_key_file,
    )
    last_exc: Exception | None = None
    for loader in loaders:
        try:
            return loader(str(path))
        except paramiko.SSHException as e:
            last_exc = e
    raise ValueError(f"Could not load SSH private key at {path}: {last_exc}")


def _fingerprint_sha256(pkey) -> str:
    digest = hashlib.sha256(pkey.asbytes()).digest()
    b64 = base64.b64encode(digest).decode().rstrip("=")
    return f"SHA256:{b64}"


def _normalize_public_key(text: str) -> str:
    parts = text.strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return text.strip()


def _local_public_key_text(path: Path) -> tuple[str, str]:
    pub_path = path.with_suffix(path.suffix + ".pub")
    if pub_path.exists():
        pub_text = pub_path.read_text().strip()
        return pub_text, _normalize_public_key(pub_text)

    pkey = _load_private_key(path)
    pub_text = f"{pkey.get_name()} {pkey.get_base64()}"
    return pub_text, _normalize_public_key(pub_text)


def list_registered_ssh_keys(client: APIClient) -> list[dict]:
    """Return SSH public keys registered in the Prime account."""
    for endpoint in ("/ssh_keys/", "/ssh_keys"):
        try:
            resp = client.request("GET", endpoint)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(resp, list):
            return [item for item in resp if isinstance(item, dict)]
        if isinstance(resp, dict):
            for key in ("ssh_keys", "items", "data", "results"):
                items = resp.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
    return []


def _key_registered(normalized_local: str, fingerprint: str, registered: list[dict]) -> bool:
    for item in registered:
        remote_fp = item.get("fingerprint") or item.get("sha256_fingerprint")
        if isinstance(remote_fp, str) and remote_fp.strip() == fingerprint:
            return True
        for field in ("public_key", "key", "ssh_public_key"):
            value = item.get(field)
            if isinstance(value, str) and _normalize_public_key(value) == normalized_local:
                return True
    return False


def check_ssh_key(client: APIClient | None = None) -> SshKeyStatus:
    """Validate local SSH key setup and Prime account registration."""
    configured = None
    try:
        from prime_cli.core.config import Config

        configured = Path(Config().ssh_key_path).expanduser()
    except Exception:  # noqa: BLE001
        pass

    key_path = resolve_ssh_key_path()
    if key_path is None:
        hint = f" (configured: {configured})" if configured else ""
        return SshKeyStatus(
            key_path=configured,
            key_exists=False,
            key_readable=False,
            fingerprint=None,
            registered=None,
            error=f"SSH private key not found{hint}. Run `prime config set-ssh-key-path PATH`.",
        )

    try:
        normalized, fingerprint = _local_public_key_text(key_path)
    except Exception as e:  # noqa: BLE001
        return SshKeyStatus(
            key_path=key_path,
            key_exists=True,
            key_readable=False,
            fingerprint=None,
            registered=None,
            error=f"Could not read SSH key at {key_path}: {e}",
        )

    registered: bool | None = None
    api_error: str | None = None
    if client is not None:
        try:
            keys = list_registered_ssh_keys(client)
            registered = _key_registered(normalized, fingerprint, keys)
            if not registered:
                api_error = (
                    f"Public key {fingerprint} is not registered in your Prime account. "
                    "Add it at https://app.primeintellect.ai/settings/ssh-keys "
                    "or run `prime ssh-keys add`."
                )
        except Exception as e:  # noqa: BLE001
            registered = None
            api_error = f"Could not verify SSH key registration: {e}"

    return SshKeyStatus(
        key_path=key_path,
        key_exists=True,
        key_readable=True,
        fingerprint=fingerprint,
        registered=registered,
        error=api_error,
    )


def require_ssh_key(client: APIClient) -> None:
    """Fail fast before provisioning if SSH auth will not work."""
    status = check_ssh_key(client)
    if not status.ok:
        raise RuntimeError(status.error or "SSH key check failed")
