"""Auth wrapper around prime_cli APIClient.

APIClient already reads PRIME_API_KEY env var, then falls back to ~/.prime/config
(written by `prime login`). We just centralize the import + a sanity probe.
"""
from __future__ import annotations

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


def get_client() -> APIClient:
    """Return an authenticated APIClient.

    Loads .env from cwd first so local PRIME_API_KEY overrides shell env.
    """
    load_dotenv(Path.cwd() / ".env", override=False)
    return APIClient()


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
