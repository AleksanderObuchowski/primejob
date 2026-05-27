"""Atomic file writes for state files (lease, manifest, log).

A torn write to ~/.primejob/runs/<id>/{lease,manifest}.json leaves the
watchdog and the `runs list` UI reading half-truncated JSON. Use
`atomic_write_text` everywhere those files are persisted.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    """Write `text` to `path` via a same-directory temp file + os.replace.

    `mode` (e.g. 0o600) is applied to the temp file before the rename so the
    final file lands with restrictive permissions atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def secure_chmod(path: Path, mode: int = 0o600) -> None:
    """Best-effort chmod; silently no-op on platforms that reject it (Windows)."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass
