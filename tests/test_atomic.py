"""Tests for atomic_write_text — torn writes leave the original intact."""
from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest import mock

import pytest

from primejob._atomic import atomic_write_text


def test_atomic_write_creates_file_with_mode(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    atomic_write_text(target, '{"x": 1}', mode=0o600)
    assert target.read_text() == '{"x": 1}'
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600


def test_atomic_write_overwrites_in_place(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    target.write_text("old")
    atomic_write_text(target, "new")
    assert target.read_text() == "new"


def test_atomic_write_preserves_original_on_error(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    target.write_text("original")
    with mock.patch("primejob._atomic.os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError):
            atomic_write_text(target, "would-be-new")
    # Original untouched, no leftover .tmp file in the directory.
    assert target.read_text() == "original"
    siblings = [p.name for p in tmp_path.iterdir() if p.name != "out.json"]
    assert siblings == [], f"Stale temp files left behind: {siblings}"
