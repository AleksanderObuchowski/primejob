"""Run option validation for dataset modes."""
from __future__ import annotations

from pathlib import Path

import pytest

from primejob.config import ProjectConfig
from primejob.run import RunOptions, _resolve_bundle_paths


def test_resolve_bundle_paths_requires_local_mode(tmp_path: Path) -> None:
    project = ProjectConfig()
    opts = RunOptions(script="train.py")
    assert _resolve_bundle_paths(tmp_path, opts, project, "none") == []


def test_resolve_bundle_paths_local_from_cli(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.json").write_text("{}")
    project = ProjectConfig()
    opts = RunOptions(script="train.py", include_data=["data"])
    paths = _resolve_bundle_paths(tmp_path, opts, project, "local")
    assert paths == [data.resolve()]


def test_resolve_bundle_paths_local_from_config(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    project = ProjectConfig(bundle_paths=["data"])
    opts = RunOptions(script="train.py")
    paths = _resolve_bundle_paths(tmp_path, opts, project, "local")
    assert paths == [data.resolve()]


def test_resolve_bundle_paths_missing_raises(tmp_path: Path) -> None:
    project = ProjectConfig()
    opts = RunOptions(script="train.py", include_data=["missing"])
    with pytest.raises(RuntimeError, match="Bundle path not found"):
        _resolve_bundle_paths(tmp_path, opts, project, "local")


def test_resolve_bundle_paths_empty_raises(tmp_path: Path) -> None:
    project = ProjectConfig()
    opts = RunOptions(script="train.py")
    with pytest.raises(RuntimeError, match="requires --include-data"):
        _resolve_bundle_paths(tmp_path, opts, project, "local")
