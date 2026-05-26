"""Run option validation for dataset modes."""
from __future__ import annotations

from pathlib import Path

import pytest

from primejob.config import ProjectConfig
from primejob.run import (
    RunOptions,
    _build_uv_args,
    _build_uv_install_cmd,
    _build_uv_run_cmd,
    _build_uv_sync_cmd,
    _resolve_bundle_paths,
)


def test_resolve_bundle_paths_requires_local_mode(tmp_path: Path) -> None:
    project = ProjectConfig()
    opts = RunOptions(script="train.py")
    assert _resolve_bundle_paths(tmp_path, opts, project, "none") == []


def test_resolve_bundle_paths_local_from_cli(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.json").write_text("{}")
    project = ProjectConfig()
    opts = RunOptions(script="train.py", include=["data"])
    paths = _resolve_bundle_paths(tmp_path, opts, project, "local")
    assert paths == [data.resolve()]


def test_resolve_bundle_paths_local_from_config_include(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    project = ProjectConfig(include=["data"])
    opts = RunOptions(script="train.py")
    paths = _resolve_bundle_paths(tmp_path, opts, project, "local")
    assert paths == [data.resolve()]


def test_resolve_bundle_paths_local_from_legacy_bundle_paths(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    project = ProjectConfig(bundle_paths=["data"])
    opts = RunOptions(script="train.py")
    paths = _resolve_bundle_paths(tmp_path, opts, project, "local")
    assert paths == [data.resolve()]


def test_resolve_bundle_paths_combines_config_and_cli_in_order(tmp_path: Path) -> None:
    for name in ("cfg", "legacy", "cli", "deprecated"):
        (tmp_path / name).mkdir()
    project = ProjectConfig(include=["cfg"], bundle_paths=["legacy", "cfg"])
    opts = RunOptions(script="train.py", include=["cli"], include_data=["deprecated", "cli"])
    paths = _resolve_bundle_paths(tmp_path, opts, project, "local")
    assert paths == [
        (tmp_path / "cfg").resolve(),
        (tmp_path / "legacy").resolve(),
        (tmp_path / "cli").resolve(),
        (tmp_path / "deprecated").resolve(),
    ]


def test_resolve_bundle_paths_missing_raises(tmp_path: Path) -> None:
    project = ProjectConfig()
    opts = RunOptions(script="train.py", include_data=["missing"])
    with pytest.raises(RuntimeError, match="Bundle path not found"):
        _resolve_bundle_paths(tmp_path, opts, project, "local")


def test_resolve_bundle_paths_empty_raises(tmp_path: Path) -> None:
    project = ProjectConfig()
    opts = RunOptions(script="train.py")
    with pytest.raises(RuntimeError, match="requires --include"):
        _resolve_bundle_paths(tmp_path, opts, project, "local")


def test_build_uv_install_cmd_uses_primejob_owned_config_dir() -> None:
    cmd = _build_uv_install_cmd()
    assert "mkdir -p /tmp/primejob/bin /tmp/primejob/config" in cmd
    assert "UV_INSTALL_DIR=/tmp/primejob/bin" in cmd
    assert "XDG_CONFIG_HOME=/tmp/primejob/config" in cmd
    assert "INSTALLER_NO_MODIFY_PATH=1" in cmd


def test_build_uv_commands_put_flags_before_python_target() -> None:
    project = ProjectConfig(uv_extras=["training"], uv_groups=["train"])
    opts = RunOptions(
        script="train.py",
        args=["--epochs", "1"],
        uv_extras=["metrics"],
        uv_groups=["train"],
        uv_all_extras=True,
    )
    uv_args = _build_uv_args(project, opts)

    assert uv_args == [
        "--all-extras",
        "--extra",
        "training",
        "--extra",
        "metrics",
        "--group",
        "train",
    ]
    assert _build_uv_sync_cmd(uv_args).endswith(
        "uv sync --all-extras --extra training --extra metrics --group train"
    )
    assert _build_uv_run_cmd(opts, uv_args).endswith(
        "uv run --all-extras --extra training --extra metrics --group train "
        "python -u train.py --epochs 1"
    )
