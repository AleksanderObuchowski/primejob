from __future__ import annotations

from primejob.config import ProjectConfig
from primejob.run import (
    RunOptions,
    _build_uv_args,
    _build_uv_install_cmd,
    _build_uv_run_cmd,
    _build_uv_sync_cmd,
)


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
