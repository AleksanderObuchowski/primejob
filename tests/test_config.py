"""Config loader reads [tool.primejob] from pyproject.toml."""
from __future__ import annotations

from pathlib import Path

from primejob.config import find_pyproject, load_project_config


def test_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_project_config(tmp_path)
    assert cfg.dataset_disk is None
    assert cfg.default_gpu == "H200"
    assert cfg.default_count == 1
    assert cfg.forward_env == []
    assert cfg.ssh_max_wait == 300
    assert cfg.ssh_retry_delay == 5.0
    assert cfg.exclude_providers == []


def test_reads_tool_primejob(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo"

[tool.primejob]
dataset_disk = "my-data"
forward_env = ["HF_TOKEN", "WANDB_API_KEY"]
default_gpu = "H100"
default_country = "US"
default_count = 2
default_disk_size = 100
ssh_max_wait = 120
ssh_retry_delay = 3
exclude_providers = ["massedcompute", "nebius"]
"""
    )
    cfg = load_project_config(tmp_path)
    assert cfg.dataset_disk == "my-data"
    assert cfg.forward_env == ["HF_TOKEN", "WANDB_API_KEY"]
    assert cfg.default_gpu == "H100"
    assert cfg.default_country == "US"
    assert cfg.default_count == 2
    assert cfg.default_disk_size == 100
    assert cfg.ssh_max_wait == 120
    assert cfg.ssh_retry_delay == 3.0
    assert cfg.exclude_providers == ["massedcompute", "nebius"]


def test_walks_up_to_find_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    nested = tmp_path / "deep" / "nested"
    nested.mkdir(parents=True)
    found = find_pyproject(nested)
    assert found == tmp_path / "pyproject.toml"
