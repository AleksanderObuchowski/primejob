"""Config loader reads [tool.primejob] from pyproject.toml."""
from __future__ import annotations

from pathlib import Path

from primejob.config import (
    ProjectConfig,
    effective_gpu_count,
    find_pyproject,
    load_project_config,
)


def test_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_project_config(tmp_path)
    assert cfg.dataset_disk is None
    assert cfg.default_gpu == "H200"
    assert cfg.default_count == 1
    assert cfg.forward_env == []
    assert cfg.include == []
    assert cfg.ssh_max_wait == 300
    assert cfg.ssh_retry_delay == 5.0
    assert cfg.ssh_auth_timeout == 90.0
    assert cfg.exclude_providers == []
    assert cfg.uv_extras == []
    assert cfg.uv_groups == []
    assert cfg.uv_all_extras is False
    assert cfg.download_outputs is True
    assert cfg.download_include == []
    assert cfg.download_exclude == []


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
include = ["data"]
bundle_paths = ["legacy-data"]
ssh_max_wait = 120
ssh_retry_delay = 3
ssh_auth_timeout = 45
exclude_providers = ["massedcompute", "nebius"]
uv_extras = ["training"]
uv_groups = ["train"]
uv_all_extras = true
download_outputs = false
download_include = ["outputs/**/best/**"]
download_exclude = ["outputs/**/checkpoint-*/*.pt"]
"""
    )
    cfg = load_project_config(tmp_path)
    assert cfg.dataset_disk == "my-data"
    assert cfg.forward_env == ["HF_TOKEN", "WANDB_API_KEY"]
    assert cfg.default_gpu == "H100"
    assert cfg.default_country == "US"
    assert cfg.default_count == 2
    assert cfg.default_disk_size == 100
    assert cfg.include == ["data", "legacy-data"]
    assert cfg.bundle_paths_deprecated is True
    assert cfg.ssh_max_wait == 120
    assert cfg.ssh_retry_delay == 3.0
    assert cfg.ssh_auth_timeout == 45.0
    assert cfg.exclude_providers == ["massedcompute", "nebius"]
    assert cfg.uv_extras == ["training"]
    assert cfg.uv_groups == ["train"]
    assert cfg.uv_all_extras is True
    assert cfg.download_outputs is False
    assert cfg.download_include == ["outputs/**/best/**"]
    assert cfg.download_exclude == ["outputs/**/checkpoint-*/*.pt"]


def test_effective_gpu_count_cli_overrides_config() -> None:
    cfg = ProjectConfig(default_count=3)
    assert effective_gpu_count(4, cfg) == 4
    assert effective_gpu_count(None, cfg) == 3


def test_walks_up_to_find_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    nested = tmp_path / "deep" / "nested"
    nested.mkdir(parents=True)
    found = find_pyproject(nested)
    assert found == tmp_path / "pyproject.toml"


def test_include_field_canonical(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo"

[tool.primejob]
include = ["data/pii/*.jsonl", "configs/"]
"""
    )
    cfg = load_project_config(tmp_path)
    assert cfg.include == ["data/pii/*.jsonl", "configs/"]
    assert cfg.bundle_paths_deprecated is False


def test_bundle_paths_alias_merges_into_include(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo"

[tool.primejob]
bundle_paths = ["data/pii/foo.jsonl", "data/pii/bar.jsonl"]
"""
    )
    cfg = load_project_config(tmp_path)
    assert cfg.include == ["data/pii/foo.jsonl", "data/pii/bar.jsonl"]
    assert cfg.bundle_paths_deprecated is True


def test_include_and_bundle_paths_merge_dedupe(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo"

[tool.primejob]
include = ["new/file.jsonl"]
bundle_paths = ["legacy/file.jsonl", "new/file.jsonl"]
"""
    )
    cfg = load_project_config(tmp_path)
    assert cfg.include == ["new/file.jsonl", "legacy/file.jsonl"]
    assert cfg.bundle_paths_deprecated is True
