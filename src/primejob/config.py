"""Project-level config from [tool.primejob] in pyproject.toml.

`include` is the canonical name for explicit paths that should ship to the pod
on top of what primejob discovers via AST analysis of the entrypoint. The older
`bundle_paths` is accepted as an alias; using it sets `bundle_paths_deprecated`,
which both `primejob run` and `primejob package` consume through the shared
`warn_once` helper in `primejob.packaging_ui` so the user sees the warning
exactly once per project, no matter which command they ran first.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from primejob.packaging import dedupe_preserve_order


@dataclass
class ProjectConfig:
    dataset_disk: str | None = None
    forward_env: list[str] = field(default_factory=list)
    default_gpu: str = "H200"
    default_country: str | None = None
    default_count: int = 1
    default_disk_size: int = 50
    # Paths/globs to force-include in the tarball. The loader merges any
    # legacy `bundle_paths` entries into this list and sets
    # `bundle_paths_deprecated` so callers know to emit the rename warning.
    include: list[str] = field(default_factory=list)
    bundle_paths_deprecated: bool = False
    pyproject_path: Path | None = None
    ssh_max_wait: int = 300
    ssh_retry_delay: float = 5.0
    ssh_auth_timeout: float = 90.0
    exclude_providers: list[str] = field(default_factory=list)
    uv_extras: list[str] = field(default_factory=list)
    uv_groups: list[str] = field(default_factory=list)
    uv_all_extras: bool = False
    download_outputs: bool = True
    download_include: list[str] = field(default_factory=list)
    download_exclude: list[str] = field(default_factory=list)


def find_pyproject(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def load_project_config(cwd: Path | None = None) -> ProjectConfig:
    cwd = cwd or Path.cwd()
    pyproject = find_pyproject(cwd)
    if pyproject is None:
        return ProjectConfig()
    data = tomllib.loads(pyproject.read_text())
    section = data.get("tool", {}).get("primejob", {})

    include_raw = list(section.get("include", []))
    bundle_raw = list(section.get("bundle_paths", []))
    merged = dedupe_preserve_order(include_raw, bundle_raw)

    return ProjectConfig(
        dataset_disk=section.get("dataset_disk"),
        forward_env=list(section.get("forward_env", [])),
        default_gpu=section.get("default_gpu", "H200"),
        default_country=section.get("default_country"),
        default_count=int(section.get("default_count", 1)),
        default_disk_size=int(section.get("default_disk_size", 50)),
        include=merged,
        bundle_paths_deprecated=bool(bundle_raw),
        pyproject_path=pyproject,
        ssh_max_wait=int(section.get("ssh_max_wait", 300)),
        ssh_retry_delay=float(section.get("ssh_retry_delay", 5.0)),
        ssh_auth_timeout=float(section.get("ssh_auth_timeout", 90.0)),
        exclude_providers=list(section.get("exclude_providers", [])),
        uv_extras=list(section.get("uv_extras", [])),
        uv_groups=list(section.get("uv_groups", [])),
        uv_all_extras=bool(section.get("uv_all_extras", False)),
        download_outputs=bool(section.get("download_outputs", True)),
        download_include=list(section.get("download_include", [])),
        download_exclude=list(section.get("download_exclude", [])),
    )


def effective_gpu_count(cli_count: int | None, cfg: ProjectConfig) -> int:
    """CLI ``--count`` overrides ``[tool.primejob].default_count`` when omitted."""
    return cli_count if cli_count is not None else cfg.default_count
