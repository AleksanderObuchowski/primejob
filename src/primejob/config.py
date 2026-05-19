"""Project-level config from [tool.primejob] in pyproject.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectConfig:
    dataset_disk: str | None = None
    forward_env: list[str] = field(default_factory=list)
    default_gpu: str = "H200"
    default_country: str | None = None
    default_count: int = 1
    default_disk_size: int = 50
    pyproject_path: Path | None = None


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
    return ProjectConfig(
        dataset_disk=section.get("dataset_disk"),
        forward_env=list(section.get("forward_env", [])),
        default_gpu=section.get("default_gpu", "H200"),
        default_country=section.get("default_country"),
        default_count=int(section.get("default_count", 1)),
        default_disk_size=int(section.get("default_disk_size", 50)),
        pyproject_path=pyproject,
    )
