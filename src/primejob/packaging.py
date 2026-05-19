"""Tarball a project directory with .gitignore awareness."""
from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pathspec


DEFAULT_EXCLUDES = [
    ".git/",
    ".git",
    "__pycache__/",
    "**/__pycache__/",
    "*.pyc",
    ".venv/",
    "venv/",
    "outputs/",
    ".env",
    ".env.*",
    ".primejob/",
    ".DS_Store",
    "*.egg-info/",
    "dist/",
    "build/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
]


@dataclass
class TarResult:
    path: Path
    bytes_size: int
    file_count: int


def _load_gitignore(root: Path) -> pathspec.PathSpec:
    patterns: list[str] = list(DEFAULT_EXCLUDES)
    for gi in [root / ".gitignore", root / ".dockerignore"]:
        if gi.exists():
            patterns.extend(
                line.strip()
                for line in gi.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def iter_project_files(root: Path) -> Iterable[Path]:
    spec = _load_gitignore(root)
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        rel = entry.relative_to(root).as_posix()
        if spec.match_file(rel):
            continue
        yield entry


def make_tarball(src_dir: Path, dest_path: Path) -> TarResult:
    src = src_dir.resolve()
    if not src.is_dir():
        raise NotADirectoryError(src)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    size = 0
    with tarfile.open(dest_path, "w:gz") as tar:
        for f in iter_project_files(src):
            arcname = f.relative_to(src).as_posix()
            tar.add(f, arcname=arcname, recursive=False)
            count += 1
            size += f.stat().st_size
    return TarResult(path=dest_path, bytes_size=size, file_count=count)
