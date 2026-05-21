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


def _iter_extra_files(root: Path, extra_paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for raw in extra_paths:
        path = raw if raw.is_absolute() else (root / raw)
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Bundle path not found: {raw}")
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if rel not in seen:
                seen.add(rel)
                yield path
            continue
        if not path.is_dir():
            raise NotADirectoryError(path)
        for entry in path.rglob("*"):
            if not entry.is_file():
                continue
            rel = entry.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            yield entry


def iter_project_files(root: Path, *, extra_paths: Iterable[Path] | None = None) -> Iterable[Path]:
    spec = _load_gitignore(root)
    seen: set[str] = set()

    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        rel = entry.relative_to(root).as_posix()
        if spec.match_file(rel):
            continue
        seen.add(rel)
        yield entry

    if extra_paths:
        for entry in _iter_extra_files(root, extra_paths):
            rel = entry.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            yield entry


def make_tarball(
    src_dir: Path,
    dest_path: Path,
    *,
    extra_paths: Iterable[Path] | None = None,
) -> TarResult:
    src = src_dir.resolve()
    if not src.is_dir():
        raise NotADirectoryError(src)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    size = 0
    with tarfile.open(dest_path, "w:gz") as tar:
        for f in iter_project_files(src, extra_paths=extra_paths):
            arcname = f.relative_to(src).as_posix()
            tar.add(f, arcname=arcname, recursive=False)
            count += 1
            size += f.stat().st_size
    return TarResult(path=dest_path, bytes_size=size, file_count=count)
