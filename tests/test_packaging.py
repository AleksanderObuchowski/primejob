"""Tarball respects .gitignore and default excludes."""
from __future__ import annotations

import tarfile
from pathlib import Path

from primejob.packaging import iter_project_files, make_tarball


def _scaffold(tmp_path: Path) -> Path:
    """Create a fake project tree."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "keep.txt").write_text("keep")
    (tmp_path / "data" / "raw" / "ignored.bin").write_text("nope")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "old.log").write_text("log")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_text("")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / ".gitignore").write_text("data/raw/*.bin\n")
    return tmp_path


def test_iter_excludes_defaults(tmp_path: Path) -> None:
    root = _scaffold(tmp_path)
    files = {p.relative_to(root).as_posix() for p in iter_project_files(root)}
    assert "src/main.py" in files
    assert "data/raw/keep.txt" in files
    assert "pyproject.toml" in files
    assert "data/raw/ignored.bin" not in files  # gitignored
    assert ".env" not in files                  # default excluded
    assert "outputs/old.log" not in files
    assert "__pycache__/x.pyc" not in files
    assert ".git/HEAD" not in files


def test_make_tarball_roundtrip(tmp_path: Path) -> None:
    root = _scaffold(tmp_path)
    dest = tmp_path / ".primejob" / "src.tar.gz"
    res = make_tarball(root, dest)
    assert dest.exists()
    assert res.file_count >= 3  # main.py, keep.txt, pyproject.toml, .gitignore
    with tarfile.open(dest, "r:gz") as t:
        names = set(t.getnames())
    assert "src/main.py" in names
    assert "data/raw/keep.txt" in names
    assert ".env" not in names
    assert "outputs/old.log" not in names


def test_make_tarball_includes_gitignored_extra_path(tmp_path: Path) -> None:
    root = _scaffold(tmp_path)
    dest = tmp_path / ".primejob" / "src.tar.gz"
    res = make_tarball(root, dest, extra_paths=[root / "data" / "raw" / "ignored.bin"])
    with tarfile.open(dest, "r:gz") as t:
        names = set(t.getnames())
    assert "data/raw/ignored.bin" in names
    assert res.file_count >= 4
