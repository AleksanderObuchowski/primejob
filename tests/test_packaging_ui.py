"""Unit tests for primejob.packaging_ui — the human-facing packaging surface.

Covers the deprecation warning persistence, throttled progress callbacks,
size hint, include-pattern merging, the unresolved-import resolver (with
its TTY-detection seams exposed for testability), and the local-dataset
path derivation.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from primejob.packaging import (
    PackagePlan,
    TarResult,
    UnresolvedImport,
    analyze_package,
)
from primejob.packaging_ui import (
    SIZE_WARN_THRESHOLD_BYTES,
    UnresolvedImportsAborted,
    emit_size_warning,
    handle_unresolved_imports,
    local_dataset_remote_path,
    make_packaging_progress,
    make_upload_progress,
    resolve_include_patterns,
    warn_once,
)


class _FakeSink:
    """Captures status/note calls so tests can assert on them."""

    def __init__(self) -> None:
        self.notes: list[str] = []

    def status_note(self, msg: str) -> None:
        self.notes.append(msg)


def _stub_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=200), buf


# ---------------------------------------------------------------------------
# warn_once: fires once per project across runs
# ---------------------------------------------------------------------------


def test_warn_once_fires_only_first_time(tmp_path: Path) -> None:
    seen: list[str] = []
    warn_once(tmp_path, "bundle_paths_renamed", "rename your config", seen.append)
    warn_once(tmp_path, "bundle_paths_renamed", "rename your config", seen.append)
    warn_once(tmp_path, "bundle_paths_renamed", "rename your config", seen.append)
    assert seen == ["rename your config"]


def test_warn_once_persists_marker(tmp_path: Path) -> None:
    warn_once(tmp_path, "X", "msg-x", lambda _msg: None)
    marker = tmp_path / ".primejob" / "deprecations.json"
    assert marker.exists()
    assert "X" in marker.read_text()


def test_warn_once_distinct_ids_both_fire(tmp_path: Path) -> None:
    seen: list[str] = []
    warn_once(tmp_path, "A", "msg-a", seen.append)
    warn_once(tmp_path, "B", "msg-b", seen.append)
    assert seen == ["msg-a", "msg-b"]


def test_warn_once_tolerates_corrupt_marker(tmp_path: Path) -> None:
    marker = tmp_path / ".primejob" / "deprecations.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("not json!")
    seen: list[str] = []
    warn_once(tmp_path, "X", "msg-x", seen.append)
    assert seen == ["msg-x"]


def test_warn_once_shared_marker_across_commands(tmp_path: Path) -> None:
    """The marker is project-keyed so `primejob run` and `primejob package`
    don't each fire the same deprecation. This is the M3 acceptance test."""
    notes_run: list[str] = []
    notes_pkg: list[str] = []
    warn_once(tmp_path, "dep1", "warning", notes_run.append)
    warn_once(tmp_path, "dep1", "warning", notes_pkg.append)
    assert notes_run == ["warning"]
    assert notes_pkg == []


# ---------------------------------------------------------------------------
# resolve_include_patterns: merge + dedupe + order
# ---------------------------------------------------------------------------


def test_resolve_include_patterns_defaults_empty() -> None:
    assert resolve_include_patterns([], []) == []


def test_resolve_include_patterns_merges_cli_and_config() -> None:
    out = resolve_include_patterns(["extra.jsonl"], ["configs/", "data/pii/*.jsonl"])
    assert out == ["configs/", "data/pii/*.jsonl", "extra.jsonl"]


def test_resolve_include_patterns_dedupes_across_sources() -> None:
    out = resolve_include_patterns(
        ["data/foo.jsonl"], ["data/foo.jsonl", "configs/"]
    )
    assert out == ["data/foo.jsonl", "configs/"]


# ---------------------------------------------------------------------------
# Packaging progress callback
# ---------------------------------------------------------------------------


def test_packaging_progress_throttles() -> None:
    sink = _FakeSink()
    cb = make_packaging_progress(sink)
    cb(100, 1_000_000)
    cb(101, 1_010_000)  # within 0.5s window → dropped
    time.sleep(0.6)
    cb(200, 2_000_000)
    assert len(sink.notes) == 2
    assert "100 files" in sink.notes[0]
    assert "200 files" in sink.notes[1]


def test_packaging_progress_formats_size() -> None:
    sink = _FakeSink()
    cb = make_packaging_progress(sink)
    cb(50, 5 * 1024 * 1024)
    assert "5.0 MB" in sink.notes[0]


# ---------------------------------------------------------------------------
# Upload progress callback
# ---------------------------------------------------------------------------


def test_upload_progress_emits_at_completion() -> None:
    sink = _FakeSink()
    cb = make_upload_progress(sink, total_bytes=10 * 1024 * 1024)
    cb(10 * 1024 * 1024, 10 * 1024 * 1024)
    assert sink.notes, "expected a final tick"
    assert "100.0%" in sink.notes[-1]


def test_upload_progress_throttles_intermediate() -> None:
    sink = _FakeSink()
    cb = make_upload_progress(sink, total_bytes=10 * 1024 * 1024)
    cb(1 * 1024 * 1024, 10 * 1024 * 1024)
    cb(2 * 1024 * 1024, 10 * 1024 * 1024)
    cb(3 * 1024 * 1024, 10 * 1024 * 1024)
    # All three within ~ms; only the first should appear.
    assert len(sink.notes) == 1


# ---------------------------------------------------------------------------
# Size warning
# ---------------------------------------------------------------------------


def test_size_warning_silent_under_threshold() -> None:
    sink = _FakeSink()
    tar = TarResult(
        path=Path("/tmp/x.tar.gz"),
        bytes_size=10 * 1024 * 1024,
        file_count=5,
        largest=[("a.txt", 5_000_000)],
    )
    emit_size_warning(sink, tar)
    assert sink.notes == []


def test_size_warning_prints_top_n_over_threshold() -> None:
    sink = _FakeSink()
    tar = TarResult(
        path=Path("/tmp/x.tar.gz"),
        bytes_size=SIZE_WARN_THRESHOLD_BYTES + 1,
        file_count=200,
        largest=[
            ("data/huge.bin", 50 * 1024 * 1024),
            ("data/big.bin", 30 * 1024 * 1024),
        ],
    )
    emit_size_warning(sink, tar)
    assert len(sink.notes) == 1
    note = sink.notes[0]
    assert "Tarball is" in note
    assert "data/huge.bin" in note
    assert "data/big.bin" in note
    assert "50.0 MB" in note


# ---------------------------------------------------------------------------
# Unresolved-import policy
# ---------------------------------------------------------------------------


def _plan_with_unresolved() -> PackagePlan:
    return PackagePlan(
        root=Path("/tmp"),
        unresolved=[
            UnresolvedImport(
                file=Path("/tmp/train.py"),
                lineno=42,
                description="importlib.import_module(<dynamic>)",
            )
        ],
    )


def test_unresolved_empty_plan_is_noop() -> None:
    """A clean plan never even consults the TTY or reads input."""
    console, buf = _stub_console()
    handle_unresolved_imports(PackagePlan(root=Path("/tmp")), console=console, yes=False)
    assert buf.getvalue() == ""


def test_unresolved_yes_mode_warns_and_proceeds() -> None:
    console, buf = _stub_console()
    handle_unresolved_imports(_plan_with_unresolved(), console=console, yes=True)
    out = buf.getvalue()
    assert "warn:" in out
    assert "train.py:42" in out


def test_unresolved_no_tty_errors() -> None:
    """When stdin/stdout aren't TTYs (CI, pipes, TUI worker thread), error."""
    console, _ = _stub_console()
    with pytest.raises(RuntimeError, match="Static analysis could not resolve"):
        handle_unresolved_imports(
            _plan_with_unresolved(),
            console=console,
            yes=False,
            stdin_is_tty=False,
            stdout_is_tty=False,
        )


def test_unresolved_tty_ship_answer() -> None:
    console, buf = _stub_console()
    handle_unresolved_imports(
        _plan_with_unresolved(),
        console=console,
        yes=False,
        stdin_is_tty=True,
        stdout_is_tty=True,
        read_line=lambda _prompt: "s",
    )
    assert "Proceeding" in buf.getvalue()


def test_unresolved_tty_abort_answer() -> None:
    console, _ = _stub_console()
    with pytest.raises(UnresolvedImportsAborted):
        handle_unresolved_imports(
            _plan_with_unresolved(),
            console=console,
            yes=False,
            stdin_is_tty=True,
            stdout_is_tty=True,
            read_line=lambda _prompt: "a",
        )


def test_unresolved_tty_empty_answer_aborts() -> None:
    """Empty input defaults to the safe choice (abort)."""
    console, _ = _stub_console()
    with pytest.raises(UnresolvedImportsAborted):
        handle_unresolved_imports(
            _plan_with_unresolved(),
            console=console,
            yes=False,
            stdin_is_tty=True,
            stdout_is_tty=True,
            read_line=lambda _prompt: "",
        )


def test_unresolved_eof_during_prompt_aborts() -> None:
    """If input() raises EOFError (e.g. piped stdin closed), treat as abort."""
    console, _ = _stub_console()

    def raise_eof(_prompt: str) -> str:
        raise EOFError

    with pytest.raises(UnresolvedImportsAborted):
        handle_unresolved_imports(
            _plan_with_unresolved(),
            console=console,
            yes=False,
            stdin_is_tty=True,
            stdout_is_tty=True,
            read_line=raise_eof,
        )


# ---------------------------------------------------------------------------
# local_dataset_remote_path: derives PRIMEJOB_DATASET_PATH from PackagePlan
# ---------------------------------------------------------------------------


def _touch(p: Path, content: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_local_dataset_remote_path_from_plan(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "data" / "pii" / "a.jsonl", "x")
    plan = analyze_package(tmp_path, include=["data/pii/"])
    out = local_dataset_remote_path(plan, "/tmp/primejob/work")
    assert out == "/tmp/primejob/work/data/pii"


def test_local_dataset_remote_path_returns_none_when_empty(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    plan = analyze_package(tmp_path, include=[])
    assert local_dataset_remote_path(plan, "/tmp/primejob/work") is None


def test_local_dataset_remote_path_returns_none_when_includes_span_root(
    tmp_path: Path,
) -> None:
    """Includes across two siblings collapse to root → no useful dataset dir."""
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "configs" / "a.yaml", "x")
    _touch(tmp_path / "data" / "b.jsonl", "y")
    plan = analyze_package(tmp_path, include=["configs/*.yaml", "data/*"])
    assert local_dataset_remote_path(plan, "/tmp/primejob/work") is None


def test_local_dataset_remote_path_glob_pattern_uses_actual_files(
    tmp_path: Path,
) -> None:
    """The old glob-stripping heuristic broke on patterns whose literal prefix
    didn't match the resolved files. This test covers that regression: an
    include like `data/**/*.jsonl` finds files under data/raw/, and the
    remote path should reflect their parent — not the pattern's literal
    prefix `data` (which the old impl would have returned)."""
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "data" / "raw" / "shard0.jsonl", "x")
    _touch(tmp_path / "data" / "raw" / "shard1.jsonl", "y")
    plan = analyze_package(tmp_path, include=["data/**/*.jsonl"])
    out = local_dataset_remote_path(plan, "/tmp/primejob/work")
    assert out == "/tmp/primejob/work/data/raw"
