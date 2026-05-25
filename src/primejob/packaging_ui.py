"""User-facing packaging surface: progress, deprecation, size warnings, prompts.

`primejob.packaging` stays pure (analysis + tarball writing); the human-facing
concerns (UI throttling, persistence of one-time warnings, the dynamic-import
resolver that may need to prompt the user) live here so `run.py` can shrink
back into pure orchestration.

The one architectural rule worth knowing: `handle_unresolved_imports` is the
*only* function in this module that may prompt via `input()`. It runs from
the CLI layer **before** Textual takes over the terminal, so the prompt
appears in the plain shell. By the time `run_training` is invoked, the
`PackagePlan` is already resolved (CLI sets `opts.package_plan`) and
`run_training` never prompts. This is the fix for the original TUI deadlock
where `_handle_unresolved_imports` lived inside the worker thread and called
raw `input()` against a terminal Textual had captured.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Protocol

from rich.console import Console

from primejob.packaging import PackagePlan, TarResult, dedupe_preserve_order


# Tarball size above which we list the top-N largest files to expose
# unintentional bloat. Set just below typical "uplink stalls feel broken"
# thresholds while above almost any code-only tarball.
SIZE_WARN_THRESHOLD_BYTES = 100 * 1024 * 1024


class _StatusNoteSink(Protocol):
    """The narrow slice of `EventSink` these helpers actually need."""

    def status_note(self, note: str) -> None: ...


# ---------------------------------------------------------------------------
# One-time deprecation tracker (.primejob/deprecations.json)
# ---------------------------------------------------------------------------


def warn_once(
    cwd: Path,
    warning_id: str,
    message: str,
    emit: Callable[[str], None],
) -> None:
    """Emit `message` via `emit` only the first time `warning_id` is seen.

    Persists seen IDs in `.primejob/deprecations.json` so the warning fires
    exactly once across a project's lifetime, regardless of whether it
    originated from `primejob run`, `primejob package`, or another command.
    `emit` takes a plain string — pass `sink.status_note` for run_training
    or `console.print` for CLI commands.
    """
    marker = cwd / ".primejob" / "deprecations.json"
    try:
        seen = set(json.loads(marker.read_text())) if marker.exists() else set()
    except Exception:  # noqa: BLE001
        seen = set()
    if warning_id in seen:
        return
    emit(message)
    seen.add(warning_id)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(sorted(seen)))
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Include-pattern resolution (CLI args + project config, dedupe-preserving)
# ---------------------------------------------------------------------------


def resolve_include_patterns(
    cli_includes: list[str], project_includes: list[str]
) -> list[str]:
    """Merge --include-data flags with [tool.primejob].include.

    CLI patterns come first so users can override config order; duplicates
    are dropped while preserving first-seen position.
    """
    return dedupe_preserve_order(cli_includes, project_includes)


# ---------------------------------------------------------------------------
# Progress callbacks (throttled for human-readable status updates)
# ---------------------------------------------------------------------------


def make_packaging_progress(
    sink: _StatusNoteSink,
) -> Callable[[int, int], None]:
    """Throttled callback for tarball walk; fires ~2 Hz so the UI doesn't stall."""
    state = {"last_t": 0.0}

    def on_tick(file_count: int, bytes_so_far: int) -> None:
        now = time.monotonic()
        if now - state["last_t"] < 0.5:
            return
        state["last_t"] = now
        mb = bytes_so_far / (1024 ** 2)
        sink.status_note(f"  Packaging: {file_count} files, {mb:.1f} MB")

    return on_tick


def make_upload_progress(
    sink: _StatusNoteSink, total_bytes: int
) -> Callable[[int, int], None]:
    """Throttled callback wired into paramiko's SFTP put(); fires ~1 Hz."""
    state = {"last_t": 0.0, "start": time.monotonic()}

    def on_progress(sent: int, _total: int) -> None:
        now = time.monotonic()
        # Always emit the very last tick so the user sees 100% before "done".
        if sent < total_bytes and now - state["last_t"] < 1.0:
            return
        state["last_t"] = now
        mb_sent = sent / (1024 ** 2)
        mb_total = total_bytes / (1024 ** 2)
        pct = 100 * sent / total_bytes if total_bytes else 0
        elapsed = max(now - state["start"], 0.001)
        rate = mb_sent / elapsed
        sink.status_note(
            f"  Uploading: {mb_sent:.1f}/{mb_total:.1f} MB ({pct:.1f}%) at {rate:.1f} MB/s"
        )

    return on_progress


# ---------------------------------------------------------------------------
# Size warning (top-N largest files when the tarball crosses the threshold)
# ---------------------------------------------------------------------------


def emit_size_warning(sink: _StatusNoteSink, tar: TarResult) -> None:
    """Print top-N largest files if the tarball crosses SIZE_WARN_THRESHOLD_BYTES."""
    if tar.bytes_size < SIZE_WARN_THRESHOLD_BYTES:
        return
    mb = tar.bytes_size / (1024 ** 2)
    lines = [f"    {sz / (1024**2):6.1f} MB  {rel}" for rel, sz in tar.largest]
    sink.status_note(
        f"[size] Tarball is {mb:.1f} MB ({tar.file_count} files). Largest files:\n"
        + "\n".join(lines)
        + "\n  Consider trimming [tool.primejob].include if anything looks unintentional."
    )


# ---------------------------------------------------------------------------
# Unresolved dynamic-import resolver (interactive — CLI layer only)
# ---------------------------------------------------------------------------


class UnresolvedImportsAborted(Exception):
    """User chose to fix pyproject.toml instead of shipping the static closure."""


def handle_unresolved_imports(
    plan: PackagePlan,
    *,
    console: Console,
    yes: bool,
    stdin_is_tty: bool | None = None,
    stdout_is_tty: bool | None = None,
    read_line: Callable[[str], str] | None = None,
) -> None:
    """Decide what to do about dynamic imports that AST analysis can't resolve.

    --yes mode  -> loud warning, proceed.
    TTY + plain -> interactive prompt (ship-static-only or abort).
    No TTY      -> raise so silent omissions never reach the pod.

    The TTY-detection and `input()` callable are injected for testability;
    callers in production leave them at their defaults.

    *Must* be called from the plain terminal (i.e. before the TUI starts).
    `cli.py` does this upfront so the interactive prompt never collides with
    Textual's hold on stdin/stdout.
    """
    if not plan.unresolved:
        return

    if stdin_is_tty is None:
        stdin_is_tty = sys.stdin.isatty()
    if stdout_is_tty is None:
        stdout_is_tty = sys.stdout.isatty()
    if read_line is None:
        read_line = input

    lines = [
        f"  {u.file.name}:{u.lineno}  {u.description}"
        for u in plan.unresolved
    ]
    summary = (
        f"Static analysis could not resolve {len(plan.unresolved)} dynamic "
        f"import(s):\n" + "\n".join(lines)
    )

    if yes:
        console.print(f"[yellow]warn:[/yellow] {summary}")
        console.print(
            "[yellow]warn:[/yellow] Proceeding with the statically-resolved closure only. "
            "If the script fails on the pod with FileNotFoundError or ImportError, "
            r"add the resolved paths to \[tool.primejob].include in pyproject.toml."
        )
        return

    if not (stdin_is_tty and stdout_is_tty):
        raise RuntimeError(
            summary
            + "\nRe-run with --yes to ship the static closure (logged as warning), "
            r"or add the resolved paths to \[tool.primejob].include."
        )

    console.print(summary)
    try:
        answer = read_line(
            "  [s]hip static closure only / [a]bort and edit pyproject.toml? [s/A]: "
        ).strip().lower()
    except EOFError:
        answer = "a"
    if answer in {"s", "ship"}:
        console.print("  Proceeding with statically-resolved closure.")
        return
    console.print(
        "  Add the missing path(s) under [tool.primejob] in pyproject.toml:\n"
        r"  include = [\"<path/or/glob>\", ...]"
    )
    raise UnresolvedImportsAborted("Resolve dynamic imports and re-run.")


# ---------------------------------------------------------------------------
# Local-mode dataset path on the pod
# ---------------------------------------------------------------------------


def local_dataset_remote_path(plan: PackagePlan, remote_work: str) -> str | None:
    """When --data-mode local, derive the PRIMEJOB_DATASET_PATH env value.

    Delegates to `PackagePlan.local_dataset_root()` so the path reflects the
    actual files that will land on the pod. The earlier glob-stripping
    heuristic (look at the first include's literal prefix) silently misled
    users for `configs/*.yaml`-style patterns and didn't combine multiple
    includes; this consults the resolved file list instead.

    Returns None when there are no explicit includes, when the includes span
    the project root (no useful "dataset directory"), or on unexpected errors.
    """
    dataset_root = plan.local_dataset_root()
    if dataset_root is None:
        return None
    try:
        rel = dataset_root.relative_to(plan.root.resolve()).as_posix()
    except ValueError:
        return None
    if not rel or rel == ".":
        return None
    return f"{remote_work}/{rel}"
