"""Decide what files to ship to the pod, and tarball them.

Two analysis modes:

  1. *AST mode* — given a Python entrypoint, walk its `import` graph to find
     every local module it transitively depends on. Combine with an always-include
     manifest (pyproject.toml, uv.lock, README*, ...) and explicit `include`
     globs from `[tool.primejob]` (data files, dynamic-import targets, configs).

  2. *Pathspec fallback* — walk the project tree with `.gitignore` semantics,
     pruning subtrees when a directory matches an ignore pattern (mirroring
     git's own behavior — solves the `data/*` non-recursion surprise). Used
     when `analyze_package` is called without an entrypoint, primarily for
     tests and external callers that want a "ship the whole project" view.

The two modes both produce a `PackagePlan`. `make_tarball(plan)` writes a tar.

`DEFAULT_EXCLUDES` acts as a safety belt in both modes — `.uv-cache/`,
`node_modules/`, `.venv/`, secrets-bearing dotfiles, etc. never ship even
if a user's pattern would have matched them.

The packaging *UX* (progress callbacks, size warnings, the unresolved-import
resolver, deprecation tracking, include-pattern merging) lives in
`primejob.packaging_ui`. This module stays pure analysis + tarball writing.
"""
from __future__ import annotations

import ast
import os
import tarfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pathspec


# Files/dirs that never ship regardless of include patterns or .gitignore.
# Extended from the original list — `.uv-cache/`, `.cache/`, `.tox/`, `.nox/`,
# `node_modules/`, `htmlcov/`, `.ipynb_checkpoints/` are the additions that
# matter most for Python ML projects.
DEFAULT_EXCLUDES: list[str] = [
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
    # Additions covering Python ML / JS-adjacent project layouts.
    ".uv-cache/",
    ".cache/",
    ".tox/",
    ".nox/",
    "node_modules/",
    "htmlcov/",
    ".ipynb_checkpoints/",
    ".coverage",
    ".coverage.*",
]


# Top-level files that always ship if they exist. These are needed for `uv sync`
# on the pod (pyproject.toml, uv.lock) or are cheap conventional metadata.
# Matched by glob against the project root only (no recursion).
ALWAYS_INCLUDE_TOPLEVEL: list[str] = [
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "requirements-*.txt",
    ".python-version",
    "README*",
    "LICENSE*",
    "NOTICE*",
]


@dataclass
class TarResult:
    path: Path
    bytes_size: int
    file_count: int
    # (path-relative-to-root, size_bytes), sorted descending by size.
    largest: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class UnresolvedImport:
    """A dynamic import the AST walker could not statically resolve."""

    file: Path  # absolute path
    lineno: int
    description: str  # e.g. "importlib.import_module(<dynamic>)"


@dataclass
class PackagePlan:
    """The complete set of files that will land in the tarball, with provenance."""

    root: Path
    # `python_imports`: local .py files reached via AST closure from the entrypoint.
    python_imports: list[Path] = field(default_factory=list)
    # `always_included`: pyproject.toml, uv.lock, README*, etc.
    always_included: list[Path] = field(default_factory=list)
    # `explicit_includes`: files matched by [tool.primejob].include patterns.
    explicit_includes: list[Path] = field(default_factory=list)
    # `pathspec_walk`: files from the gitignore-aware walk (fallback mode only).
    pathspec_walk: list[Path] = field(default_factory=list)
    # Dynamic imports the AST walker could not resolve.
    unresolved: list[UnresolvedImport] = field(default_factory=list)

    def all_files(self) -> list[Path]:
        """Deduplicated union of every source list, preserving first-seen order."""
        seen: set[Path] = set()
        out: list[Path] = []
        for group in (
            self.python_imports,
            self.always_included,
            self.explicit_includes,
            self.pathspec_walk,
        ):
            for f in group:
                resolved = f.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                out.append(f)
        return out

    def summary(self) -> str:
        """One-line human summary for the status sink."""
        parts: list[str] = []
        if self.python_imports:
            parts.append(f"{len(self.python_imports)} python")
        if self.explicit_includes:
            parts.append(f"{len(self.explicit_includes)} include")
        if self.always_included:
            parts.append(f"{len(self.always_included)} always")
        if self.pathspec_walk:
            parts.append(f"{len(self.pathspec_walk)} from walk")
        return ", ".join(parts) if parts else "0 files"

    def local_dataset_root(self) -> Path | None:
        """Common ancestor of `explicit_includes`, suitable for PRIMEJOB_DATASET_PATH.

        Returns the deepest directory that contains every explicitly-included
        file. Returns None if there are no explicit includes, if the common
        ancestor is the project root itself (no useful "dataset directory"),
        or if a single file include points at a file in the root.

        Used by `--data-mode local` to set the env var on the pod. The earlier
        glob-stripping heuristic (look at the first pattern's literal prefix)
        broke on patterns like `[abc]/data/*` and silently misled the user
        when only `configs/*.yaml` matched. This method consults the matched
        files instead, so the result is always a real directory the user's
        script can `os.scandir()`.
        """
        if not self.explicit_includes:
            return None
        try:
            root_resolved = self.root.resolve()
        except OSError:
            return None
        ancestor: Path | None = None
        for f in self.explicit_includes:
            try:
                parent = f.resolve().parent
            except OSError:
                continue
            ancestor = parent if ancestor is None else _common_ancestor(ancestor, parent)
            if ancestor is None or ancestor == root_resolved:
                return None
        if ancestor is None or ancestor == root_resolved:
            return None
        try:
            ancestor.relative_to(root_resolved)
        except ValueError:
            return None
        return ancestor


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_package(
    root: Path,
    *,
    entrypoint: Path | None = None,
    include: Iterable[str] = (),
) -> PackagePlan:
    """Decide what files belong in the tarball.

    If `entrypoint` is given and is a `.py` file, walk its import closure.
    Otherwise fall back to a pruned `.gitignore`-aware walk of `root`.
    `include` patterns are always honored (shell-glob, root-relative).
    `DEFAULT_EXCLUDES` is applied as a safety filter in both modes.
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    plan = PackagePlan(root=root)
    plan.always_included = list(_collect_always_included(root))
    plan.explicit_includes = list(_collect_explicit_includes(root, include))

    if entrypoint is not None and entrypoint.suffix == ".py":
        entry = entrypoint if entrypoint.is_absolute() else (root / entrypoint)
        entry = entry.resolve()
        if not entry.is_file():
            raise FileNotFoundError(f"Entrypoint not found: {entrypoint}")
        if not _is_inside(entry, root):
            raise ValueError(f"Entrypoint must live inside {root}: {entrypoint}")
        python_files, unresolved = _walk_imports(root, entry)
        plan.python_imports = sorted(python_files)
        plan.unresolved = unresolved
    else:
        plan.pathspec_walk = list(_pathspec_walk(root))

    return plan


def _collect_always_included(root: Path) -> list[Path]:
    """Top-level files matching ALWAYS_INCLUDE_TOPLEVEL globs (no recursion)."""
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in ALWAYS_INCLUDE_TOPLEVEL:
        for match in sorted(root.glob(pattern)):
            if not match.is_file():
                continue
            resolved = match.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(match)
    return out


def _collect_explicit_includes(root: Path, patterns: Iterable[str]) -> list[Path]:
    """Expand shell-glob patterns rooted at `root` into a file list.

    A pattern ending in `/` or naming a directory expands to the whole subtree.
    DEFAULT_EXCLUDES is applied as a safety belt.
    """
    excluder = pathspec.PathSpec.from_lines("gitignore", DEFAULT_EXCLUDES)
    out: list[Path] = []
    seen: set[Path] = set()

    for pattern in patterns:
        pat = pattern.strip()
        if not pat:
            continue
        # Normalize directory-shorthand: "data/pii/" -> "data/pii/**/*".
        # Also handle a bare directory name without the trailing slash.
        if pat.endswith("/"):
            pat = pat + "**/*"
        else:
            candidate = root / pat
            if candidate.is_dir():
                pat = pat.rstrip("/") + "/**/*"

        # Path.glob in Python 3.11 supports `**` only as the entire segment;
        # use rglob when pattern starts with **/ for portability.
        for match in _glob_root(root, pat):
            if not match.is_file():
                continue
            resolved = match.resolve()
            if resolved in seen:
                continue
            rel = resolved.relative_to(root).as_posix()
            if excluder.match_file(rel):
                continue
            seen.add(resolved)
            out.append(match)

    return out


def _glob_root(root: Path, pattern: str) -> Iterable[Path]:
    """Glob `pattern` relative to `root`, accepting shell-style `**`."""
    # Path.glob handles `**` since Python 3.5 but only as a full path component.
    # All our patterns are root-relative, so this is sufficient.
    return root.glob(pattern)


def _pathspec_walk(root: Path) -> Iterable[Path]:
    """Walk `root` honoring .gitignore + DEFAULT_EXCLUDES, pruning subtrees.

    This is the fallback for when no entrypoint is supplied. Critically, it
    *prunes* directories that match an ignore pattern instead of recursing
    into them and per-file filtering — matching git's own behavior and fixing
    the `data/*` non-recursion bug.
    """
    spec = _load_gitignore(root)

    # os.walk with topdown=True so we can mutate dirnames in place to prune.
    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        # Prune subdirectories that match the ignore spec.
        remaining: list[str] = []
        for d in dirnames:
            rel = (dirpath / d).resolve().relative_to(root).as_posix() + "/"
            if spec.match_file(rel) or spec.match_file(rel.rstrip("/")):
                continue
            remaining.append(d)
        dirnames[:] = remaining

        for f in filenames:
            fp = dirpath / f
            if not fp.is_file():
                continue
            rel = fp.resolve().relative_to(root).as_posix()
            if spec.match_file(rel):
                continue
            yield fp


def _load_gitignore(root: Path) -> pathspec.PathSpec:
    patterns: list[str] = list(DEFAULT_EXCLUDES)
    for gi in (root / ".gitignore", root / ".dockerignore"):
        if gi.exists():
            for line in gi.read_text().splitlines():
                stripped = line.strip()
                if stripped and not stripped.lstrip().startswith("#"):
                    patterns.append(stripped)
    return pathspec.PathSpec.from_lines("gitignore", patterns)


# ---------------------------------------------------------------------------
# AST import walking
# ---------------------------------------------------------------------------


def _walk_imports(root: Path, entrypoint: Path) -> tuple[set[Path], list[UnresolvedImport]]:
    """Walk the import closure of `entrypoint`, returning (local_files, unresolved).

    Local files are those that resolve to a module under `root` (or under
    `root/src/` for the src-layout convention). Third-party and stdlib imports
    are skipped — they install via `uv sync` on the pod.
    """
    local_modules = _build_local_modules(root)
    # Reverse index: file path -> module name. Avoids an O(n) scan over
    # local_modules every time _walk_imports visits a file. When the same
    # file appears under multiple dotted names (e.g. flat + src-layout),
    # prefer the longest name so relative-import resolution has the
    # correct anchor.
    path_to_module: dict[Path, str] = {}
    for name, p in local_modules.items():
        existing = path_to_module.get(p)
        if existing is None or len(name) > len(existing):
            path_to_module[p] = name

    queue: list[Path] = [entrypoint]
    visited: set[Path] = set()
    unresolved: list[UnresolvedImport] = []

    while queue:
        py = queue.pop()
        py_resolved = py.resolve()
        if py_resolved in visited:
            continue
        visited.add(py_resolved)

        try:
            source = py.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        try:
            tree = ast.parse(source, filename=str(py))
        except SyntaxError:
            continue

        # Module name of the current file (for resolving relative imports).
        current_module = path_to_module.get(py_resolved)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = local_modules.get(alias.name) or local_modules.get(
                        _toplevel(alias.name)
                    )
                    if target and target not in visited:
                        queue.append(target)
                        # Also queue the package __init__.py chain.
                        for parent in _parent_modules(alias.name):
                            p = local_modules.get(parent)
                            if p and p not in visited:
                                queue.append(p)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                resolved_name = _resolve_relative(module, node.level, current_module)
                if resolved_name is None:
                    continue
                # The module itself.
                target = local_modules.get(resolved_name)
                if target and target not in visited:
                    queue.append(target)
                # Each `from X import name` may name a submodule, not just an attr.
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    sub = f"{resolved_name}.{alias.name}" if resolved_name else alias.name
                    sub_target = local_modules.get(sub)
                    if sub_target and sub_target not in visited:
                        queue.append(sub_target)
                # Queue parent __init__.py files.
                for parent in _parent_modules(resolved_name):
                    p = local_modules.get(parent)
                    if p and p not in visited:
                        queue.append(p)
            elif isinstance(node, ast.Call):
                u = _detect_dynamic_import(node, py_resolved)
                if u is not None:
                    unresolved.append(u)

    return visited, unresolved


def _build_local_modules(root: Path) -> dict[str, Path]:
    """Map dotted module name -> file path for every .py under `root`.

    Considers both flat layout (`mypkg/__init__.py` at root) and src layout
    (`src/mypkg/__init__.py`). If `src/__init__.py` exists, `src` is a
    regular package (not a src-layout root), so we register only the
    `src.*`-prefixed names and skip the bare `main`-style alias. This
    keeps the reverse index unambiguous for relative-import resolution.
    """
    out: dict[str, Path] = {}
    src_roots: list[Path] = [root]
    src_dir = root / "src"
    if src_dir.is_dir() and not (src_dir / "__init__.py").is_file():
        src_roots.append(src_dir)

    excluder = pathspec.PathSpec.from_lines("gitignore", DEFAULT_EXCLUDES)

    for src_root in src_roots:
        for py in src_root.rglob("*.py"):
            try:
                rel_to_root = py.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            if excluder.match_file(rel_to_root):
                continue
            rel = py.relative_to(src_root)
            parts = list(rel.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts:
                continue
            name = ".".join(parts)
            # First registration wins so flat layout beats nested duplicates.
            out.setdefault(name, py.resolve())
    return out


def _toplevel(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _parent_modules(dotted: str) -> list[str]:
    """For `a.b.c.d` return `['a', 'a.b', 'a.b.c']`."""
    parts = dotted.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts) - 1)]


def _resolve_relative(module: str, level: int, current: str | None) -> str | None:
    """Resolve a `from X import Y` to an absolute module name.

    `level=0` is an absolute import. `level>0` is a relative import; we strip
    that many components from `current` (which is the importing file's module
    name) and prepend the result to `module`.
    """
    if level == 0:
        return module or None
    if current is None:
        return None
    parts = current.split(".")
    # `current` includes the file's own name. For relative imports inside a
    # package's `__init__.py`, `current` is the package itself, so level 1
    # refers to the same package. For a non-__init__ file, level 1 refers
    # to the parent package. We don't track __init__ status precisely; use
    # the conservative "drop `level` components" rule, which matches the
    # common case.
    if level > len(parts):
        return None
    anchor = parts[:-level] if level <= len(parts) else []
    if module:
        return ".".join([*anchor, module])
    return ".".join(anchor) if anchor else None


def _detect_dynamic_import(node: ast.Call, file: Path) -> UnresolvedImport | None:
    """Flag `importlib.import_module(x)` and `__import__(x)` when x is not a literal."""
    func = node.func
    name: str | None = None
    if isinstance(func, ast.Attribute):
        if (
            isinstance(func.value, ast.Name)
            and func.value.id == "importlib"
            and func.attr == "import_module"
        ):
            name = "importlib.import_module"
    elif isinstance(func, ast.Name) and func.id == "__import__":
        name = "__import__"

    if name is None:
        return None
    # Literal string first arg -> we could resolve statically; not an unresolved.
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return None
    return UnresolvedImport(
        file=file,
        lineno=getattr(node, "lineno", 0),
        description=f"{name}(<dynamic>)",
    )


# ---------------------------------------------------------------------------
# Tarball writing
# ---------------------------------------------------------------------------


def make_tarball(
    root: Path,
    dest: Path,
    plan: PackagePlan,
    *,
    on_tick: Callable[[int, int], None] | None = None,
    top_n_largest: int = 5,
) -> TarResult:
    """Write the files in `plan` to a gzipped tar at `dest`.

    `on_tick(file_count, bytes_so_far)` fires every ~200 files / ~500 ms so
    callers can render packaging progress. The final `TarResult.largest`
    is the top-N largest files by size for the size-warning UI.
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    dest.parent.mkdir(parents=True, exist_ok=True)

    files = plan.all_files()
    # Pre-compute sizes once (we need them for the top-N largest anyway).
    sized: list[tuple[Path, int]] = []
    for f in files:
        try:
            sized.append((f, f.stat().st_size))
        except OSError:
            continue

    count = 0
    bytes_in = 0
    last_tick_count = 0
    import time as _time
    last_tick_t = _time.monotonic()

    with tarfile.open(dest, "w:gz") as tar:
        for f, size in sized:
            try:
                arcname = f.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            tar.add(f, arcname=arcname, recursive=False)
            count += 1
            bytes_in += size
            if on_tick is not None:
                now = _time.monotonic()
                if count - last_tick_count >= 200 or (now - last_tick_t) >= 0.5:
                    on_tick(count, bytes_in)
                    last_tick_count = count
                    last_tick_t = now

    # Final tick so the UI shows the final count.
    if on_tick is not None and count != last_tick_count:
        on_tick(count, bytes_in)

    largest = [
        (p.resolve().relative_to(root).as_posix(), s)
        for p, s in sorted(sized, key=lambda x: x[1], reverse=True)[:top_n_largest]
    ]
    return TarResult(
        path=dest,
        bytes_size=dest.stat().st_size,
        file_count=count,
        largest=largest,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _common_ancestor(a: Path, b: Path) -> Path | None:
    """Deepest directory that is a prefix of both `a` and `b`. None if disjoint."""
    a_parts = a.parts
    b_parts = b.parts
    shared: list[str] = []
    for x, y in zip(a_parts, b_parts):
        if x != y:
            break
        shared.append(x)
    if not shared:
        return None
    return Path(*shared)


def dedupe_preserve_order(*lists: Iterable[str]) -> list[str]:
    """Concatenate the inputs and remove duplicates, keeping first-seen order.

    The canonical helper for merging include / exclude lists across config
    + CLI + project. Three call sites used to reimplement this with a `seen`
    set; they all now route through here.
    """
    seen: set[str] = set()
    out: list[str] = []
    for items in lists:
        for entry in items:
            if entry not in seen:
                seen.add(entry)
                out.append(entry)
    return out
