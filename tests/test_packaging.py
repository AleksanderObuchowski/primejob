"""Tests for primejob.packaging — AST closure, pathspec walk, and tarball write."""
from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from primejob.packaging import (
    ALWAYS_INCLUDE_TOPLEVEL,
    DEFAULT_EXCLUDES,
    analyze_package,
    dedupe_preserve_order,
    make_tarball,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(p: Path, content: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _arcnames(tarball: Path) -> set[str]:
    with tarfile.open(tarball, "r:gz") as tf:
        return {m.name for m in tf.getmembers() if m.isfile()}


# ---------------------------------------------------------------------------
# AST closure mode
# ---------------------------------------------------------------------------


def test_ast_closure_picks_up_transitive_local_imports(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _touch(tmp_path / "uv.lock", "")
    _touch(
        tmp_path / "train.py",
        "from src.models import Model\nfrom src.data import load\n",
    )
    _touch(tmp_path / "src" / "__init__.py", "")
    _touch(tmp_path / "src" / "models.py", "from src.layers import Linear\n")
    _touch(tmp_path / "src" / "data.py", "import json\n")
    _touch(tmp_path / "src" / "layers.py", "")
    # An unrelated file that nobody imports — must NOT ship.
    _touch(tmp_path / "src" / "ghost.py", "raise RuntimeError('orphan')\n")

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.py")
    imported = {p.relative_to(tmp_path).as_posix() for p in plan.python_imports}
    assert "train.py" in imported
    assert "src/__init__.py" in imported
    assert "src/models.py" in imported
    assert "src/data.py" in imported
    assert "src/layers.py" in imported
    assert "src/ghost.py" not in imported


def test_ast_closure_skips_third_party_and_stdlib(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _touch(tmp_path / "uv.lock", "")
    _touch(
        tmp_path / "train.py",
        "import torch\nimport json\nfrom transformers import AutoModel\n",
    )

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.py")
    imported = {p.relative_to(tmp_path).as_posix() for p in plan.python_imports}
    assert "train.py" in imported
    # torch / transformers must NOT be added — they're third-party.
    assert not any("torch" in name for name in imported)
    assert not any("transformers" in name for name in imported)


def test_ast_closure_resolves_relative_imports(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "src" / "__init__.py", "")
    _touch(
        tmp_path / "src" / "main.py",
        "from .utils import helper\nfrom . import config\n",
    )
    _touch(tmp_path / "src" / "utils.py", "")
    _touch(tmp_path / "src" / "config.py", "")

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "src" / "main.py")
    imported = {p.relative_to(tmp_path).as_posix() for p in plan.python_imports}
    assert "src/main.py" in imported
    assert "src/utils.py" in imported
    assert "src/config.py" in imported


def test_ast_closure_supports_src_layout(tmp_path: Path) -> None:
    """src/foo/bar.py should be reachable as `foo.bar`."""
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "from foo.bar import thing\n")
    _touch(tmp_path / "src" / "foo" / "__init__.py", "")
    _touch(tmp_path / "src" / "foo" / "bar.py", "")

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.py")
    imported = {p.resolve().as_posix() for p in plan.python_imports}
    assert (tmp_path / "src" / "foo" / "bar.py").resolve().as_posix() in imported
    assert (tmp_path / "src" / "foo" / "__init__.py").resolve().as_posix() in imported


def test_ast_closure_flags_dynamic_imports(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _touch(tmp_path / "uv.lock", "")
    _touch(
        tmp_path / "train.py",
        "import importlib\n"
        "name = some_runtime_value()\n"
        "mod = importlib.import_module(name)\n"
        "literal = importlib.import_module('json')  # this one IS resolvable\n",
    )

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.py")
    assert len(plan.unresolved) == 1
    u = plan.unresolved[0]
    assert "importlib.import_module" in u.description
    assert u.lineno == 3


def test_ast_closure_handles_broken_python_gracefully(tmp_path: Path) -> None:
    """A SyntaxError in a sibling file must not crash analysis of train.py."""
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "from src.broken import thing\n")
    _touch(tmp_path / "src" / "__init__.py", "")
    _touch(tmp_path / "src" / "broken.py", "def f(:::: invalid\n")

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.py")
    # Both should still be in the closure (broken.py is referenced by name).
    imported = {p.relative_to(tmp_path).as_posix() for p in plan.python_imports}
    assert "train.py" in imported
    assert "src/broken.py" in imported


# ---------------------------------------------------------------------------
# Always-include manifest
# ---------------------------------------------------------------------------


def test_always_includes_pyproject_and_lock(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "README.md", "# x")
    _touch(tmp_path / "LICENSE", "MIT")
    _touch(tmp_path / "train.py", "")

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.py")
    always = {p.relative_to(tmp_path).as_posix() for p in plan.always_included}
    assert "pyproject.toml" in always
    assert "uv.lock" in always
    assert "README.md" in always
    assert "LICENSE" in always


def test_always_includes_skips_missing_optional_files(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.py")
    always = {p.relative_to(tmp_path).as_posix() for p in plan.always_included}
    assert always == {"pyproject.toml", "uv.lock"}


# ---------------------------------------------------------------------------
# Explicit include patterns
# ---------------------------------------------------------------------------


def test_include_literal_file(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")
    _touch(tmp_path / "data" / "pii" / "foo.jsonl", "...")
    _touch(tmp_path / "data" / "pii" / "bar.jsonl", "...")

    plan = analyze_package(
        tmp_path,
        entrypoint=tmp_path / "train.py",
        include=["data/pii/foo.jsonl"],
    )
    inc = {p.relative_to(tmp_path).as_posix() for p in plan.explicit_includes}
    assert inc == {"data/pii/foo.jsonl"}


def test_include_glob_pattern(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")
    _touch(tmp_path / "data" / "pii" / "foo.jsonl", "...")
    _touch(tmp_path / "data" / "pii" / "bar.jsonl", "...")
    _touch(tmp_path / "data" / "pii" / "skip.tgz", "...")

    plan = analyze_package(
        tmp_path,
        entrypoint=tmp_path / "train.py",
        include=["data/pii/*.jsonl"],
    )
    inc = {p.relative_to(tmp_path).as_posix() for p in plan.explicit_includes}
    assert inc == {"data/pii/foo.jsonl", "data/pii/bar.jsonl"}


def test_include_directory_shorthand_recurses(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")
    _touch(tmp_path / "configs" / "base.yaml", "")
    _touch(tmp_path / "configs" / "exp" / "a.yaml", "")

    plan = analyze_package(
        tmp_path,
        entrypoint=tmp_path / "train.py",
        include=["configs/"],
    )
    inc = {p.relative_to(tmp_path).as_posix() for p in plan.explicit_includes}
    assert inc == {"configs/base.yaml", "configs/exp/a.yaml"}


def test_include_double_star_glob(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")
    _touch(tmp_path / "configs" / "base.yaml", "")
    _touch(tmp_path / "configs" / "exp" / "a.yaml", "")
    _touch(tmp_path / "configs" / "exp" / "b.json", "")

    plan = analyze_package(
        tmp_path,
        entrypoint=tmp_path / "train.py",
        include=["configs/**/*.yaml"],
    )
    inc = {p.relative_to(tmp_path).as_posix() for p in plan.explicit_includes}
    assert "configs/base.yaml" in inc
    assert "configs/exp/a.yaml" in inc
    assert "configs/exp/b.json" not in inc


# ---------------------------------------------------------------------------
# DEFAULT_EXCLUDES safety belt
# ---------------------------------------------------------------------------


def test_default_excludes_block_uv_cache(tmp_path: Path) -> None:
    """A wide include pattern must NOT pull .uv-cache files."""
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")
    _touch(tmp_path / ".uv-cache" / "simple-v20" / "pypi" / "aiohttp.rkyv", "x" * 100)
    _touch(tmp_path / "data" / "real.jsonl", "...")

    plan = analyze_package(
        tmp_path,
        entrypoint=tmp_path / "train.py",
        include=["**/*"],  # user accidentally writes "everything"
    )
    inc = {p.relative_to(tmp_path).as_posix() for p in plan.explicit_includes}
    assert "data/real.jsonl" in inc
    assert not any(".uv-cache" in name for name in inc)


def test_default_excludes_block_node_modules(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")
    _touch(tmp_path / "node_modules" / "react" / "index.js", "...")

    plan = analyze_package(
        tmp_path,
        entrypoint=tmp_path / "train.py",
        include=["**/*"],
    )
    inc = {p.relative_to(tmp_path).as_posix() for p in plan.explicit_includes}
    assert not any("node_modules" in name for name in inc)


def test_default_excludes_includes_uv_cache_and_node_modules() -> None:
    """Regression: keep these in the exclusion list."""
    assert ".uv-cache/" in DEFAULT_EXCLUDES
    assert "node_modules/" in DEFAULT_EXCLUDES
    assert ".cache/" in DEFAULT_EXCLUDES
    assert ".tox/" in DEFAULT_EXCLUDES


# ---------------------------------------------------------------------------
# Pathspec fallback — prunes subtrees (the user's reported bug)
# ---------------------------------------------------------------------------


def test_pathspec_fallback_prunes_nested_gitignored_dirs(tmp_path: Path) -> None:
    """`data/*` in .gitignore must prune `data/pii/*.jsonl` (not just direct children)."""
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / ".gitignore", "data/*\n")
    _touch(tmp_path / "src" / "main.py", "")
    _touch(tmp_path / "data" / "top_level.jsonl", "...")
    _touch(tmp_path / "data" / "pii" / "nested.jsonl", "..." * 1000)
    _touch(tmp_path / "data" / "pii" / "big.tgz", "..." * 1000)

    # No entrypoint -> pathspec fallback walks the tree with prune.
    plan = analyze_package(tmp_path)
    walked = {p.relative_to(tmp_path).as_posix() for p in plan.pathspec_walk}

    assert "pyproject.toml" in walked
    assert "src/main.py" in walked
    # Neither direct nor nested data files should ship.
    assert "data/top_level.jsonl" not in walked
    assert "data/pii/nested.jsonl" not in walked
    assert "data/pii/big.tgz" not in walked


def test_pathspec_fallback_excludes_uv_cache_even_without_gitignore(tmp_path: Path) -> None:
    """The user's second bug: .uv-cache/ ships even without .gitignore mentioning it."""
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "src" / "main.py", "")
    _touch(tmp_path / ".uv-cache" / "simple-v20" / "pypi" / "aiohttp.rkyv", "x" * 100)
    _touch(tmp_path / ".uv-cache" / "archive-v0" / "abc" / "file.so", "x" * 100)

    plan = analyze_package(tmp_path)
    walked = {p.relative_to(tmp_path).as_posix() for p in plan.pathspec_walk}
    assert "src/main.py" in walked
    assert not any(".uv-cache" in name for name in walked)


# ---------------------------------------------------------------------------
# Regression: the original bug-report scenario, end-to-end
# ---------------------------------------------------------------------------


def test_regression_user_bug_report_scenario(tmp_path: Path) -> None:
    """Exactly the layout from the bug report: data/* in .gitignore + .uv-cache/
    + 4 jsonl files declared as `include` + a 162 MB tarball + nested PII files.
    Must produce a small tarball with only the 4 declared jsonl + code.
    """
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / ".gitignore", "data/*\n.uv-cache/\n")
    _touch(tmp_path / "train.py", "from src.data import load\n")
    _touch(tmp_path / "src" / "__init__.py", "")
    _touch(tmp_path / "src" / "data.py", "")

    # The 4 jsonl files the user declared as bundle_paths.
    declared = [
        "data/pii/ai4privacy_pl.jsonl",
        "data/pii/pii_synth_full.jsonl",
        "data/pii/pii_synth_full_normalized.jsonl",
        "data/pii/nkjp.jsonl",
    ]
    for d in declared:
        _touch(tmp_path / d, "x" * 1000)

    # The accidental large junk: nested PII files NOT in bundle_paths.
    _touch(tmp_path / "data" / "pii" / "NKJP-PodkorpusMilionowy-1.0.tgz", "x" * 5000)
    _touch(tmp_path / "data" / "pii" / "pii_synth_full copy.jsonl", "x" * 5000)
    _touch(tmp_path / "data" / "pii" / "kpwr_ner_native_train.jsonl", "x" * 5000)

    # The .uv-cache/ blob.
    for i in range(10):
        _touch(tmp_path / ".uv-cache" / "simple-v20" / "pypi" / f"file{i}.rkyv", "x" * 1000)

    plan = analyze_package(
        tmp_path,
        entrypoint=tmp_path / "train.py",
        include=declared,
    )

    shipped = {p.relative_to(tmp_path).as_posix() for p in plan.all_files()}

    # The four declared jsonl files ship.
    for d in declared:
        assert d in shipped, f"declared file {d} should ship"
    # The accidental nested files do NOT ship.
    assert "data/pii/NKJP-PodkorpusMilionowy-1.0.tgz" not in shipped
    assert "data/pii/pii_synth_full copy.jsonl" not in shipped
    assert "data/pii/kpwr_ner_native_train.jsonl" not in shipped
    # .uv-cache is entirely absent.
    assert not any(".uv-cache" in name for name in shipped)
    # Code + manifest still there.
    assert "train.py" in shipped
    assert "src/data.py" in shipped
    assert "pyproject.toml" in shipped
    assert "uv.lock" in shipped


# ---------------------------------------------------------------------------
# make_tarball
# ---------------------------------------------------------------------------


def test_make_tarball_writes_expected_files(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.py")
    out = tmp_path / "out.tar.gz"
    res = make_tarball(tmp_path, out, plan)

    assert out.exists()
    assert res.file_count >= 3
    names = _arcnames(out)
    assert "train.py" in names
    assert "pyproject.toml" in names
    assert "uv.lock" in names


def test_make_tarball_reports_largest_files(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")
    _touch(tmp_path / "data" / "huge.bin", "x" * 100_000)
    _touch(tmp_path / "data" / "tiny.txt", "x")
    _touch(tmp_path / "data" / "mid.bin", "x" * 10_000)

    plan = analyze_package(
        tmp_path,
        entrypoint=tmp_path / "train.py",
        include=["data/"],
    )
    res = make_tarball(tmp_path, tmp_path / "out.tar.gz", plan, top_n_largest=3)
    assert len(res.largest) == 3
    # Ordered descending by size.
    sizes = [s for _, s in res.largest]
    assert sizes == sorted(sizes, reverse=True)
    assert res.largest[0][0] == "data/huge.bin"


def test_make_tarball_fires_progress_ticks(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")
    for i in range(250):  # > 200 threshold to force at least one tick
        _touch(tmp_path / "files" / f"f{i}.txt", "x" * 10)

    plan = analyze_package(
        tmp_path,
        entrypoint=tmp_path / "train.py",
        include=["files/"],
    )
    ticks: list[tuple[int, int]] = []
    make_tarball(tmp_path, tmp_path / "out.tar.gz", plan, on_tick=lambda c, b: ticks.append((c, b)))
    assert ticks, "expected at least one progress tick"
    # The final tick should report the full count.
    final_count, final_bytes = ticks[-1]
    assert final_count == plan.all_files().__len__()


def test_make_tarball_progress_tick_for_small_run(tmp_path: Path) -> None:
    """Even tiny runs should get a final tick so plain-mode shows totals."""
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.py", "")

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.py")
    ticks: list[tuple[int, int]] = []
    make_tarball(tmp_path, tmp_path / "out.tar.gz", plan, on_tick=lambda c, b: ticks.append((c, b)))
    assert ticks, "expected a final tick even for tiny runs"


# ---------------------------------------------------------------------------
# dedupe_preserve_order helper
# ---------------------------------------------------------------------------


def test_dedupe_preserves_first_seen_order() -> None:
    assert dedupe_preserve_order(["a", "b"], ["b", "c"], ["a", "d"]) == ["a", "b", "c", "d"]


def test_dedupe_handles_empty_inputs() -> None:
    assert dedupe_preserve_order() == []
    assert dedupe_preserve_order([]) == []
    assert dedupe_preserve_order([], ["x"], []) == ["x"]


def test_dedupe_accepts_arbitrary_iterables() -> None:
    """Sets, generators, tuples — anything Iterable[str] should work."""
    gen = (s for s in ["a", "b", "a"])
    assert dedupe_preserve_order(gen, ("c", "b")) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# PackagePlan.local_dataset_root
# ---------------------------------------------------------------------------


def test_local_dataset_root_returns_common_ancestor(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "data" / "shard0.parquet", "x")
    _touch(tmp_path / "data" / "shard1.parquet", "y")
    plan = analyze_package(tmp_path, include=["data/**/*"])
    root = plan.local_dataset_root()
    assert root is not None
    assert root == (tmp_path / "data").resolve()


def test_local_dataset_root_handles_nested(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "data" / "raw" / "a.jsonl", "x")
    _touch(tmp_path / "data" / "raw" / "b.jsonl", "y")
    plan = analyze_package(tmp_path, include=["data/raw/"])
    assert plan.local_dataset_root() == (tmp_path / "data" / "raw").resolve()


def test_local_dataset_root_none_when_files_span_root(tmp_path: Path) -> None:
    """Includes from two unrelated subdirs collapse to the project root → None."""
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "configs" / "a.yaml", "x")
    _touch(tmp_path / "data" / "b.jsonl", "y")
    plan = analyze_package(tmp_path, include=["configs/*.yaml", "data/*"])
    assert plan.local_dataset_root() is None


def test_local_dataset_root_none_when_no_includes(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    plan = analyze_package(tmp_path, include=[])
    assert plan.local_dataset_root() is None


def test_always_include_toplevel_constants() -> None:
    """Sanity check the manifest list itself."""
    assert "pyproject.toml" in ALWAYS_INCLUDE_TOPLEVEL
    assert "uv.lock" in ALWAYS_INCLUDE_TOPLEVEL
    assert "README*" in ALWAYS_INCLUDE_TOPLEVEL


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_analyze_rejects_non_dir_root(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        analyze_package(tmp_path / "missing")


def test_analyze_rejects_entrypoint_outside_root(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    other = tmp_path.parent / "outside.py"
    other.write_text("")
    try:
        with pytest.raises(ValueError):
            analyze_package(tmp_path, entrypoint=other)
    finally:
        other.unlink()


def test_analyze_rejects_missing_entrypoint(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml", "")
    with pytest.raises(FileNotFoundError):
        analyze_package(tmp_path, entrypoint=tmp_path / "nope.py")


def test_analyze_non_python_entrypoint_falls_back_to_pathspec(tmp_path: Path) -> None:
    """A non-.py entrypoint goes through the pathspec walk (CLI rejects this
    separately; the analysis itself stays functional for debugging)."""
    _touch(tmp_path / "pyproject.toml", "")
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "train.sh", "#!/bin/sh\n")

    plan = analyze_package(tmp_path, entrypoint=tmp_path / "train.sh")
    # No AST walk happened.
    assert plan.python_imports == []
    # The pathspec walk produced a non-empty result.
    assert plan.pathspec_walk
