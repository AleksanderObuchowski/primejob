"""`primejob package` CLI subcommand tests.

Exercises the dry-run analysis output and the real tarball write path so the
inspection workflow stays usable as packaging.py and config.py evolve.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

from typer.testing import CliRunner

from primejob.cli import app

runner = CliRunner()


def _make_project(tmp_path: Path, *, pyproject: str = "") -> Path:
    """Create a minimal uv project at tmp_path and return its root."""
    (tmp_path / "pyproject.toml").write_text(
        pyproject
        or """
[project]
name = "demo"
"""
    )
    (tmp_path / "uv.lock").write_text("")
    return tmp_path


def test_package_dry_run_lists_ast_and_always_includes(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    (root / "src").mkdir()
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "data.py").write_text("import json\n")
    (root / "train.py").write_text("from src import data\n")

    result = runner.invoke(app, ["package", str(root / "train.py"), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Python imports (AST closure) (3)" in result.output
    assert "src/__init__.py" in result.output
    assert "src/data.py" in result.output
    assert "train.py" in result.output
    assert "Always included (2)" in result.output
    assert "pyproject.toml" in result.output
    assert "uv.lock" in result.output


def test_package_dry_run_does_not_write_tarball(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    (root / "train.py").write_text("x = 1\n")
    out = root / "out.tar.gz"

    result = runner.invoke(
        app, ["package", str(root / "train.py"), "--dry-run", "-o", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert not out.exists()


def test_package_writes_tarball_with_expected_contents(tmp_path: Path) -> None:
    root = _make_project(
        tmp_path,
        pyproject="""
[project]
name = "demo"

[tool.primejob]
include = ["data/keep.jsonl"]
""",
    )
    (root / "train.py").write_text("x = 1\n")
    (root / "data").mkdir()
    (root / "data" / "keep.jsonl").write_text("ok\n")
    (root / "data" / "pii").mkdir()
    (root / "data" / "pii" / "secret.jsonl").write_text("sshh\n")
    out = root / "primejob-package.tar.gz"

    result = runner.invoke(app, ["package", str(root / "train.py"), "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    with tarfile.open(out, "r:gz") as tar:
        members = sorted(m.name for m in tar.getmembers())
    assert members == ["data/keep.jsonl", "pyproject.toml", "train.py", "uv.lock"]


def test_package_flags_unresolved_dynamic_imports(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    (root / "train.py").write_text(
        "import importlib\n"
        "mod_name = input('module: ')\n"
        "importlib.import_module(mod_name)\n"
    )

    result = runner.invoke(app, ["package", str(root / "train.py"), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Unresolved dynamic imports" in result.output
    assert "importlib.import_module" in result.output
    assert "[tool.primejob].include" in result.output


def test_package_extra_include_merges_with_pyproject(tmp_path: Path) -> None:
    root = _make_project(
        tmp_path,
        pyproject="""
[project]
name = "demo"

[tool.primejob]
include = ["configs/base.yaml"]
""",
    )
    (root / "train.py").write_text("x = 1\n")
    (root / "configs").mkdir()
    (root / "configs" / "base.yaml").write_text("lr: 1e-3\n")
    (root / "configs" / "tune.yaml").write_text("lr: 5e-4\n")

    result = runner.invoke(
        app,
        [
            "package",
            str(root / "train.py"),
            "--dry-run",
            "-i",
            "configs/tune.yaml",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "configs/base.yaml" in result.output
    assert "configs/tune.yaml" in result.output


def test_package_warns_on_bundle_paths_deprecation(tmp_path: Path) -> None:
    root = _make_project(
        tmp_path,
        pyproject="""
[project]
name = "demo"

[tool.primejob]
bundle_paths = ["data/keep.jsonl"]
""",
    )
    (root / "train.py").write_text("x = 1\n")
    (root / "data").mkdir()
    (root / "data" / "keep.jsonl").write_text("ok\n")

    result = runner.invoke(app, ["package", str(root / "train.py"), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "deprecation" in result.output.lower()
    assert "bundle_paths" in result.output
    # The deprecated values still flow through to the plan.
    assert "data/keep.jsonl" in result.output


def test_package_rejects_non_python_script(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    (root / "train.sh").write_text("echo hi\n")

    result = runner.invoke(app, ["package", str(root / "train.sh"), "--dry-run"])

    assert result.exit_code == 1
    assert ".py" in result.output


def test_package_rejects_missing_script(tmp_path: Path) -> None:
    root = _make_project(tmp_path)

    result = runner.invoke(app, ["package", str(root / "missing.py"), "--dry-run"])

    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_package_errors_when_no_pyproject(tmp_path: Path) -> None:
    # No pyproject anywhere up the tree (the temp dir itself has nothing).
    (tmp_path / "train.py").write_text("x = 1\n")

    result = runner.invoke(app, ["package", str(tmp_path / "train.py"), "--dry-run"])

    assert result.exit_code == 1
    assert "pyproject" in result.output.lower()


def test_package_finds_pyproject_from_script_directory(tmp_path: Path) -> None:
    """`primejob package` should work even when cwd is unrelated to the script."""
    root = _make_project(tmp_path)
    (root / "train.py").write_text("x = 1\n")

    # Invoke the runner with the absolute path; we do not chdir into root.
    result = runner.invoke(
        app,
        ["package", str(root / "train.py"), "--dry-run"],
        # Avoid Rich line-wrapping breaking up paths in CI/narrow terminals.
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    assert "Root:" in result.output
    # The path may wrap across lines for long temp dirs, so check the basename.
    assert root.name in result.output
