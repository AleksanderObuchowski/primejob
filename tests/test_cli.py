from __future__ import annotations

from typer.testing import CliRunner

from primejob.cli import app
from primejob.run import RunResult
from primejob.state import RunRecord


runner = CliRunner()


def test_run_include_data_emits_deprecation_warning(monkeypatch) -> None:
    captured = {}

    def fake_run_training(client, opts):
        captured["opts"] = opts
        return RunResult(record=RunRecord(
            run_id="r1",
            pod_id=None,
            gpu_type="H100_80GB",
            gpu_count=1,
            country="US",
            provider="test",
            rate_per_hr=1.0,
            script="train.py",
            exit_code=0,
        ))

    monkeypatch.setattr("primejob.cli.get_client", lambda: object())
    monkeypatch.setattr("primejob.run.run_training", fake_run_training)

    result = runner.invoke(
        app,
        ["run", "--plain", "--include-data", "data", "train.py"],
    )

    assert result.exit_code == 0
    assert "--include-data is deprecated" in result.output
    assert captured["opts"].include_data == ["data"]


def test_run_include_does_not_emit_deprecation_warning(monkeypatch) -> None:
    captured = {}

    def fake_run_training(client, opts):
        captured["opts"] = opts
        return RunResult(record=RunRecord(
            run_id="r1",
            pod_id=None,
            gpu_type="H100_80GB",
            gpu_count=1,
            country="US",
            provider="test",
            rate_per_hr=1.0,
            script="train.py",
            exit_code=0,
        ))

    monkeypatch.setattr("primejob.cli.get_client", lambda: object())
    monkeypatch.setattr("primejob.run.run_training", fake_run_training)

    result = runner.invoke(
        app,
        ["run", "--plain", "--include", "data", "train.py"],
    )

    assert result.exit_code == 0
    assert "deprecated" not in result.output
    assert captured["opts"].include == ["data"]
