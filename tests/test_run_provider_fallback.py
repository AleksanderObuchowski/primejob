from __future__ import annotations

from types import SimpleNamespace

from primejob.backend.ssh import ExecResult, SshAuthPropagationTimeout, SshEndpoint
from primejob.backend.pods import TerminateResult
from primejob.config import ProjectConfig
from primejob.pricing import GpuOption
from primejob.run import RunOptions, run_training


class RecordingSink:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def status(self, msg: str) -> None:
        self.messages.append(msg)

    def status_note(self, note: str) -> None:
        self.messages.append(note)

    def log_line(self, stream: str, line: str) -> None:
        pass

    def phase(self, phase, *, failed: bool = False) -> None:
        pass

    def meta(self, meta) -> None:
        pass

    def cost(self, *, started_at: float, rate_per_hr: float, spent: float) -> None:
        pass

    def ssh_ready(self, endpoint) -> None:
        pass

    def confirm(self, request) -> bool:
        return True

    def finish(self, summary) -> None:
        pass


class FakeSshClient:
    def __init__(self, endpoint, *, prec_connected=None) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass

    def upload(self, local_path, remote_path, **kwargs) -> None:
        pass

    def exec(self, cmd: str) -> ExecResult:
        return ExecResult(exit_code=0, stdout="", stderr="")

    def exec_stream(self, cmd: str, *, env=None, on_line=None) -> int:
        return 0

    def download(self, *args, **kwargs) -> None:
        pass


def _gpu(provider: str, price: float) -> GpuOption:
    return GpuOption(
        gpu_type="H100_80GB",
        gpu_count=1,
        country="US",
        data_center=f"{provider}-dc",
        cloud_id=f"{provider}-cloud",
        provider=provider,
        socket=None,
        security=None,
        gpu_memory=80,
        vcpu_default=16,
        memory_default=120,
        disk_default=100,
        disk_min=None,
        disk_max=None,
        images=["ubuntu"],
        on_demand_price=price,
        community_price=None,
        currency="USD",
        stock_status="available",
        is_spot=False,
    )


def test_run_training_falls_back_on_auth_propagation_timeout(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "train.py").write_text("print('ok')\n")
    monkeypatch.setattr("primejob.state.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("primejob.run.require_ssh_key", lambda client: None)
    monkeypatch.setattr("primejob.run.check_ssh_key", lambda client: SimpleNamespace())
    monkeypatch.setattr("primejob.run.ssh_auth_failure_hint", lambda status, provider: "hint")
    monkeypatch.setattr("primejob.run.SSH_POST_READY_SLEEP_S", 0.0)
    monkeypatch.setattr("primejob.run.SshClient", FakeSshClient)
    monkeypatch.setattr("primejob.run.new_run_id", lambda: "run1")
    monkeypatch.setattr("primejob.run.create_lease", lambda run_id, pod_id: None)
    monkeypatch.setattr("primejob.run.start_watchdog", lambda run_id, pod_id: None)
    monkeypatch.setattr("primejob.run.release_lease", lambda run_id: None)
    monkeypatch.setattr("primejob.run.heartbeat", lambda run_id: None)

    options = [_gpu("massedcompute", 1.0), _gpu("nebius", 1.2)]
    pick_excludes: list[list[str]] = []

    def fake_pick_cheapest(*args, exclude_providers=None, **kwargs):
        pick_excludes.append(list(exclude_providers or []))
        return options[len(pick_excludes) - 1]

    created: list[str] = []
    terminated: list[str] = []

    monkeypatch.setattr("primejob.run.pick_cheapest", fake_pick_cheapest)
    monkeypatch.setattr(
        "primejob.run.create_pod",
        lambda client, spec: SimpleNamespace(id=f"pod-{spec.gpu_option.provider}"),
    )
    monkeypatch.setattr(
        "primejob.run.wait_for_running",
        lambda client, pod_id, on_progress: SimpleNamespace(
            ssh_connection=f"root@{pod_id}.example:22"
        ),
    )
    monkeypatch.setattr(
        "primejob.run.get_pod",
        lambda client, pod_id: SimpleNamespace(status="running", attached_resources=[]),
    )
    monkeypatch.setattr(
        "primejob.run.parse_ssh_endpoint",
        lambda raw: SshEndpoint(host=str(raw), port=22, user="root"),
    )

    def fake_create(client, spec):
        pod_id = f"pod-{spec.gpu_option.provider}"
        created.append(pod_id)
        return SimpleNamespace(id=pod_id)

    monkeypatch.setattr("primejob.run.create_pod", fake_create)
    def fake_terminate_pod(client, pod_id):
        terminated.append(pod_id)
        return TerminateResult(success=True)

    monkeypatch.setattr("primejob.run.terminate_pod", fake_terminate_pod)

    def fake_terminate_run_pod(client, record, **kwargs):
        if record.pod_id:
            terminated.append(record.pod_id)

    monkeypatch.setattr("primejob.run.terminate_run_pod", fake_terminate_run_pod)
    connect_calls = 0

    def fake_wait_for_ssh_connect(*args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == 1:
            raise SshAuthPropagationTimeout("stalled")
        return object()

    monkeypatch.setattr("primejob.run.wait_for_ssh_connect", fake_wait_for_ssh_connect)

    sink = RecordingSink()
    result = run_training(
        object(),
        RunOptions(script="train.py", yes=True, data_mode="none", no_download=True),
        project=ProjectConfig(ssh_auth_timeout=90),
        cwd=tmp_path,
        sink=sink,
    )

    assert result.record.provider == "nebius"
    assert created == ["pod-massedcompute", "pod-nebius"]
    assert terminated == ["pod-massedcompute", "pod-nebius"]
    assert pick_excludes == [[], ["massedcompute"]]
    assert any("previous=massedcompute" in msg for msg in sink.messages)
    assert any("next=nebius" in msg for msg in sink.messages)


def test_run_training_does_not_fallback_when_terminate_fails(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "train.py").write_text("print('ok')\n")
    monkeypatch.setattr("primejob.state.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("primejob.run.require_ssh_key", lambda client: None)
    monkeypatch.setattr("primejob.run.check_ssh_key", lambda client: SimpleNamespace())
    monkeypatch.setattr("primejob.run.ssh_auth_failure_hint", lambda status, provider: "hint")
    monkeypatch.setattr("primejob.run.SSH_POST_READY_SLEEP_S", 0.0)
    monkeypatch.setattr("primejob.run.new_run_id", lambda: "run1")
    monkeypatch.setattr("primejob.run.create_lease", lambda run_id, pod_id: None)
    monkeypatch.setattr("primejob.run.start_watchdog", lambda run_id, pod_id: None)
    monkeypatch.setattr("primejob.run.release_lease", lambda run_id: None)
    monkeypatch.setattr("primejob.run.heartbeat", lambda run_id: None)

    pick_calls = 0

    def fake_pick_cheapest(*args, **kwargs):
        nonlocal pick_calls
        pick_calls += 1
        return _gpu("massedcompute", 1.0)

    monkeypatch.setattr("primejob.run.pick_cheapest", fake_pick_cheapest)
    monkeypatch.setattr(
        "primejob.run.create_pod",
        lambda client, spec: SimpleNamespace(id=f"pod-{spec.gpu_option.provider}"),
    )
    monkeypatch.setattr(
        "primejob.run.wait_for_running",
        lambda client, pod_id, on_progress: SimpleNamespace(
            ssh_connection=f"root@{pod_id}.example:22"
        ),
    )
    monkeypatch.setattr(
        "primejob.run.get_pod",
        lambda client, pod_id: SimpleNamespace(status="running", attached_resources=[]),
    )
    monkeypatch.setattr(
        "primejob.run.parse_ssh_endpoint",
        lambda raw: SshEndpoint(host=str(raw), port=22, user="root"),
    )
    monkeypatch.setattr(
        "primejob.run.terminate_pod",
        lambda client, pod_id: TerminateResult(success=False, error="api down"),
    )
    monkeypatch.setattr(
        "primejob.run.wait_for_ssh_connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SshAuthPropagationTimeout("stalled")
        ),
    )

    sink = RecordingSink()
    try:
        run_training(
            object(),
            RunOptions(script="train.py", yes=True, data_mode="none", no_download=True),
            project=ProjectConfig(ssh_auth_timeout=90),
            cwd=tmp_path,
            sink=sink,
        )
    except RuntimeError as e:
        assert "could not be terminated" in str(e)
    else:
        raise AssertionError("expected provider fallback to stop on terminate failure")

    assert pick_calls == 1
    assert any("terminate failed: api down" in msg for msg in sink.messages)
