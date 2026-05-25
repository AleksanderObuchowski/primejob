"""Unit tests for dataset push remote-path logic (no Prime API calls)."""

from __future__ import annotations

from pathlib import Path

from primejob.dataset import _remote_push_destination


def test_remote_push_destination_directory_default_name(tmp_path: Path) -> None:
    d = tmp_path / "mydata"
    d.mkdir()
    mk, dest = _remote_push_destination("/mnt/data", local_path=d, subdir=None)
    assert mk == "/mnt/data/mydata"
    assert dest == "/mnt/data/mydata"


def test_remote_push_destination_directory_custom_subdir(tmp_path: Path) -> None:
    d = tmp_path / "ignored"
    d.mkdir()
    mk, dest = _remote_push_destination(
        "/mnt/data", local_path=d, subdir="nested/dir"
    )
    assert mk == "/mnt/data/nested/dir"
    assert dest == "/mnt/data/nested/dir"


def test_remote_push_destination_file_default_basename(tmp_path: Path) -> None:
    f = tmp_path / "file.bin"
    f.write_bytes(b"x")
    mk, dest = _remote_push_destination("/mnt/disk", local_path=f, subdir=None)
    assert dest == "/mnt/disk/file.bin"
    assert mk == "/mnt/disk"


def test_remote_push_destination_file_with_explicit_subdir(tmp_path: Path) -> None:
    f = tmp_path / "local.csv"
    f.write_bytes(b"x")
    mk, dest = _remote_push_destination("/mnt/vol", local_path=f, subdir="staging/out.csv")
    assert dest == "/mnt/vol/staging/out.csv"
    assert mk == "/mnt/vol/staging"


def test_ssh_wait_invoked_when_spawning_dataset_helper_pod(monkeypatch) -> None:
    """Smoke-test that helpers don't skip SSH propagation waiting."""
    from primejob import dataset as dataset_mod

    calls: list[object] = []

    class _Sentinel:
        pass

    sentinel = _Sentinel()

    def fake_wait(client, ssh, *, provider=None):
        calls.append((ssh.host, provider))
        return sentinel

    monkeypatch.setattr(dataset_mod, "_wait_ssh_dataset_helper", fake_wait)

    class _Opt:
        provider = "testprovider"

    opt = _Opt()

    class _Fresh:
        attached_resources = []

    class _Status:
        status = "running"
        ssh_connection = "ubuntu@example.com"

    created: dict[str, object] = {}

    def fake_pick(*_a, **_k):
        return opt

    def fake_cpu_spec(**_kwargs):
        return object()

    def fake_create_pod(_client, _spec):
        class _Pod:
            id = "pod-xyz"

        return _Pod()

    def fake_wait_running(_client, pid, **_k):
        return _Status()

    def fake_get_pod(_client, _pid):
        return _Fresh()

    def fake_mount(_pod, _disk_id):
        return "/mnt/persist"

    def fake_parse_ssh(_raw):
        class _SSH:
            host = "example.com"
            port = 22
            user = "ubuntu"
            key_path = None

        return _SSH()

    monkeypatch.setattr(dataset_mod, "pick_cheapest", fake_pick)
    monkeypatch.setattr(dataset_mod, "PodSpec", fake_cpu_spec)
    monkeypatch.setattr(dataset_mod, "create_pod", fake_create_pod)
    monkeypatch.setattr(dataset_mod, "wait_for_running", fake_wait_running)
    monkeypatch.setattr(dataset_mod, "get_pod", fake_get_pod)
    monkeypatch.setattr(dataset_mod, "mount_path_for_disk", fake_mount)
    monkeypatch.setattr(dataset_mod, "parse_ssh_endpoint", fake_parse_ssh)
    monkeypatch.setattr(dataset_mod, "terminate", lambda *_a, **_k: created.setdefault("stopped", True))

    class DummyClient:
        pass

    pid, ssh, mount, conn = dataset_mod._spawn_helper_pod(
        DummyClient(),
        name="job",
        country="US",
        disk_id="disk-1",
    )
    assert pid == "pod-xyz"
    assert mount == "/mnt/persist"
    assert conn is sentinel
    assert calls == [("example.com", "testprovider")]
    assert hasattr(ssh, "host")

