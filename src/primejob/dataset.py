"""Dataset operations on a persistent disk.

Both push and list need a running pod with the disk attached. We spin up the
cheapest CPU_NODE in the disk's region, do the SFTP/exec, then terminate.
"""
from __future__ import annotations

import math
import posixpath
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import paramiko
from prime_cli.api.client import APIClient

from primejob.auth import check_ssh_key, ssh_auth_failure_hint
from primejob.backend.disks import disk_location, ensure_disk, find_disk
from primejob.backend.pods import (
    PodSpec,
    create_pod,
    get_pod,
    mount_path_for_disk,
    terminate,
    wait_for_running,
)
from primejob.backend.ssh import (
    SSH_POST_READY_SLEEP_S,
    SshClient,
    SshEndpoint,
    parse_ssh_endpoint,
    wait_for_ssh_connect,
)
from primejob.config import load_project_config
from primejob.pricing import pick_cheapest


HELPER_DISK_MOUNT_FALLBACK = "/mnt/persistent"


@dataclass
class DatasetPushResult:
    disk_id: str
    disk_name: str
    pod_id: str
    files_uploaded: int
    bytes_uploaded: int
    elapsed_s: float


@dataclass
class DatasetPullResult:
    disk_id: str
    disk_name: str
    pod_id: str
    local_path: Path
    files_downloaded: int
    bytes_downloaded: int
    elapsed_s: float


def _local_size_bytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _local_file_count(p: Path) -> int:
    if p.is_file():
        return 1
    return sum(1 for f in p.rglob("*") if f.is_file())


def _auto_disk_size_gb(local_path: Path, min_gb: int = 50) -> int:
    bytes_used = _local_size_bytes(local_path)
    # 50% headroom, round up to nearest 10GB.
    gb_needed = math.ceil(bytes_used * 1.5 / (1024 ** 3) / 10) * 10
    return max(min_gb, gb_needed)


def _wait_ssh_dataset_helper(
    client: APIClient,
    ssh: SshEndpoint,
    *,
    provider: str | None,
) -> paramiko.SSHClient:
    """Apply the same post-ready pause + auth warmup as ``run_training``.

    Dataset helper pods reuse this path so intermittent provider key propagation
    does not regress ``primejob dataset *`` versus ``primejob run``.
    """
    project = load_project_config()
    if SSH_POST_READY_SLEEP_S > 0:
        time.sleep(SSH_POST_READY_SLEEP_S)

    pod_ready_mono = time.monotonic()
    ssh_status = check_ssh_key(client)
    auth_hint = ssh_auth_failure_hint(ssh_status, provider=provider)

    auth_window = min(float(project.ssh_max_wait), 300.0)
    return wait_for_ssh_connect(
        ssh,
        max_wait_s=float(project.ssh_max_wait),
        retry_delay_s=float(project.ssh_retry_delay),
        pod_ready_monotonic=pod_ready_mono,
        auth_warmup_s=auth_window,
        auth_failure_hint=auth_hint,
    )


def _remote_push_destination(
    mount: str,
    *,
    local_path: Path,
    subdir: str | None,
) -> tuple[str | None, str]:
    """Return ``(mkdir_p_parent_or_None, upload_destination_path)``.

    Directory uploads recreate the subtree under ``mount/<name>`` where ``name`` is
    ``subdir`` when given (relative path under ``mount``), otherwise the basename
    of ``local_path``.

    File uploads resolve to a concrete remote file path; the returned parent is
    the directory that must exist before uploading.
    """
    root = mount.rstrip("/")

    if local_path.is_dir():
        if subdir is not None:
            leaf = posixpath.normpath(subdir.replace("\\", "/")).lstrip("/")
            if not leaf:
                leaf = local_path.name
        else:
            leaf = local_path.name
        remote_dir = posixpath.join(root, leaf)
        return remote_dir, remote_dir

    if subdir is not None:
        rel = posixpath.normpath(subdir.replace("\\", "/")).lstrip("/")
        remote_dest = posixpath.join(root, rel)
    else:
        remote_dest = posixpath.join(root, local_path.name)

    parent = posixpath.dirname(remote_dest)
    mkdir_for: str | None
    if not parent or parent == ".":
        mkdir_for = None
    else:
        mkdir_for = parent
    return mkdir_for, remote_dest


def _spawn_helper_pod(
    client: APIClient,
    *,
    name: str,
    country: str | None,
    disk_id: str,
    on_progress: Callable | None = None,
) -> tuple[str, SshEndpoint, str, paramiko.SSHClient]:
    """Create CPU_NODE pod with disk attached.

    Returns pod_id, SSH endpoint, mount path, and a connected paramiko client
    for passing to ``SshClient(..., prec_connected=...)``.
    """
    opt = pick_cheapest(client, gpu_type="CPU_NODE", gpu_count=1, country=country, disks=[disk_id])
    spec = PodSpec(
        name=name,
        gpu_option=opt,
        disk_ids=[disk_id],
    )
    pod = create_pod(client, spec)
    try:
        status = wait_for_running(client, pod.id, on_progress=on_progress)
        fresh = get_pod(client, pod.id)
        mount_path = mount_path_for_disk(fresh, disk_id) or HELPER_DISK_MOUNT_FALLBACK
        ssh = parse_ssh_endpoint(status.ssh_connection)
        connected = _wait_ssh_dataset_helper(
            client, ssh, provider=opt.provider,
        )
        return pod.id, ssh, mount_path, connected
    except Exception:
        terminate(client, pod.id)
        raise


def push(
    client: APIClient,
    *,
    disk_name: str,
    local_path: Path,
    disk_size_gb: int | None = None,
    country: str | None = None,
    subdir: str | None = None,
    on_progress: Callable | None = None,
) -> DatasetPushResult:
    """Upload local_path into a persistent disk, creating the disk if missing.

    subdir: optional path on the disk to upload into (default: local_path.name).
    """
    if not local_path.exists():
        raise FileNotFoundError(local_path)

    size = disk_size_gb or _auto_disk_size_gb(local_path)
    disk = ensure_disk(client, name=disk_name, size_gb=size, country=country, wait=True)

    # If disk already had a country (existing disk), prefer that for helper pod
    disk_country, _, _ = disk_location(disk)
    helper_country = disk_country or country

    pod_id, ssh, mount, connected = _spawn_helper_pod(
        client,
        name=f"primejob-helper-{int(time.time())}",
        country=helper_country,
        disk_id=disk.id,
        on_progress=on_progress,
    )
    mkdir_for, upload_dest = _remote_push_destination(
        mount, local_path=local_path, subdir=subdir
    )

    files = _local_file_count(local_path)
    bytes_total = _local_size_bytes(local_path)
    started = time.monotonic()
    try:
        with SshClient(ssh, prec_connected=connected) as sh:
            if local_path.is_dir():
                sh.exec(f"mkdir -p {shlex.quote(upload_dest)}").check()
            elif mkdir_for:
                sh.exec(f"mkdir -p {shlex.quote(mkdir_for)}").check()
            sh.upload(local_path, upload_dest)
    finally:
        terminate(client, pod_id)

    return DatasetPushResult(
        disk_id=disk.id,
        disk_name=disk_name,
        pod_id=pod_id,
        files_uploaded=files,
        bytes_uploaded=bytes_total,
        elapsed_s=time.monotonic() - started,
    )


def list_files(
    client: APIClient,
    *,
    disk_name: str,
    country: str | None = None,
    on_progress: Callable | None = None,
) -> list[str]:
    """List all files on the persistent disk by SSHing into an ephemeral pod."""
    disk = find_disk(client, disk_name)
    if disk is None:
        raise FileNotFoundError(f"No persistent disk named '{disk_name}'")
    disk_country, _, _ = disk_location(disk)
    helper_country = disk_country or country

    pod_id, ssh, mount, connected = _spawn_helper_pod(
        client,
        name=f"primejob-helper-{int(time.time())}",
        country=helper_country,
        disk_id=disk.id,
        on_progress=on_progress,
    )
    try:
        with SshClient(ssh, prec_connected=connected) as sh:
            result = sh.exec(
                f"find {shlex.quote(mount)} -type f -printf '%P\\n' 2>/dev/null"
            )
            if result.exit_code != 0 and not result.stdout.strip():
                result.check()
            return [line for line in result.stdout.splitlines() if line.strip()]
    finally:
        terminate(client, pod_id)


def pull(
    client: APIClient,
    *,
    disk_name: str,
    local_path: Path,
    country: str | None = None,
    subdir: str | None = None,
    on_progress: Callable | None = None,
) -> DatasetPullResult:
    """Download a disk dataset to local_path via a short-lived helper pod.

    If subdir is None, the whole disk mount is downloaded. This is intended for
    staging/copy mode; for large datasets users should pass a narrow subdir.
    """
    disk = find_disk(client, disk_name)
    if disk is None:
        raise FileNotFoundError(f"No persistent disk named '{disk_name}'")
    disk_country, _, _ = disk_location(disk)
    helper_country = disk_country or country

    pod_id, ssh, mount, connected = _spawn_helper_pod(
        client,
        name=f"primejob-helper-{int(time.time())}",
        country=helper_country,
        disk_id=disk.id,
        on_progress=on_progress,
    )
    if subdir is not None:
        rel = posixpath.normpath(subdir.replace("\\", "/")).lstrip("/")
        remote_root = posixpath.join(mount.rstrip("/"), rel) if rel else mount
    else:
        remote_root = mount

    started = time.monotonic()
    try:
        with SshClient(ssh, prec_connected=connected) as sh:
            sh.download(remote_root, local_path, ignore_permission_denied=True)
    finally:
        terminate(client, pod_id)

    return DatasetPullResult(
        disk_id=disk.id,
        disk_name=disk_name,
        pod_id=pod_id,
        local_path=local_path,
        files_downloaded=_local_file_count(local_path) if local_path.exists() else 0,
        bytes_downloaded=_local_size_bytes(local_path) if local_path.exists() else 0,
        elapsed_s=time.monotonic() - started,
    )
