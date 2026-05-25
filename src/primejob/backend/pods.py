"""Pod lifecycle wrapping prime_cli.api.pods."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from prime_cli.api.client import APIClient
from prime_cli.api.pods import Pod, PodConfig, PodsClient, PodStatus

from primejob.pricing import GpuOption


RUNNING_STATES = {"running", "active", "ready"}
FAILED_STATES = {"failed", "error", "terminated"}


@dataclass
class TerminateResult:
    """Outcome of a pod terminate API call."""

    success: bool
    error: str | None = None


@dataclass
class PodSpec:
    name: str
    gpu_option: GpuOption
    image: str | None = None
    disk_size_gb: int | None = None
    disk_ids: list[str] | None = None
    env_vars: dict[str, str] | None = None
    ssh_key_id: str | None = None
    extra: dict | None = None

    def to_create_payload(self, *, ssh_key_id: str | None = None) -> dict:
        """Build the POST /pods payload.

        Shape mirrors what the `prime` CLI sends: top-level dict with `pod`
        (the inner config), `provider`, `disks`, `team`. The PodConfig pydantic
        model in prime_cli.api.pods describes the inner `pod` block.
        """
        o = self.gpu_option
        image = self.image or (o.images[0] if o.images else None)
        if image is None:
            raise ValueError(f"No image available for {o.gpu_type} @ {o.provider}")
        disk_size = self.disk_size_gb or o.disk_default
        if o.disk_min and disk_size and disk_size < o.disk_min:
            disk_size = o.disk_min

        inner = {
            "name": self.name,
            "cloudId": o.cloud_id,
            "gpuType": o.gpu_type,
            "socket": o.socket,
            "gpuCount": o.gpu_count,
            "diskSize": disk_size,
            "vcpus": o.vcpu_default,
            "memory": o.memory_default,
            "image": image,
            "dataCenterId": o.data_center,
            "maxPrice": None,
            "country": None,
            "security": None,
            "jupyterPassword": None,
            "autoRestart": False,
            "customTemplateId": None,
            "envVars": [{"key": k, "value": v} for k, v in (self.env_vars or {}).items()],
        }
        key_id = ssh_key_id if ssh_key_id is not None else self.ssh_key_id
        if key_id:
            inner["sshKeyId"] = key_id

        payload: dict = {
            "pod": inner,
            "provider": {"type": o.provider} if o.provider else {},
            "disks": list(self.disk_ids) if self.disk_ids else [],
        }
        # team is auto-populated by PodsClient.create() from the user's
        # default team config — we leave it for the SDK to fill in.
        if self.extra:
            payload.update(self.extra)
        return payload


def create_pod(client: APIClient, spec: PodSpec) -> Pod:
    """Create a pod; inject ``sshKeyId`` when the local key is registered in Prime."""
    ssh_key_id = spec.ssh_key_id
    if ssh_key_id is None:
        from primejob.auth import resolve_pod_ssh_key_id

        ssh_key_id = resolve_pod_ssh_key_id(client)
    payload = spec.to_create_payload(ssh_key_id=ssh_key_id)
    return PodsClient(client).create(payload)


def get_pod(client: APIClient, pod_id: str) -> Pod:
    return PodsClient(client).get(pod_id)


def get_status(client: APIClient, pod_id: str) -> PodStatus:
    statuses = PodsClient(client).get_status([pod_id])
    if not statuses:
        raise RuntimeError(f"No status returned for pod {pod_id}")
    return statuses[0]


def wait_for_running(
    client: APIClient,
    pod_id: str,
    *,
    timeout: int = 900,
    poll_interval: float = 5.0,
    on_progress: Callable[[PodStatus], None] | None = None,
) -> PodStatus:
    """Poll get_status until pod is RUNNING and has an ssh_connection."""
    start = time.monotonic()
    last: PodStatus | None = None
    while True:
        status = get_status(client, pod_id)
        last = status
        if on_progress:
            on_progress(status)
        state = (status.status or "").lower()
        if state in FAILED_STATES:
            raise RuntimeError(
                f"Pod {pod_id} entered terminal state '{state}': "
                f"{status.installation_failure or '(no detail)'}"
            )
        install_raw = getattr(status, "installation_progress", None)
        install_ready = True
        if install_raw is not None:
            try:
                install_ready = int(install_raw) >= 100
            except (TypeError, ValueError):
                install_ready = True
        if state in RUNNING_STATES and status.ssh_connection and install_ready:
            return status
        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"Pod {pod_id} not running after {timeout}s "
                f"(last status='{state}', progress={status.installation_progress})"
            )
        time.sleep(poll_interval)


def terminate_pod(client: APIClient, pod_id: str) -> TerminateResult:
    """Terminate a pod and return whether the API call succeeded."""
    try:
        PodsClient(client).delete(pod_id)
        return TerminateResult(success=True)
    except Exception as e:  # noqa: BLE001
        return TerminateResult(success=False, error=str(e))


def delete_pod(client: APIClient, pod_id: str) -> bool:
    """Request pod deletion via the API.

    Returns True if the SDK delete call succeeded, False otherwise
    (including when the pod is already gone depending on SDK behavior).

    Prefer :func:`terminate` for automated cleanup paths that must not leak
    diagnostics; use :func:`delete_pod` when the CLI needs to react to failures.
    """
    return terminate_pod(client, pod_id).success


def terminate(client: APIClient, pod_id: str) -> None:
    """Best-effort terminate — swallow errors (pod may already be gone)."""
    delete_pod(client, pod_id)


def mount_path_for_disk(pod: Pod, disk_id: str) -> str | None:
    """Find where a given disk is mounted on a running pod."""
    for r in pod.attached_resources or []:
        if r.id == disk_id and r.mount_path:
            return r.mount_path
    return None
