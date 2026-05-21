"""Pod create payload includes explicit sshKeyId when registered."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from primejob.auth import resolve_pod_ssh_key_id
from primejob.backend.pods import PodSpec, create_pod
from primejob.pricing import GpuOption


def _gpu_option() -> GpuOption:
    return GpuOption(
        gpu_type="L40S_48GB",
        gpu_count=1,
        country="US",
        data_center="dc1",
        cloud_id="cloud",
        provider="massedcompute",
        socket="PCIe",
        security="secure_cloud",
        gpu_memory=48,
        vcpu_default=8,
        memory_default=64,
        disk_default=120,
        disk_min=80,
        disk_max=500,
        images=["ubuntu_22_cuda_12"],
        on_demand_price=1.0,
        community_price=None,
        currency="USD",
        stock_status="Available",
        is_spot=False,
    )


def test_pod_spec_payload_includes_ssh_key_id() -> None:
    spec = PodSpec(name="test-pod", gpu_option=_gpu_option(), ssh_key_id="key-abc-123")
    payload = spec.to_create_payload()
    assert payload["pod"]["sshKeyId"] == "key-abc-123"


def test_pod_spec_payload_omits_ssh_key_id_when_unset() -> None:
    spec = PodSpec(name="test-pod", gpu_option=_gpu_option())
    payload = spec.to_create_payload()
    assert "sshKeyId" not in payload["pod"]


def test_create_pod_resolves_ssh_key_id() -> None:
    client = MagicMock()
    spec = PodSpec(name="test-pod", gpu_option=_gpu_option())
    created = MagicMock()
    created.id = "pod-1"

    with (
        patch(
            "primejob.auth.resolve_pod_ssh_key_id",
            return_value="resolved-key-id",
        ) as mock_resolve,
        patch("primejob.backend.pods.PodsClient") as mock_pods_cls,
    ):
        mock_pods_cls.return_value.create.return_value = created
        pod = create_pod(client, spec)

    assert pod.id == "pod-1"
    mock_resolve.assert_called_once_with(client)
    call_payload = mock_pods_cls.return_value.create.call_args[0][0]
    assert call_payload["pod"]["sshKeyId"] == "resolved-key-id"


def test_create_pod_skips_resolve_when_ssh_key_id_set() -> None:
    client = MagicMock()
    spec = PodSpec(name="test-pod", gpu_option=_gpu_option(), ssh_key_id="preset-key")
    created = MagicMock()
    created.id = "pod-1"

    with (
        patch("primejob.auth.resolve_pod_ssh_key_id") as mock_resolve,
        patch("primejob.backend.pods.PodsClient") as mock_pods_cls,
    ):
        mock_pods_cls.return_value.create.return_value = created
        pod = create_pod(client, spec)

    assert pod.id == "pod-1"
    mock_resolve.assert_not_called()
    call_payload = mock_pods_cls.return_value.create.call_args[0][0]
    assert call_payload["pod"]["sshKeyId"] == "preset-key"


def test_resolve_pod_ssh_key_id_returns_matched_id() -> None:
    client = MagicMock()
    with patch(
        "primejob.auth.check_ssh_key",
        return_value=MagicMock(registered=True, matched_key_id="k99"),
    ):
        assert resolve_pod_ssh_key_id(client) == "k99"


def test_resolve_pod_ssh_key_id_none_when_unregistered() -> None:
    client = MagicMock()
    with patch(
        "primejob.auth.check_ssh_key",
        return_value=MagicMock(registered=False, matched_key_id=None),
    ):
        assert resolve_pod_ssh_key_id(client) is None
