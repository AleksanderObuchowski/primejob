"""Tests for pod deletion helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from primejob.backend import pods


def test_delete_pod_returns_false_on_sdk_error(monkeypatch) -> None:
    mock_client = object()

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pods, "PodsClient", lambda *_a: MagicMock(delete=boom))

    assert pods.delete_pod(mock_client, "pod-id") is False


def test_delete_pod_returns_true_on_success(monkeypatch) -> None:
    mock_client = object()

    class _Pods:
        def delete(self, pod_id: str) -> None:
            assert pod_id == "pod-id"

    monkeypatch.setattr(pods, "PodsClient", lambda *_a: _Pods())

    assert pods.delete_pod(mock_client, "pod-id") is True


def test_terminate_swallows_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        pods, "PodsClient", lambda *_a: MagicMock(delete=lambda *_: (_ for _ in ()).throw(RuntimeError())),
    )

    pods.terminate(object(), "whatever")  # should not raise
