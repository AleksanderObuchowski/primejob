"""Tests for the tightened error-line classifier in run.py."""
from __future__ import annotations

from primejob.run import _compile_error_patterns, _is_error_line


def test_real_errors_are_flagged() -> None:
    pat = _compile_error_patterns()
    assert _is_error_line("Traceback (most recent call last):", patterns=pat)
    assert _is_error_line("RuntimeError: cuda kernel failed", patterns=pat)
    assert _is_error_line("ValueError: bad shape", patterns=pat)
    assert _is_error_line("torch.cuda.OutOfMemoryError: out of memory", patterns=pat)
    assert _is_error_line("  AssertionError: tensor shape mismatch", patterns=pat)
    assert _is_error_line("Segmentation fault (core dumped)", patterns=pat)
    assert _is_error_line("FAILED tests/test_x.py::test_y", patterns=pat)
    # CUDA-style lowercase variants are still caught by the OOM/CUDA pattern
    assert _is_error_line("cuda error: device-side assert", patterns=pat)
    assert _is_error_line("RuntimeError: out of memory at allocation", patterns=pat)


def test_noisy_lines_are_not_flagged() -> None:
    pat = _compile_error_patterns()
    # transformers / urllib3 commonly log these in healthy runs:
    assert not _is_error_line(
        "WARN: deprecation Error in module X will be removed in v5", patterns=pat
    )
    assert not _is_error_line("Retrying after error: connection reset", patterns=pat)
    assert not _is_error_line(
        "INFO: handling exception-class case gracefully", patterns=pat
    )
    # urllib3 / huggingface_hub style — lowercase "error:" in prose must not fire
    assert not _is_error_line("urllib3.connection: error: retrying", patterns=pat)


def test_extra_user_patterns_are_added() -> None:
    pat = _compile_error_patterns(["NaN loss detected"])
    assert _is_error_line("epoch=3 NaN loss detected at step 42", patterns=pat)
