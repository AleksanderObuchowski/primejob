"""Smoke tests for the Textual dashboard.

These tests do NOT spin up real pods or SSH connections — they exercise the
widgets, modal flows, and the EventSink → UI marshalling in isolation."""
from __future__ import annotations

import pytest

from primejob.events import ConfirmRequest, ConsoleSink
from primejob.tui.state import (
    FinalSummary,
    GpuMetric,
    PHASE_ORDER,
    Phase,
    RunMeta,
    gpu_badge,
    script_label,
)
from primejob.tui.widgets.gpus import GpuTable, _util_bar
from primejob.tui.widgets.log import style_log_line
from primejob.tui.workers.nvidia import _decode_throttle, parse_nvidia_smi


# ---------------------------------------------------------------- state helpers


def test_script_label_truncates_long_args() -> None:
    meta = RunMeta(run_id="r", script="train.py", args=["--epochs", "100"] * 20)
    label = script_label(meta)
    assert label.endswith("...")
    assert len(label) <= 60


def test_gpu_badge_contains_count_and_location() -> None:
    meta = RunMeta(
        run_id="r", script="t.py", gpu_type="H100", gpu_count=2,
        country="US", provider="datacrunch",
    )
    badge = gpu_badge(meta)
    assert "H100×2" in badge
    assert "US" in badge
    assert "datacrunch" in badge


def test_gpu_badge_empty_when_no_type() -> None:
    meta = RunMeta(run_id="r", script="t.py", gpu_type="")
    assert gpu_badge(meta) == ""


# ---------------------------------------------------------------- log styling


def test_style_log_line_highlights_traceback() -> None:
    text = style_log_line("stderr", "Traceback (most recent call last):")
    # Rich's Text exposes spans via .spans; we just check it stylized.
    rendered = text.plain
    assert "Traceback" in rendered
    # Look for bold red in spans.
    found = any("red" in str(span.style) for span in text.spans)
    assert found


def test_style_log_line_highlights_warning() -> None:
    text = style_log_line("stdout", "DeprecationWarning: foo")
    found = any("yellow" in str(span.style) for span in text.spans)
    assert found


def test_style_log_line_plain_stdout_is_uncolored() -> None:
    text = style_log_line("stdout", "epoch 1/10 loss=0.42")
    plain = text.plain
    assert plain == "epoch 1/10 loss=0.42"


# ---------------------------------------------------------------- nvidia parser


def test_parse_nvidia_smi_basic() -> None:
    out = (
        "0, 87, 76000, 81920, 71, 412, 0\n"
        "1, 82, 71000, 81920, 69, 398, 0x8\n"
    )
    metrics = parse_nvidia_smi(out)
    assert len(metrics) == 2
    assert metrics[0].index == 0
    assert metrics[0].util_pct == 87.0
    assert metrics[0].mem_used_mb == 76000
    assert metrics[0].mem_total_mb == 81920
    assert metrics[0].temp_c == 71
    assert metrics[0].power_w == 412
    assert metrics[0].throttle == ""
    assert metrics[1].throttle == "HW slowdown"


def test_parse_nvidia_smi_handles_na_power() -> None:
    out = "0, 50, 1000, 10000, 60, [N/A], 0\n"
    metrics = parse_nvidia_smi(out)
    assert metrics[0].power_w == 0.0


def test_parse_nvidia_smi_skips_malformed() -> None:
    out = "not a row\n0, 50, 1000, 10000, 60, 100, 0\nshort,row\n"
    metrics = parse_nvidia_smi(out)
    assert len(metrics) == 1
    assert metrics[0].index == 0


def test_decode_throttle_no_active() -> None:
    assert _decode_throttle("0") == ""
    assert _decode_throttle("0x0") == ""
    assert _decode_throttle("Not Active") == ""


def test_decode_throttle_known_bits() -> None:
    # 0x4 = SW pwr cap, 0x40 = HW thermal
    assert "SW pwr cap" in _decode_throttle("0x44")
    assert "HW thermal" in _decode_throttle("0x44")


# ---------------------------------------------------------------- gpu widget helpers


def test_util_bar_width_matches() -> None:
    assert len(_util_bar(50.0, width=10)) == 10
    assert _util_bar(0.0, width=8) == "░" * 8
    assert _util_bar(100.0, width=8) == "█" * 8


def test_util_bar_clamps_out_of_range() -> None:
    assert _util_bar(-10.0, width=5) == "░" * 5
    assert _util_bar(200.0, width=5) == "█" * 5


# ---------------------------------------------------------------- phase ordering


def test_phase_order_complete() -> None:
    assert PHASE_ORDER[0] is Phase.PREFLIGHT
    assert PHASE_ORDER[-1] is Phase.WRAP
    assert len(PHASE_ORDER) == 6


# ---------------------------------------------------------------- ConsoleSink


def test_console_sink_yes_skips_prompt() -> None:
    sink = ConsoleSink(yes=True)
    request = ConfirmRequest(
        prompt="confirm?",
        gpu_type="H100", gpu_count=1, rate_per_hr=2.43,
        provider="datacrunch", country="US",
    )
    assert sink.confirm(request) is True


def test_console_sink_writes_log_file(tmp_path):
    log_path = tmp_path / "log.txt"
    sink = ConsoleSink(log_file=log_path, yes=True)
    sink.log_line("stdout", "hello")
    sink.log_line("stderr", "boom")
    sink.close()
    contents = log_path.read_text()
    assert "hello" in contents
    assert "[stderr] boom" in contents


# ---------------------------------------------------------------- Textual Pilot


def test_app_starts_and_quits_in_attach_mode(tmp_path, monkeypatch):
    """End-to-end: PrimejobApp boots in attach mode without spinning a worker."""
    import asyncio
    import importlib

    monkeypatch.setenv("HOME", str(tmp_path))
    from primejob import state as state_mod
    importlib.reload(state_mod)

    record = state_mod.RunRecord(
        run_id="20260519T120000-deadbe",
        pod_id=None,
        gpu_type="H100",
        gpu_count=1,
        country="US",
        provider="datacrunch",
        rate_per_hr=2.43,
        script="train.py",
        args=["--epochs", "1"],
        status="finished",
        exit_code=0,
        total_cost=0.0123,
    )
    record.save()
    record.log_path.write_text("epoch 1/1 loss=0.5\nDone.\n")

    from primejob.tui import app as app_mod
    monkeypatch.setattr(app_mod, "load_run", state_mod.load_run)

    async def go():
        pj_app = app_mod.PrimejobApp(record=record, attach=True)
        async with pj_app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()

    asyncio.run(go())


def test_app_preflight_modal_confirm_path(tmp_path, monkeypatch):
    """The preflight modal returns True on `y`."""
    import asyncio
    from primejob.tui.screens.preflight import PreflightModal
    from textual.app import App

    class Wrap(App):
        result = None

        def on_mount(self) -> None:
            def done(answer):
                self.result = answer
                self.exit()
            self.push_screen(
                PreflightModal(
                    gpu="H100", count=1, rate_per_hr=2.43,
                    provider="datacrunch", country="US",
                ),
                done,
            )

    async def go():
        app = Wrap()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
        assert app.result is True

    asyncio.run(go())


def test_app_preflight_modal_cancel_path():
    """The preflight modal returns False on `n`."""
    import asyncio
    from primejob.tui.screens.preflight import PreflightModal
    from textual.app import App

    class Wrap(App):
        result = "unset"

        def on_mount(self) -> None:
            def done(answer):
                self.result = answer
                self.exit()
            self.push_screen(
                PreflightModal(
                    gpu="H100", count=1, rate_per_hr=2.43,
                    provider=None, country=None,
                ),
                done,
            )

    async def go():
        app = Wrap()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
        assert app.result is False

    asyncio.run(go())


def test_app_help_overlay_opens_and_closes():
    """Pressing `?` opens HelpScreen, esc closes."""
    import asyncio
    from primejob.tui.app import PrimejobApp
    from primejob.state import RunRecord

    record = RunRecord(
        run_id="20260519T130000-aaaaaa",
        pod_id=None,
        gpu_type="H100",
        gpu_count=1,
        country="US",
        provider="datacrunch",
        rate_per_hr=2.43,
        script="train.py",
        args=[],
        status="finished",
        exit_code=0,
    )

    async def go():
        app = PrimejobApp(record=record, attach=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("question_mark")
            await pilot.pause()
            # HelpScreen should be on top of the stack.
            from primejob.tui.screens.help import HelpScreen
            assert any(isinstance(s, HelpScreen) for s in app.screen_stack)
            await pilot.press("escape")
            await pilot.pause()
            assert not any(isinstance(s, HelpScreen) for s in app.screen_stack)
            await pilot.press("q")
            await pilot.pause()

    asyncio.run(go())
