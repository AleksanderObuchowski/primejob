"""Log tailer resumes after an initial replay cookie."""

from __future__ import annotations

import time
from pathlib import Path

from primejob.tui.workers.log_tail import LogTailer


def test_log_tail_initial_seek_skips_replayed_lines(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    log.write_text("line-one\n")

    with log.open(encoding="utf-8", errors="replace") as fh:
        fh.read()
        seek_cookie = fh.tell()

    seen: list[str] = []

    def on_line(stream: str, payload: str) -> None:
        seen.append(payload)

    tail = LogTailer(
        log,
        interval=0.05,
        initial_text_seek=seek_cookie,
        on_line=on_line,
    )
    tail.start()
    time.sleep(0.06)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("line-two\n")

    for _ in range(50):
        if seen:
            break
        time.sleep(0.05)
    tail.stop()
    assert seen == ["line-two"]
