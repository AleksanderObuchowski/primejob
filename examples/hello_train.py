"""Tiny smoke test for `primejob run`.

Writes a hello line, dumps nvidia-smi if present, and saves a result file
under outputs/ — primejob will download that back locally.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    print(f"hello from primejob — python {sys.version.split()[0]}")
    print(f"args = {sys.argv[1:]}")
    print(f"dataset path = {os.environ.get('PRIMEJOB_DATASET_PATH', '(none)')}")

    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        print("nvidia-smi -L:\n" + (out.stdout or out.stderr))
    except FileNotFoundError:
        print("nvidia-smi not present (CPU-only pod)")

    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    (outputs / "result.txt").write_text(
        f"completed at {time.time()}\nargs={sys.argv[1:]}\n"
    )
    print(f"wrote {outputs / 'result.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
