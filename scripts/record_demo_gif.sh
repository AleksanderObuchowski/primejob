#!/usr/bin/env bash
# Re-record docs/assets/demo.gif from the offline TUI demo.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run playwright install chromium
mkdir -p docs/assets
uv run tapegif record examples/demo_tui.py:DemoApp \
  --tape examples/demo.tape \
  --output docs/assets/demo.gif
echo "Wrote docs/assets/demo.gif"
