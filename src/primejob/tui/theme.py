"""Custom Textual theme inspired by Prime Intellect's brand:
pure black background, white text, Ubuntu-orange accent."""
from __future__ import annotations

from textual.theme import Theme


PRIME_THEME = Theme(
    name="prime",
    # Chrome stays black, panels nearly black so cards barely lift off the bg.
    background="#000000",
    surface="#000000",
    panel="#0a0a0a",
    # Foreground is pure white — minimalist mono terminal feel.
    foreground="#f5f5f5",
    # Primary = white (used for borders, titles).
    primary="#ffffff",
    # Accent = Ubuntu orange (matches the orange badges in the PI web UI).
    accent="#f47421",
    # Secondary used rarely — Verda-ish purple.
    secondary="#7c3aed",
    # Status colors.
    success="#00d4aa",
    warning="#ffb300",
    error="#ff4444",
    dark=True,
    # Slightly muted text-on-bg by default; bold elements stand out cleaner.
    text_alpha=0.92,
    # Tight luminosity spread keeps everything black-leaning, no grayish wash.
    luminosity_spread=0.08,
)


THEME_CYCLE = [
    "prime",
    "tokyo-night",
    "nord",
    "catppuccin-mocha",
    "gruvbox",
    "monokai",
]
