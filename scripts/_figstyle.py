"""Global matplotlib style for paper figures.

Importing this module sets rcParams to render every text element in
Times New Roman (with STIX as the math font, the closest serif math set
in matplotlib so that math expressions blend with body text).

Usage in any figure script: place `import _figstyle  # noqa: F401`
near the top, after the matplotlib imports.

Falls back gracefully if Times New Roman is not installed: the serif
fontset still resolves to a Times-like substitute on every platform.
"""
from __future__ import annotations

import matplotlib as _mpl

_TNR_STACK = [
    "Times New Roman",
    "Times",
    "Liberation Serif",
    "DejaVu Serif",
    "serif",
]


def apply_style() -> None:
    rc = _mpl.rcParams
    rc["font.family"] = "serif"
    rc["font.serif"] = _TNR_STACK
    rc["mathtext.fontset"] = "stix"
    rc["mathtext.rm"] = "Times New Roman"
    rc["mathtext.it"] = "Times New Roman:italic"
    rc["mathtext.bf"] = "Times New Roman:bold"
    rc["axes.titleweight"] = "normal"
    rc["axes.labelweight"] = "normal"
    rc["pdf.fonttype"] = 42
    rc["ps.fonttype"] = 42


apply_style()
