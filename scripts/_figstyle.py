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
    # Math text: keep matplotlib's default (Computer Modern via STIX) for
    # math expressions like rel-L_2. User explicitly asked NOT to change
    # the math font; only the surrounding body text uses Times New Roman.
    rc["axes.titleweight"] = "normal"
    rc["axes.labelweight"] = "normal"
    rc["pdf.fonttype"] = 42
    rc["ps.fonttype"] = 42
    # Doubled-from-default font sizes for paper-ready consistency.
    # matplotlib defaults: font.size=10, axes.titlesize/labelsize=10, ticks=10,
    # legend=10. Doubled to 18-20 for camera-ready legibility.
    rc["font.size"] = 18
    rc["axes.titlesize"] = 20
    rc["axes.labelsize"] = 18
    rc["xtick.labelsize"] = 16
    rc["ytick.labelsize"] = 16
    rc["legend.fontsize"] = 18
    rc["figure.titlesize"] = 22


apply_style()
