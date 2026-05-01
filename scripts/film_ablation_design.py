"""
FiLM-vs-equivariance ablation suite — self-test script.

The model factories themselves now live in `src/models/film_ablations.py`
(proper module location).  This script imports them and exposes the same
build_variant / VARIANTS dispatch helpers used by reviewers, plus a
parameter-count self-test under __main__.

Build interface (matches build_model contract in src/train/build_model.py):
    create_<variant>(in_channels: int,
                     out_channels: int,
                     config: dict,
                     length: int | None = None) -> nn.Module

Run:
    python scripts/film_ablation_design.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.film_ablations import (  # noqa: E402
    create_fno_film_nd,         # A1
    create_noneq_film_nd,       # A2
    create_lemo_bcorrect_nd,    # A3
    count_parameters,
)
from models.lemo_pc_nd import create_lemo_pc_nd  # noqa: E402


def create_full_lemo_pc_nd(in_channels: int, out_channels: int, config: dict,
                            length: Optional[int] = None) -> nn.Module:
    """A4 — full LEMO-PC ND, the headline architecture, baseline reference."""
    return create_lemo_pc_nd(in_channels, out_channels, config, length=length)


# =====================================================================
# Dispatcher.  The launch script passes one of these names as --model and
# the trainer resolves it through this dict.
# =====================================================================

VARIANTS = {
    "fno_film_nd":       create_fno_film_nd,        # A1
    "noneq_film_nd":     create_noneq_film_nd,      # A2
    "lemo_bcorrect_nd":  create_lemo_bcorrect_nd,   # A3
    "lemo_pc_nd":        create_full_lemo_pc_nd,    # A4 (baseline ref)
}


def build_variant(name: str, in_channels: int, out_channels: int,
                  config: dict, length: Optional[int] = None) -> nn.Module:
    if name not in VARIANTS:
        raise ValueError(f"unknown FiLM-ablation variant {name!r}; "
                          f"allowed: {sorted(VARIANTS.keys())}")
    return VARIANTS[name](in_channels, out_channels, config, length=length)


if __name__ == "__main__":
    # Self-test: instantiate each variant on a 2D distributed-delay
    # configuration and print parameter counts.
    cfg = {
        "model": {
            "length":         64,
            "spatial_shape":  [64, 64],
            "spatial_modes":  [12, 12],
            "lag_modes":      24,
            "width":          64,
            "n_layers":       3,
            "params_dim":     2,
            "film_hidden":    64,
            "kernel_hidden":  64,
            "sigma":          None,
        },
    }
    for name in VARIANTS:
        m = build_variant(name, in_channels=6, out_channels=1, config=cfg, length=64)
        print(f"  {name:<22s}  params = {count_parameters(m):>12,d}")
