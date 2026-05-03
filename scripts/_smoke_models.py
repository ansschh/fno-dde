"""Build-only smoke test for every unique model in the offload cell list.

Constructs each model with a representative config, runs ONE forward pass
on a tiny synthetic input, verifies output shape, and exits.  Catches
import errors and missing-deps issues per model BEFORE we burn 200 epochs
on a misconfigured model.

Exit codes:
    0  all models built + forward-passed cleanly
    1+ N models failed (see stderr)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _caltech_offload_cells import all_cells

# Unique models in the offload cell list.
unique_models = sorted({c["model"] for c in all_cells()})
print(f"[smoke-models] testing {len(unique_models)} unique models: {unique_models}")

# Representative config for forward-pass smoke (matches data layout for
# dist_*_rd_2d clean: 1×64 lag, 64×64 spatial, 1 state channel + aux).
B = 2
n_total = 128       # 64 hist + 64 out
n_hist = 64
spatial = (64, 64)
in_channels = 1 + 1 + 1 + 5  # state + mask + time + 5 params
out_channels = 1
config_template = {
    "residual_anchor": True,
    "regime": "clean",
    "noise_std": 0.05,
    "downsample_factor": 2,
    "model": {
        "length":         n_total,
        "spatial_shape":  list(spatial),
        "spatial_modes":  [12, 12],
        "lag_modes":      24,
        "physical_shape": [n_total, *spatial],
        "modes":          [24, 12, 12],
        "width":          64,
        "n_layers":       3,
        "kernel_hidden":  64,
        "params_dim":     5,
        "sigma":          None,
        "activation":     "gelu",
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[smoke-models] device: {device}")
x = torch.randn(B, n_total, *spatial, in_channels, device=device)

from train.build_model import build_model

failed = []
for m in unique_models:
    cfg = dict(config_template)
    cfg["model_class"] = m
    try:
        model = build_model(cfg, in_channels=in_channels,
                             out_channels=out_channels, length=n_total)
        model = model.to(device).eval()
        with torch.no_grad():
            y = model(x)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ok = (y.shape[0] == B and y.shape[-1] == out_channels)
        if not ok:
            raise RuntimeError(f"unexpected output shape {tuple(y.shape)}")
        print(f"  [PASS] {m:20s}  params={n_p:>10,d}  out={tuple(y.shape)}")
        del model, y
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        failed.append((m, str(e)))
        print(f"  [FAIL] {m:20s}  -> {e}", file=sys.stderr)

if failed:
    print(f"\n[smoke-models] {len(failed)} of {len(unique_models)} models FAILED:",
          file=sys.stderr)
    for m, msg in failed:
        print(f"  {m}: {msg}", file=sys.stderr)
    sys.exit(len(failed))
print(f"\n[smoke-models] all {len(unique_models)} models PASS")
sys.exit(0)
