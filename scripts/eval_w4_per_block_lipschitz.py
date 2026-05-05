"""W4: per-block Lipschitz decomposition C_lag,ℓ + C_sp,ℓ + C_film,ℓ + C_out per cell.

Extracts each component's Lipschitz constant from a trained checkpoint:
  - C_lag,ℓ = max_ω σ_max(K̂_ℓ(ω)) — lag-conv spectral norm
  - C_sp,ℓ  = max_ξ σ_max(R_ℓ(ξ))  — spatial-FNO spectral norm
  - C_film,ℓ = ‖tanh(γ_ℓ)‖_∞ ≤ 1   — FiLM modulator (already 1-bounded by tanh)
  - C_out  = ‖head1·head2‖_op       — readout linear ops
  - C_B,ℓ   = ‖B_ℓ‖_op               — 1×1 channel-mix in residual
  - η(θ)   = product over all components

Writes `per_block_lipschitz.json` next to each `best_model.pt`.

Usage:
  python scripts/eval_w4_per_block_lipschitz.py \\
    --roots extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod ...
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def parse_path(ckpt_path):
    parts = ckpt_path.parts
    try:
        idx = parts.index("raw")
    except ValueError:
        return None
    if idx + 4 >= len(parts):
        return None
    fam, reg, mdl, seed_str = parts[idx + 1: idx + 5]
    if not seed_str.startswith("s"):
        return None
    return fam, reg, mdl, int(seed_str[1:])


def per_block_lipschitz(ckpt_path: Path):
    """Load checkpoint and compute per-block C_*."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    cfg = ckpt["config"]

    # Identify blocks via state_dict key prefixes.
    # Expected: blocks.{i}.A_lag.weights__re/im (lag conv kernel)
    #            blocks.{i}.A_spat.weights__re/im (spatial FNO kernel)
    #            blocks.{i}.B.weight (1x1 channel conv)
    #            blocks.{i}.A_lag.film_net.{0,2}.weight
    n_layers = int(cfg.get("model", {}).get("n_layers", cfg.get("n_layers", 3)))

    components = {"per_layer": []}
    for ell in range(n_layers):
        block = {}
        # C_lag: spectral norm over modes
        try:
            kre = sd[f"blocks.{ell}.A_lag.weights__re"].numpy()
            kim = sd[f"blocks.{ell}.A_lag.weights__im"].numpy()
            K = kre + 1j * kim   # shape (in_ch, out_ch, n_modes)
            # max over modes of σ_max(K[:,:,m])
            sigmas = []
            for m in range(K.shape[-1]):
                _, s, _ = np.linalg.svd(K[:, :, m], full_matrices=False)
                sigmas.append(float(s.max()))
            block["C_lag"] = float(max(sigmas))
            block["C_lag_per_mode"] = sigmas
        except Exception as e:
            block["C_lag"] = None
            block["C_lag_err"] = str(e)
        # C_sp: spectral FNO kernel (spatial dims, e.g. (in_ch, out_ch, spatial_modes_x, spatial_modes_y))
        try:
            sre = sd[f"blocks.{ell}.A_spat.weights__re"].numpy()
            sim = sd[f"blocks.{ell}.A_spat.weights__im"].numpy()
            R = sre + 1j * sim
            # R shape: (in_ch, out_ch, *spatial_modes). Iterate over spatial-mode product.
            spatial_modes = R.shape[2:]
            n_spatial = int(np.prod(spatial_modes))
            R_flat = R.reshape(R.shape[0], R.shape[1], n_spatial)
            sigmas_sp = []
            for m in range(n_spatial):
                _, s, _ = np.linalg.svd(R_flat[:, :, m], full_matrices=False)
                sigmas_sp.append(float(s.max()))
            block["C_sp"] = float(max(sigmas_sp))
            block["C_sp_max_n"] = len(sigmas_sp)
        except Exception as e:
            block["C_sp"] = None
            block["C_sp_err"] = str(e)
        # C_B: 1x1 channel-mix in residual (operator norm)
        try:
            B_w = sd[f"blocks.{ell}.B.weight"].numpy()
            # Conv2d/3d weight has shape (out_ch, in_ch, 1, 1, ...) for 1x1 conv
            B_flat = B_w.reshape(B_w.shape[0], B_w.shape[1])
            block["C_B"] = float(np.linalg.svd(B_flat, compute_uv=False).max())
        except Exception as e:
            block["C_B"] = None
            block["C_B_err"] = str(e)
        # C_film: ‖γ‖_∞ — γ is bounded by tanh, so theoretically ≤ 1.
        # Could also estimate the FiLM net's Lipschitz, but for tanh-bounded the easy bound is 1.
        block["C_film"] = 1.0
        block["C_film_note"] = "tanh-bounded multiplier"
        components["per_layer"].append(block)

    # C_out: readout heads
    try:
        h1 = sd["head1.weight"].numpy()
        h2 = sd["head2.weight"].numpy()
        s1 = float(np.linalg.svd(h1, compute_uv=False).max())
        s2 = float(np.linalg.svd(h2, compute_uv=False).max())
        components["C_out"] = float(s1 * s2)
        components["C_out_h1"] = s1
        components["C_out_h2"] = s2
    except Exception as e:
        components["C_out"] = None
        components["C_out_err"] = str(e)

    # No softmax pool in current architecture; CausalSmoother is 1-Lip
    components["C_pool"] = 1.0
    components["C_pool_note"] = "identity / 1-Lipschitz CausalSmoother"

    # eta(theta) = C_out * C_pool * prod_l (C_sp * C_lag * C_film * C_act)
    eta = (components.get("C_out") or 1.0) * components["C_pool"]
    for blk in components["per_layer"]:
        for c in ("C_lag", "C_sp", "C_film"):
            v = blk.get(c)
            if v is not None:
                eta *= v
    # Activation: bounded by 1 if 1-Lipschitz (ReLU); GELU has Lip ≈ 1.0837
    # We don't know which without parsing config — assume 1.0 for ReLU,
    # 1.0837 for GELU.
    activation = cfg.get("model", {}).get("activation",
                  cfg.get("activation", "relu" if cfg.get("sigma") else "gelu"))
    if activation in ("gelu", "GELU"):
        # GELU's Lipschitz constant ≈ 1.0837
        eta *= (1.0837 ** n_layers)
        components["C_act"] = 1.0837
    else:
        eta *= (1.0 ** n_layers)
        components["C_act"] = 1.0
    components["C_act_per_layer"] = activation
    components["eta_total"] = float(eta)
    components["n_layers"] = n_layers
    return components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    ckpts = []
    for r in args.roots:
        rp = Path(r)
        if not rp.is_absolute():
            rp = REPO / rp
        ckpts.extend(sorted(rp.glob("**/best_model.pt")))
    print(f"[w4-perblock] {len(ckpts)} ckpts", flush=True)

    n_done = n_skip = n_fail = 0
    for ckpt in ckpts:
        out_path = ckpt.parent / "per_block_lipschitz.json"
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue
        meta = parse_path(ckpt)
        if meta is None:
            continue
        fam, reg, mdl, seed = meta
        try:
            t0 = time.time()
            comp = per_block_lipschitz(ckpt)
            comp["family"] = fam
            comp["regime"] = reg
            comp["model"] = mdl
            comp["seed"] = seed
            comp["elapsed_s"] = time.time() - t0
            out_path.write_text(json.dumps(comp, indent=2))
            n_done += 1
            print(f"  ok {fam}/{reg}/{mdl}/s{seed} eta={comp['eta_total']:.4f} "
                  f"C_out={comp.get('C_out')} t={comp['elapsed_s']:.1f}s", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {ckpt.parent.name}: {e}", flush=True)
    print(f"[w4-perblock] done={n_done} skip={n_skip} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
