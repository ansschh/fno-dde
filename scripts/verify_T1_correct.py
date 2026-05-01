"""Verify Theorem T1 (cyclic-shift equivariance) the CORRECT way:
shift the WHOLE input tensor along the lag axis (all channels uniformly).

The earlier capture_equivariance / M4 test shifted ONLY the state channels
and kept mask/time/params fixed.  That is NOT testing T1 — it feeds the
model an inconsistent input where the state moved but the per-lag aux
channels didn't.  The theorem statement is: shift along the lag axis
preserves the prediction up to the same shift, FOR THE WHOLE INPUT.

Usage:
    python3 scripts/verify_T1_correct.py [--ckpt path] [--family fam]

Reports per-shift relative L2 error of |LEMO(roll(x)) - roll(LEMO(x))| /
|LEMO(x)|.  Architecture is cyclic-equivariant iff error is at machine
precision (~1e-6 in float32) for every shift.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_CKPT = (REPO / "extracted_lemo_pc" / "outputs" / "dist_kernel_v2_p1"
                / "raw" / "dist_exp_rd_2d" / "clean" / "lemo_pc_nd" / "s42"
                / "best_model.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--family", default="dist_exp_rd_2d")
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--shifts", default="1,4,16,32,63")
    args = ap.parse_args()

    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ra = bool(cfg.get("residual_anchor", True))
    regime = cfg.get("regime", "clean")
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    _, _, test_loader = create_apebench_dataloaders(
        args.data_dir, args.family, batch_size=2,
        regime=regime, noise_std=noise_std, downsample_factor=downsample_factor,
        residual_anchor=ra, seed=42)
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    x = sample["input"].to(device).float()
    print(f"input shape: {tuple(x.shape)}  (B, T, ..., C)")
    print(f"  C = {in_ch}, T = {n_total}, residual_anchor = {ra}")

    with torch.no_grad():
        y0 = model(x)
        # Synthetic random-input check: equivariance must hold for ANY input,
        # not just training-distribution inputs.  This isolates the
        # architectural property from data effects.
        x_rand = torch.randn_like(x)
        y_rand = model(x_rand)

        shifts = [int(s) for s in args.shifts.split(",")]
        print("\n=== TEST 1: real test input, shift ALL channels along lag axis ===")
        print("              shift k    rel_err_real      rel_err_random")
        for k in shifts:
            x_shift     = torch.roll(x,      shifts=k, dims=1)   # ALL channels along T
            x_rand_shift = torch.roll(x_rand, shifts=k, dims=1)
            y_shift     = model(x_shift)
            y_rand_sh   = model(x_rand_shift)
            y_rolled    = torch.roll(y0,    shifts=k, dims=1)
            y_rand_rolled = torch.roll(y_rand, shifts=k, dims=1)
            num_real = (y_shift     - y_rolled).flatten(1).norm(dim=1)
            den_real = y_rolled.flatten(1).norm(dim=1).clamp_min(1e-12)
            num_rand = (y_rand_sh    - y_rand_rolled).flatten(1).norm(dim=1)
            den_rand = y_rand_rolled.flatten(1).norm(dim=1).clamp_min(1e-12)
            rel_real = (num_real / den_real).mean().item()
            rel_rand = (num_rand / den_rand).mean().item()
            print(f"                  {k:3d}      {rel_real:.3e}        {rel_rand:.3e}")

        print("\n=== TEST 2: state-only shift (the OLD broken test) ===")
        n_state_ch = sample["target"].shape[-1]
        for k in shifts:
            x_state = x[..., :n_state_ch]
            x_aux   = x[..., n_state_ch:]
            x_state_shift = torch.roll(x_state, shifts=k, dims=1)
            x_state_only  = torch.cat([x_state_shift, x_aux], dim=-1)  # AUX UNCHANGED
            y_state_only_shift = model(x_state_only)
            y_rolled = torch.roll(y0, shifts=k, dims=1)
            num = (y_state_only_shift - y_rolled).flatten(1).norm(dim=1)
            den = y_rolled.flatten(1).norm(dim=1).clamp_min(1e-12)
            rel = (num / den).mean().item()
            print(f"  shift k={k:3d}: rel-err = {rel:.3e} "
                  f"(this is the WRONG test; not testing T1)")


if __name__ == "__main__":
    main()
