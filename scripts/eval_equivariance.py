"""Equivariance error E_orbit (Metric M3 from the plan).

For each (dataset, model, seed), compute:

    E_orbit = avg over (x in test, k in C_n) of
              || F(shift_k x) - shift_k F(x) ||_2 / || F(x) ||_2

where shift_k is a cyclic shift along the lag axis by k positions.

A perfectly lag-equivariant operator F has E_orbit = 0.
LEMO is designed to satisfy this (Theorem T1).
UNet, FNO etc. have no such guarantee.

This is the test that DIRECTLY measures the lag-equivariance theorem,
unlike tau-parameter-OOD which tests something else.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def cyclic_shift_state_only(x: torch.Tensor, k: int, n_state_channels: int) -> torch.Tensor:
    """Cyclic shift along axis 1 (lag) by k, BUT only on the first
    n_state_channels of the channel dim.  Time/mask/params channels are kept
    fixed because rotating "time" wraps t_max → t_0 (violates monotonicity)
    and FiLM params are read at a fixed lag position so shifting them
    changes the conditioning.  T1 only guarantees equivariance under shifts
    of the underlying state.
    """
    if k % x.shape[1] == 0:
        return x.clone()
    state = x[..., :n_state_channels]
    rest = x[..., n_state_channels:]
    state_shifted = torch.roll(state, shifts=k, dims=1)
    return torch.cat([state_shifted, rest], dim=-1)


def compute_e_orbit(model, test_loader, device, shifts, n_state_channels=1,
                    n_batches=8):
    """For each shift k, compute mean ||F(s_k x) - s_k F(x)|| / denom across
    n_batches test batches.  Shifts only the state channels of the input.
    Output shift is on all channels of F(x) since F(x) is purely state.

    The denominator floors at 0.1 × median(||F(x)||) to avoid near-zero
    outputs blowing up relative error.

    Returns dict: k → {mean, std, n}.
    """
    out = {int(k): [] for k in shifts}
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= n_batches:
                break
            x = batch["input"].to(device).float()
            Fx = model(x)
            Fx_flat = Fx.reshape(Fx.shape[0], -1)
            Fx_norm = torch.norm(Fx_flat, dim=1)
            denom_floor = max(Fx_norm.median().item() * 0.1, 1e-12)
            denom = Fx_norm.clamp_min(denom_floor)
            for k in shifts:
                sk_x = cyclic_shift_state_only(x, int(k), n_state_channels)
                F_sk_x = model(sk_x)
                sk_Fx = torch.roll(Fx, shifts=int(k), dims=1)
                diff = (F_sk_x - sk_Fx).reshape(Fx.shape[0], -1)
                err = torch.norm(diff, dim=1) / denom
                out[int(k)].extend(err.cpu().tolist())
    return {k: {"mean": float(np.mean(v)) if v else float("nan"),
                "std":  float(np.std(v))  if v else float("nan"),
                "n":    len(v)}
            for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer5_root", required=True)
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--shifts", default="1,4,16,32",
                    help="Cyclic-shift values to evaluate.")
    args = ap.parse_args()

    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model

    shifts = [int(s) for s in args.shifts.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer5 = Path(args.layer5_root)
    ckpts = sorted(layer5.glob("raw/**/best_model.pt"))
    print(f"found {len(ckpts)} checkpoints")
    results = {}
    for c in ckpts:
        parts = c.parts
        family = parts[-5]; regime = parts[-4]; model_name = parts[-3]; seed = parts[-2]
        if regime != "clean":
            continue
        try:
            ckpt = torch.load(c, map_location=device, weights_only=False)
            cfg = ckpt["config"]
            _, _, test_loader = create_apebench_dataloaders(
                args.data_dir, family, batch_size=8, seed=42)
            sample = next(iter(test_loader))
            in_ch = sample["input"].shape[-1]
            out_ch = sample["target"].shape[-1]
            n_total = sample["input"].shape[1]
            model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device).eval()
            e = compute_e_orbit(model, test_loader, device, shifts,
                                  n_state_channels=out_ch, n_batches=8)
            per_k_means = [v["mean"] for v in e.values()]
            r = {"e_orbit": e, "mean": float(np.mean(per_k_means))}
            r.update(family=family, model=model_name, seed=seed)
            results[(family, model_name, seed)] = r
            per_k_str = {k: f"{v['mean']:.2e}±{v['std']:.2e}(n={v['n']})" for k, v in e.items()}
            print(f"  {family}/{model_name}/{seed}: E_orbit mean={r['mean']:.4f}  per-k={per_k_str}")
        except Exception as e:
            print(f"  {family}/{model_name}/{seed}: FAILED ({e})")

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({f"{k[0]}__{k[1]}__{k[2]}": v for k, v in results.items()},
              open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
