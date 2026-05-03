"""B5 boundary-error audit — interior vs boundary error decomposition.

Punch list B5 deliverable:
  - Boundary-error audit on at least one family: split per-cell error into
    "interior" (away from wrap-around region by ≥ k cells) vs "boundary".
  - Report the boundary fraction of total error per family.
  - If boundary fraction is non-negligible (>10%), narrow §6.6 interpretation.

Audits cyclic LEMO-PC checkpoints by computing per-time-step errors and
splitting them by position. Writes `boundary_audit.json` next to each
best_model.pt.

Usage:
  python3 scripts/eval_b5_boundary_audit.py \
      --output_root outputs/sigma_0.5_runpod \
      --boundary_k 8

Output: per-cell JSON with:
  total_rel_l2: aggregate error
  interior_rel_l2: error on interior positions (k <= t < L-k)
  boundary_rel_l2: error on boundary positions (t < k OR t >= L-k)
  boundary_fraction: boundary_rel_l2^2 / total_rel_l2^2 (energy fraction)
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


def boundary_audit_batch(model, test_loader, device, boundary_k=8, n_batches=8):
    """Compute interior vs boundary per-time-step errors."""
    model.eval()
    interior_sq, boundary_sq, total_sq = [], [], []
    interior_target_sq, boundary_target_sq, total_target_sq = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= n_batches:
                break
            x = batch["input"].to(device).float()
            y_true = batch["target"].to(device).float()
            y_pred = model(x)
            err = y_pred - y_true                       # (B, L, *spatial, ch)
            L = y_true.shape[1]
            k = min(boundary_k, L // 2)
            # Interior mask: k <= t < L-k
            interior_idx = torch.arange(k, L - k, device=device)
            boundary_idx = torch.cat([torch.arange(0, k, device=device),
                                       torch.arange(L - k, L, device=device)])
            err_int = err.index_select(1, interior_idx)
            err_bd = err.index_select(1, boundary_idx)
            y_int = y_true.index_select(1, interior_idx)
            y_bd = y_true.index_select(1, boundary_idx)
            # Sum-of-squares on per-batch basis
            int_sq = (err_int ** 2).sum(dim=tuple(range(1, err_int.dim()))).cpu().numpy()
            bd_sq = (err_bd ** 2).sum(dim=tuple(range(1, err_bd.dim()))).cpu().numpy()
            tot_sq = (err ** 2).sum(dim=tuple(range(1, err.dim()))).cpu().numpy()
            int_t_sq = (y_int ** 2).sum(dim=tuple(range(1, y_int.dim()))).cpu().numpy()
            bd_t_sq = (y_bd ** 2).sum(dim=tuple(range(1, y_bd.dim()))).cpu().numpy()
            tot_t_sq = (y_true ** 2).sum(dim=tuple(range(1, y_true.dim()))).cpu().numpy()
            interior_sq.extend(int_sq.tolist())
            boundary_sq.extend(bd_sq.tolist())
            total_sq.extend(tot_sq.tolist())
            interior_target_sq.extend(int_t_sq.tolist())
            boundary_target_sq.extend(bd_t_sq.tolist())
            total_target_sq.extend(tot_t_sq.tolist())
    interior_sq = np.array(interior_sq)
    boundary_sq = np.array(boundary_sq)
    total_sq = np.array(total_sq)
    return {
        "boundary_k": boundary_k,
        "interior_rel_l2": float(np.sqrt(interior_sq.sum() / max(np.array(interior_target_sq).sum(), 1e-12))),
        "boundary_rel_l2": float(np.sqrt(boundary_sq.sum() / max(np.array(boundary_target_sq).sum(), 1e-12))),
        "total_rel_l2": float(np.sqrt(total_sq.sum() / max(np.array(total_target_sq).sum(), 1e-12))),
        "boundary_energy_fraction": float(boundary_sq.sum() / max(total_sq.sum(), 1e-12)),
        "interior_energy_fraction": float(interior_sq.sum() / max(total_sq.sum(), 1e-12)),
        "n_batches": n_batches,
        "n_samples": len(interior_sq),
    }


def find_checkpoints(output_root: Path) -> list:
    return list(output_root.glob("**/best_model.pt"))


def parse_path(ckpt_path: Path):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--boundary_k", type=int, default=8)
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    root = Path(args.output_root)
    if not root.is_absolute():
        root = REPO / root
    ckpts = find_checkpoints(root)
    print(f"[B5 audit] {len(ckpts)} checkpoints under {root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(REPO / "src"))
    from train.build_model import build_model

    for ckpt in sorted(ckpts):
        out_path = ckpt.parent / "boundary_audit.json"
        if args.skip_existing and out_path.exists():
            continue
        meta = parse_path(ckpt)
        if meta is None:
            continue
        fam, reg, mdl, seed = meta
        tr = ckpt.parent / "test_results.json"
        if not tr.exists():
            continue
        with open(tr) as fh:
            tr_data = json.load(fh)
        cfg = tr_data.get("config", {
            "model_class": mdl,
            "model": {
                "lag_modes": 24, "spatial_modes": [12, 12],
                "spatial_shape": [64, 64], "width": 64,
                "n_layers": 3, "params_dim": 4,
                "sigma": tr_data.get("sigma"),
            },
        })
        try:
            from data.apebench_loader import create_apebench_dataloaders
            _, _, test_loader = create_apebench_dataloaders(
                args.data_dir, fam, batch_size=4, regime=reg,
                noise_std=0.05, downsample_factor=2, seed=seed,
                residual_anchor=True,
            )
            sample = next(iter(test_loader))
            in_channels = sample["input"].shape[-1]
            out_channels = sample["target"].shape[-1]
            length = sample["input"].shape[1]
            model = build_model(cfg, in_channels=in_channels,
                                  out_channels=out_channels, length=length)
            model = model.to(device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            sd = state if isinstance(state, dict) and "state_dict" not in state else state.get("state_dict", state)
            model.load_state_dict(sd, strict=False)
            audit = boundary_audit_batch(model, test_loader, device,
                                          boundary_k=args.boundary_k,
                                          n_batches=args.n_batches)
            audit["family"] = fam
            audit["regime"] = reg
            audit["model"] = mdl
            audit["seed"] = seed
            with open(out_path, "w") as fh:
                json.dump(audit, fh, indent=2)
            print(f"  OK {ckpt.parent.relative_to(root)}: "
                  f"boundary_frac={audit['boundary_energy_fraction']:.3f}")
        except Exception as e:
            print(f"  FAIL {ckpt}: {e}")


if __name__ == "__main__":
    main()
