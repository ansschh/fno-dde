"""W3 — re-evaluate orbit OOD on a FIXED test set across all m values.

Original orbit OOD reports test rel-L2 on each m's *own* test set (the
complement of train shifts). This makes the cross-m comparison apples-to-oranges
because the test composition changes with m: at m=8 the test set is 32
interspersed shifts (r(A)≈2.5); at m=32 the test set is a connected strip
{32,...,39} (r(A)≈4 hardest shifts).

To match the theoretical prediction (per-lag MLP error ∝ C·r(A); LEMO-PC error
flat) we evaluate EVERY checkpoint on the SAME fixed test set: the m=32 test
shard `data_orbit_ood/m32/dist_exp_rd_2d_orbit/test/shard_000.npz`.

Output: per-cell `orbit_fixed_test.json` with rel-L2 on the canonical test set.
Plus an aggregated CSV.

Usage:
  python scripts/eval_w3_fixed_testset.py \\
    --roots extracted/pod_pulls_2026_05_03_final/NewPod_h100/outputs/orbit_ood_h100 \\
            extracted/pod_pulls_2026_05_03_final/Pod2_h100/outputs/orbit_ood_h100 \\
    --canonical_data data_orbit_ood/m32/dist_exp_rd_2d_orbit/test/shard_000.npz
"""
from __future__ import annotations
import argparse
import json
import re
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


def get_m_from_path(ckpt_path):
    """Extract m from path segment like 'lemo_pc_nd_m8' or 'per_lag_mlp_nd_m32'."""
    pattern = re.compile(r"(?:^|_)m(\d+)$")
    for seg in ckpt_path.parts:
        m = pattern.search(seg)
        if m:
            return int(m.group(1))
    return None


def evaluate_on_canonical_via_loader(ckpt_path, canonical_data_dir, family, device):
    """Use create_apebench_dataloaders to load the canonical test set,
    applying the model's own residual_anchor / noise_std / downsample pipeline."""
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ra = bool(cfg.get("residual_anchor", False))
    noise_std = float(cfg.get("noise_std", 0.0))
    downsample_factor = int(cfg.get("downsample_factor", 1))

    # Load canonical test loader. The 'family' is dist_exp_rd_2d_orbit; data lives
    # at canonical_data_dir/dist_exp_rd_2d_orbit/{train,val,test}/.
    _, _, test_loader = create_apebench_dataloaders(
        canonical_data_dir, family, batch_size=4,
        regime="clean", noise_std=noise_std,
        downsample_factor=downsample_factor,
        residual_anchor=ra, seed=42,
    )
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]

    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    rels = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            yhat = model(x)
            num = torch.linalg.vector_norm(((yhat - y) * mask_bc).reshape(y.shape[0], -1), dim=1)
            den = torch.linalg.vector_norm((y * mask_bc).reshape(y.shape[0], -1), dim=1).clamp_min(1e-12)
            rels.extend((num / den).cpu().numpy().tolist())
    return float(np.mean(rels)), float(np.std(rels)), len(rels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--canonical_data_dir", required=True,
                    help="Path to the canonical data dir (e.g. data_orbit_ood/m32). "
                         "Must contain dist_exp_rd_2d_orbit/{train,val,test}/")
    ap.add_argument("--family", default="dist_exp_rd_2d_orbit")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    ckpts = []
    for r in args.roots:
        rp = Path(r)
        if not rp.is_absolute():
            rp = REPO / rp
        ckpts.extend(sorted(rp.glob("**/best_model.pt")))
    print(f"[w3-fixed] {len(ckpts)} ckpts to evaluate", flush=True)

    results = []
    for ckpt in ckpts:
        meta = parse_path(ckpt)
        if meta is None:
            continue
        fam, reg, mdl, seed = meta
        m = get_m_from_path(ckpt)
        if m is None:
            continue
        try:
            t0 = time.time()
            mean_rel, std_rel, n = evaluate_on_canonical_via_loader(
                ckpt, args.canonical_data_dir, args.family, device)
            elapsed = time.time() - t0
            result = {
                "family": fam, "regime": reg, "model": mdl, "seed": seed, "m": m,
                "canonical_rel_l2_mean": mean_rel,
                "canonical_rel_l2_std": std_rel,
                "n_samples": n,
                "canonical_data_dir": args.canonical_data_dir,
                "elapsed_s": elapsed,
            }
            (ckpt.parent / "orbit_fixed_test.json").write_text(json.dumps(result, indent=2))
            results.append(result)
            print(f"  {mdl}/m{m}/s{seed}: rel_l2={mean_rel:.4f}±{std_rel:.4f}  n={n}  t={elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"  FAIL {ckpt.parent.name}: {e}", flush=True)

    print(f"[w3-fixed] processed {len(results)} cells", flush=True)


if __name__ == "__main__":
    main()
