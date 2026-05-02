"""Dense per-frame eval — generates per_frame.json for any checkpoint that
lacks one. Used by F06 (rollout) and F_boundary_error.

Mirrors eval_equivariance_dense / eval_adversarial_dense / eval_noise_dense:
  - Crawls multiple sweep roots
  - Skip-if-exists for resumability
  - --families / --models / --regimes filters
  - Writes per_frame.json next to each best_model.pt

Output schema:
    {"rel_l2_per_step": [...], "naive_copy_per_step": [...]}
matching the existing post_hoc per_frame format so make_paper_figures.py and
boundary_audit.py pick it up automatically.
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
sys.path.insert(0, str(REPO / "src"))


def per_frame_eval(model, test_loader, device, n_max_samples=16):
    """Returns dict with rel_l2_per_step and naive_copy_per_step.
    Computes spatial-mean rel-L2 at each rollout step, averaged over batches.
    """
    per_step_rel = None       # [T] running sum
    per_step_naive = None
    per_step_count = None
    seen = 0
    with torch.no_grad():
        for batch in test_loader:
            if seen >= n_max_samples:
                break
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            n_state_ch = y.shape[-1]
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            yhat = model(x)
            # Reduce over spatial + channel dims, keep [B, T] structure
            spatial_dims = tuple(range(2, yhat.dim()))
            num = ((yhat - y) ** 2 * mask_bc).sum(dim=spatial_dims).sqrt()
            den = (y ** 2 * mask_bc).sum(dim=spatial_dims).sqrt().clamp_min(1e-12)
            rel = (num / den).cpu().numpy()  # [B, T]
            # Naive baseline: predict last history frame
            x_state = x[..., :n_state_ch]
            last_hist = x_state[:, -1:, ...].expand_as(y)
            num_n = ((last_hist - y) ** 2 * mask_bc).sum(dim=spatial_dims).sqrt()
            naive = (num_n / den).cpu().numpy()
            B, T = rel.shape
            if per_step_rel is None:
                per_step_rel   = np.zeros(T)
                per_step_naive = np.zeros(T)
                per_step_count = np.zeros(T)
            per_step_rel   += rel.sum(axis=0)
            per_step_naive += naive.sum(axis=0)
            per_step_count += B
            seen += B
    if per_step_rel is None:
        return None
    rel_mean = (per_step_rel / per_step_count).tolist()
    naive_mean = (per_step_naive / per_step_count).tolist()
    return {"rel_l2_per_step": rel_mean,
            "naive_copy_per_step": naive_mean}


def evaluate_checkpoint(ckpt_path: Path, data_dir: str, device, n_max_samples: int):
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model

    parts = ckpt_path.parts
    family = parts[-5]; regime = parts[-4]; model_name = parts[-3]; seed = parts[-2]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ra = bool(cfg.get("residual_anchor", False))
    _, _, test_loader = create_apebench_dataloaders(
        data_dir, family, batch_size=8, residual_anchor=ra, seed=42)
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return per_frame_eval(model, test_loader, device, n_max_samples=n_max_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--families", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--regimes", default="clean")
    ap.add_argument("--n_max_samples", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    families_filter = set(args.families.split(",")) if args.families else None
    models_filter = set(args.models.split(",")) if args.models else None
    regimes_filter = set(args.regimes.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[per-frame] device={device} regimes={regimes_filter}")

    ckpts = []
    for root in args.roots:
        rp = Path(root)
        ckpts.extend(sorted(rp.glob("raw/**/best_model.pt")))
    print(f"[per-frame] found {len(ckpts)} candidate checkpoints")

    n_total = n_skipped = n_done = n_failed = 0
    t0 = time.time()
    for c in ckpts:
        parts = c.parts
        family = parts[-5]; regime = parts[-4]; model_name = parts[-3]; seed = parts[-2]
        if families_filter and family not in families_filter:
            continue
        if models_filter and model_name not in models_filter:
            continue
        if regime not in regimes_filter:
            continue
        n_total += 1
        out_path = c.parent / "per_frame.json"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue
        try:
            t1 = time.time()
            r = evaluate_checkpoint(c, args.data_dir, device, args.n_max_samples)
            if r is None:
                n_failed += 1
                print(f"  FAILED (no batches) {family}/{regime}/{model_name}/{seed}")
                continue
            json.dump(r, open(out_path, "w"), indent=2)
            n_done += 1
            dt = time.time() - t1
            T = len(r["rel_l2_per_step"])
            print(f"  [{n_done:3d}/{n_total}] {family}/{regime}/{model_name}/{seed} "
                  f"({dt:.1f}s) T={T} mean={np.mean(r['rel_l2_per_step']):.3e}")
        except Exception as ex:
            n_failed += 1
            print(f"  FAILED {family}/{regime}/{model_name}/{seed}: {ex}")
    elapsed = time.time() - t0
    print(f"[per-frame] done={n_done} skipped={n_skipped} failed={n_failed} "
          f"total_seen={n_total} elapsed={elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
