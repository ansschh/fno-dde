"""Long-horizon rollout eval: autoregressive prediction at h ∈ {64, 128, 256, 512}.

Demonstrates the certified-rollout claim (Cor 5.15b) holds at long times for
σ-projected LEMO-PC vs unconstrained alternatives.

For each checkpoint:
  - Roll out autoregressively for n_chain ∈ {1, 2, 4, 8} extensions of the
    base 64-frame window, giving effective horizons {64, 128, 256, 512}.
  - Per-step rel-L2 averaged over batch + spatial dimensions.
  - Track ‖H_t‖ at each step (state norm) — flags blowup.
  - Compare against ground truth where available (h ≤ 64 from test set);
    beyond that, just track norm growth.

Writes `long_horizon.json` next to each best_model.pt with:
  {
    "h_64": {"rel_l2_mean": ..., "rel_l2_per_step": [...], "norm_per_step": [...]},
    "h_128": {...}, "h_256": {...}, "h_512": {...},
    "blow_up": bool,  // any norm > 100x initial
  }

Usage:
  python scripts/eval_long_horizon.py \\
    --roots extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod ... \\
    --data_dir data_dde_pde
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


def load_cell(ckpt_path, data_dir, family, device):
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    parts = ckpt_path.parts
    regime = cfg.get("regime", parts[-4] if len(parts) >= 4 else "clean")
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    ra = bool(cfg.get("residual_anchor", False))
    _, _, test_loader = create_apebench_dataloaders(
        data_dir, family, batch_size=4,
        regime=regime, noise_std=noise_std,
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
    return model, test_loader, cfg


def long_horizon_rollout(model, test_loader, device, n_chain_max=8, n_max_samples=32):
    """Chained autoregressive rollout with norm tracking.

    Each chain extends the prediction by `n_total` frames. At chain n, effective
    horizon is `(n+1) * n_total` frames (initial + n extensions).
    """
    model.eval()
    norm_per_step = []   # list of (chain_norm,) arrays, one per batch
    rel_l2_step0 = []     # rel_l2 vs GT on first 64 frames (where GT is available)
    n_seen = 0

    with torch.no_grad():
        for batch in test_loader:
            if n_seen >= n_max_samples:
                break
            x = batch["input"].to(device).float()      # (B, T, *spatial, C+aux)
            y = batch["target"].to(device).float()      # (B, T, *spatial, C_state)
            mask = batch["loss_mask"].to(device).float()
            B = x.shape[0]
            T = x.shape[1]
            C_state = y.shape[-1]

            x_curr = x.clone()
            chain_norms = []   # list of (T,) per chain step
            for c in range(n_chain_max):
                yhat = model(x_curr)   # (B, T, *spatial, C_state)
                spatial_dims = tuple(range(2, yhat.dim()))
                norm_per_t = torch.sqrt((yhat ** 2).sum(dim=spatial_dims)).mean(dim=0).cpu().numpy()
                chain_norms.append(norm_per_t)
                if c == 0:
                    n_spatial = y.dim() - 3
                    mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
                    diff_sq = ((yhat - y) ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim())))
                    tgt_sq = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))) + 1e-12
                    rel = torch.sqrt(diff_sq / tgt_sq).cpu().numpy()
                    rel_l2_step0.extend(rel.tolist())
                # Build next chain: shift x_curr forward by n_total state frames
                # so the newest n_total frames become the history.
                x_state = x_curr[..., :C_state]
                # Shift forward: prepend prediction state, drop oldest
                # New history = the prediction we just made
                x_state_next = yhat
                # Replace the state channels in x_curr with the prediction,
                # keeping aux channels (mask/time/params).
                aux = x_curr[..., C_state:]
                x_curr = torch.cat([x_state_next, aux], dim=-1)
            norm_per_step.append(np.stack(chain_norms))   # (n_chain, T)
            n_seen += B

    if not norm_per_step:
        return None
    norms = np.stack(norm_per_step).mean(axis=0)   # (n_chain, T)
    h_64 = norms[0]
    h_128 = np.concatenate([norms[0], norms[1]]) if norms.shape[0] >= 2 else norms[0]
    h_256 = np.concatenate(list(norms[:4])) if norms.shape[0] >= 4 else h_128
    h_512 = np.concatenate(list(norms[:8])) if norms.shape[0] >= 8 else h_256
    initial_norm = float(norms[0, 0])
    blow_up = bool(np.any([n.max() > 100 * max(initial_norm, 1e-6) for n in [h_64, h_128, h_256, h_512]]))

    out = {
        "h_64":  {"final_norm": float(h_64[-1]), "max_norm": float(h_64.max()),
                  "norm_per_step": h_64.tolist()},
        "h_128": {"final_norm": float(h_128[-1]), "max_norm": float(h_128.max()),
                  "norm_per_step": h_128.tolist()},
        "h_256": {"final_norm": float(h_256[-1]), "max_norm": float(h_256.max()),
                  "norm_per_step": h_256.tolist()},
        "h_512": {"final_norm": float(h_512[-1]), "max_norm": float(h_512.max()),
                  "norm_per_step": h_512.tolist()},
        "rel_l2_h64_mean": float(np.mean(rel_l2_step0)) if rel_l2_step0 else None,
        "rel_l2_h64_max": float(np.max(rel_l2_step0)) if rel_l2_step0 else None,
        "blow_up": blow_up,
        "n_samples": int(n_seen),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=None,
                    help="Sweep roots to crawl for ckpts. Required unless --shard given.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_chain_max", type=int, default=8)
    ap.add_argument("--n_max_samples", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default=None,
                    help="Optional shard file (one ckpt path per line). "
                         "If given, only process these.")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.shard:
        ckpts = [Path(line.strip()) for line in open(args.shard) if line.strip()]
    elif args.roots:
        ckpts = []
        for r in args.roots:
            rp = Path(r)
            if not rp.is_absolute():
                rp = REPO / rp
            ckpts.extend(sorted(rp.glob("**/best_model.pt")))
    else:
        print("[long-h] ERROR: must provide either --shard or --roots", flush=True)
        sys.exit(2)
    print(f"[long-h] {len(ckpts)} ckpts, device={device}", flush=True)

    n_done = n_skip = n_fail = 0
    for i, ckpt in enumerate(ckpts, 1):
        out_path = ckpt.parent / "long_horizon.json"
        if out_path.exists():
            n_skip += 1
            continue
        meta = parse_path(ckpt)
        if meta is None:
            n_fail += 1
            continue
        fam, reg, mdl, seed = meta
        try:
            t0 = time.time()
            model, test_loader, cfg = load_cell(ckpt, args.data_dir, fam, device)
            result = long_horizon_rollout(model, test_loader, device,
                                           n_chain_max=args.n_chain_max,
                                           n_max_samples=args.n_max_samples)
            if result is None:
                n_fail += 1
                continue
            sigma = (cfg.get("model", {}).get("sigma")
                     if cfg.get("model") else None) or cfg.get("sigma")
            result.update({
                "family": fam, "regime": reg, "model": mdl, "seed": seed,
                "sigma": sigma, "elapsed_s": time.time() - t0,
            })
            out_path.write_text(json.dumps(result, indent=2))
            n_done += 1
            print(f"[long-h] [{i}/{len(ckpts)}] {fam}/{reg}/{mdl}/s{seed} "
                  f"σ={sigma} h512_max_norm={result['h_512']['max_norm']:.3f} "
                  f"blow_up={result['blow_up']} t={result['elapsed_s']:.1f}s", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[long-h] [{i}/{len(ckpts)}] FAIL {ckpt}: {e}", flush=True)

    print(f"[long-h] DONE done={n_done} skip={n_skip} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
