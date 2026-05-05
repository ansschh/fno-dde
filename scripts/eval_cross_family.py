"""Cross-family generalization eval: evaluate each checkpoint on all 5 dist_*_rd_2d
test sets to fill F10's in-distribution-vs-OOD scatter for every model.

For each best_model.pt under the given --roots:
  - Read the training family from the path
  - Load model, run inference on all 5 dist_*_rd_2d test sets at the cell's regime
  - Save per-cell `cross_family_relL2.json` with the same schema lemo_pc_nd uses:

    {
      "ckpt_family": "dist_exp_rd_2d",
      "rel_l2": {"dist_exp_rd_2d": 0.014, "dist_gaussian_rd_2d": 0.045, ...},
      "skipped": [],
      "source_fingerprint": {"spatial_shape": [64,64], "n_hist": 64, ...},
      "elapsed_s": 12.3
    }

Idempotent: skips cells whose `cross_family_relL2.json` already exists.

Usage:
  python scripts/eval_cross_family.py --data_dir data_dde_pde \\
    --roots extracted outputs

  # or restrict:
  python scripts/eval_cross_family.py --data_dir data_dde_pde \\
    --roots extracted/pod_pulls_2026_05_03_final \\
    --models s4_nd nide_nd ndde_nd
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

FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]


def parse_path(ckpt_path):
    parts = ckpt_path.parts
    try:
        seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
    except IndexError:
        return None
    if not seed.startswith("s"):
        return None
    return fam, reg, model, seed


def load_model(ckpt_path: Path, fam: str, data_dir: str, device):
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    regime = cfg.get("regime", "clean")
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    ra = bool(cfg.get("residual_anchor", False))
    _, _, test_loader = create_apebench_dataloaders(
        data_dir, fam, batch_size=4,
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
    return model, cfg


def eval_on_family(model, eval_fam: str, regime: str, noise_std: float,
                    downsample_factor: int, ra: bool, data_dir: str,
                    device, n_batches: int = 16) -> float | None:
    from datasets.apebench_dataset import create_apebench_dataloaders
    try:
        _, _, test_loader = create_apebench_dataloaders(
            data_dir, eval_fam, batch_size=4,
            regime=regime, noise_std=noise_std,
            downsample_factor=downsample_factor,
            residual_anchor=ra, seed=42,
        )
    except Exception:
        return None
    rel_l2_vals = []
    with torch.no_grad():
        seen = 0
        for batch in test_loader:
            if seen >= n_batches:
                break
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            try:
                yhat = model(x)
            except Exception:
                return None
            num = ((yhat - y) ** 2 * mask_bc).sum(dim=tuple(range(1, yhat.dim()))).sqrt()
            den = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))).sqrt().clamp_min(1e-12)
            rel = (num / den).cpu().tolist()
            rel_l2_vals.extend(rel)
            seen += 1
    if not rel_l2_vals:
        return None
    return float(np.mean(rel_l2_vals))


def evaluate_checkpoint(ckpt_path: Path, data_dir: str, device,
                         n_batches: int = 16, force: bool = False):
    out_path = ckpt_path.parent / "cross_family_relL2.json"
    if out_path.exists() and not force:
        return "skip"
    meta = parse_path(ckpt_path)
    if meta is None:
        return "FAIL bad path"
    train_fam, regime, model_name, seed = meta
    if train_fam not in FAMS:
        return "skip non-dist family"
    t0 = time.time()
    try:
        model, cfg = load_model(ckpt_path, train_fam, data_dir, device)
    except Exception as e:
        return f"FAIL load {type(e).__name__}: {e}"
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    ra = bool(cfg.get("residual_anchor", False))

    rel_l2_dict = {}
    skipped = []
    for fam in FAMS:
        v = eval_on_family(model, fam, regime, noise_std, downsample_factor,
                            ra, data_dir, device, n_batches)
        if v is None:
            skipped.append(fam)
        else:
            rel_l2_dict[fam] = v

    if not rel_l2_dict:
        return "FAIL no families"

    payload = {
        "ckpt_family": train_fam,
        "regime": regime,
        "model": model_name,
        "seed": seed,
        "rel_l2": rel_l2_dict,
        "skipped": skipped,
        "source_fingerprint": {
            "spatial_shape": cfg.get("spatial_shape", [64, 64]),
            "spatial_dims": 2,
            "num_channels": int(cfg.get("num_channels", 1)),
            "n_hist": int(cfg.get("n_hist", 64)),
            "n_out": int(cfg.get("n_out", 64)),
            "params_dim": int(cfg.get("params_dim", 5)),
        },
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    in_dist_v = rel_l2_dict.get(train_fam, float("nan"))
    ood_v = float(np.mean([v for f, v in rel_l2_dict.items() if f != train_fam])) if len(rel_l2_dict) > 1 else float("nan")
    return f"ok in_dist={in_dist_v:.4f} ood_mean={ood_v:.4f} t={payload['elapsed_s']:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None,
                     help="Sweep roots to crawl for ckpts. Required unless --shard given.")
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--models", nargs="*", default=None,
                     help="Optional model-name filter")
    ap.add_argument("--families", nargs="*", default=None,
                     help="Optional ckpt-family filter")
    ap.add_argument("--regimes", nargs="*", default=None,
                     help="Optional regime filter")
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max_cells", type=int, default=None)
    ap.add_argument("--shard", default=None,
                     help="Optional newline-separated list of ckpt paths to process")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    if args.shard:
        ckpts = [Path(p) for p in Path(args.shard).read_text().splitlines()
                 if p.strip()]
    else:
        if not args.roots:
            print("[cross_family] ERROR: must provide --roots or --shard")
            return
        ckpts = []
        for r in args.roots:
            ckpts.extend(Path(r).rglob("best_model.pt"))

    filtered = []
    for c in ckpts:
        meta = parse_path(c)
        if meta is None:
            continue
        fam, reg, mdl, seed = meta
        if fam not in FAMS:
            continue
        if args.families and fam not in args.families:
            continue
        if args.regimes and reg not in args.regimes:
            continue
        if args.models and mdl not in args.models:
            continue
        filtered.append(c)

    if args.max_cells:
        filtered = filtered[: args.max_cells]

    print(f"[cross_family] {len(filtered)} cells, gpu={args.gpu}, device={device}")
    n_ok = n_skip = n_fail = 0
    for i, ckpt in enumerate(filtered, 1):
        msg = evaluate_checkpoint(ckpt, args.data_dir, device,
                                    n_batches=args.n_batches, force=args.force)
        meta = parse_path(ckpt)
        tag = "/".join(meta) if meta else "?"
        print(f"[{i}/{len(filtered)}] {tag}: {msg}", flush=True)
        if msg.startswith("ok"):
            n_ok += 1
        elif msg.startswith("skip"):
            n_skip += 1
        else:
            n_fail += 1
    print(f"[cross_family] ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
