"""Regenerate `viz_samples.npz` from any best_model.pt checkpoint.

Specifically used to fill the V01 Power-column gap for fno_film_nd ×
dist_powerlaw_rd_2d (no viz_samples.npz was saved during the original
training run). Also useful for any future model whose viz wasn't captured.

Usage:
  # Fill the Power FNO+FiLM gap:
  python scripts/regen_viz_samples.py \\
    --roots /workspace/dde-fno/extracted /workspace/dde-fno/outputs \\
    --models fno_film_nd --families dist_powerlaw_rd_2d \\
    --data_dir data_dde_pde

  # Force regenerate everything for a model:
  python scripts/regen_viz_samples.py --roots ... --models lemo_pc_nd --force
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def parse_path(p: Path):
    parts = p.parts
    try:
        seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
    except IndexError:
        return None
    if not seed.startswith("s"):
        return None
    return fam, reg, model, seed


def regen_one(ckpt_path: Path, data_dir: str, device, n_viz: int = 4,
               force: bool = False) -> str:
    out_path = ckpt_path.parent / "viz_samples.npz"
    if out_path.exists() and not force:
        return "skip"
    meta = parse_path(ckpt_path)
    if meta is None:
        return "FAIL bad path"
    fam, regime, model_name, seed = meta
    try:
        from datasets.apebench_dataset import create_apebench_dataloaders
        from train.build_model import build_model
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        noise_std = float(cfg.get("noise_std", 0.05))
        downsample_factor = int(cfg.get("downsample_factor", 2))
        ra = bool(cfg.get("residual_anchor", False))
        _, _, test_loader = create_apebench_dataloaders(
            data_dir, fam, batch_size=n_viz,
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

        x = sample["input"][: n_viz].to(device).float()
        y = sample["target"][: n_viz].to(device).float()
        with torch.no_grad():
            yhat = model(x)
        np.savez(out_path,
                 input=x.cpu().numpy(),
                 target=y.cpu().numpy(),
                 pred=yhat.cpu().numpy())
        return "ok"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n_viz", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[regen-viz] device={device}")

    ckpts = []
    for r in args.roots:
        ckpts.extend(Path(r).rglob("best_model.pt"))

    filtered = []
    for c in ckpts:
        meta = parse_path(c)
        if meta is None:
            continue
        fam, reg, mdl, seed = meta
        if args.families and fam not in args.families:
            continue
        if args.regimes and reg not in args.regimes:
            continue
        if args.models and mdl not in args.models:
            continue
        filtered.append(c)

    print(f"[regen-viz] {len(filtered)} candidate cells")
    n_ok = n_skip = n_fail = 0
    for i, ckpt in enumerate(filtered, 1):
        t0 = time.time()
        msg = regen_one(ckpt, args.data_dir, device, n_viz=args.n_viz, force=args.force)
        meta = parse_path(ckpt)
        tag = "/".join(meta) if meta else "?"
        print(f"[{i}/{len(filtered)}] {tag}: {msg} ({time.time()-t0:.1f}s)", flush=True)
        if msg == "ok":
            n_ok += 1
        elif msg == "skip":
            n_skip += 1
        else:
            n_fail += 1
    print(f"[regen-viz] ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
