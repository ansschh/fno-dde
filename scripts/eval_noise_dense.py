"""Dense-sigma Gaussian-noise eval across ALL local checkpoints.

Mirrors the structure of `eval_equivariance_dense.py` and
`eval_adversarial_dense.py`:
  - Denser σ grid: σ in {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}
  - Crawls multiple sweep roots in one invocation
  - Writes per-cell `noise_dense.json` next to each `best_model.pt`
  - Skips cells whose dense file already exists (resumable)

Adds Gaussian noise of std σ to input state channels (only) — average-case
robustness, complementary to adversarial FGSM (worst-case).
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

DENSE_SIGMA_DEFAULT = "0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0"


def noise_eval(model, test_loader, device, sigmas, n_batches=16, rng_seed=0):
    """Returns {sigma -> {mean, std, n}}. Adds Gaussian to state channels only."""
    out = {float(s): [] for s in sigmas}
    rng = np.random.default_rng(rng_seed)
    seen_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            if seen_batches >= n_batches:
                break
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            n_state_ch = y.shape[-1]
            x_state = x[..., :n_state_ch]
            x_aux = x[..., n_state_ch:]
            for sigma in sigmas:
                sigma = float(sigma)
                if sigma > 0:
                    eps = torch.from_numpy(
                        rng.normal(0.0, sigma, size=tuple(x_state.shape))
                    ).float().to(device)
                    x_pert = x_state + eps
                else:
                    x_pert = x_state
                x_full = torch.cat([x_pert, x_aux], dim=-1)
                yhat = model(x_full)
                num = ((yhat - y) ** 2 * mask_bc).sum(dim=tuple(range(1, yhat.dim()))).sqrt()
                den = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))).sqrt().clamp_min(1e-12)
                rel = (num / den).cpu().tolist()
                out[sigma].extend(rel)
            seen_batches += 1
    return {f"{s:g}": {"mean": float(np.mean(v)) if v else float("nan"),
                       "std":  float(np.std(v))  if v else float("nan"),
                       "n":    len(v)}
            for s, v in out.items()}


def evaluate_checkpoint(ckpt_path: Path, data_dir: str, sigmas, device, n_batches: int):
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
    res = noise_eval(model, test_loader, device, sigmas, n_batches=n_batches)
    return {"family": family, "regime": regime, "model": model_name, "seed": seed,
            "sigmas": [float(s) for s in sigmas], "noise": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--sigmas", default=DENSE_SIGMA_DEFAULT)
    ap.add_argument("--families", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--regimes", default="clean")
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]
    families_filter = set(args.families.split(",")) if args.families else None
    models_filter = set(args.models.split(",")) if args.models else None
    regimes_filter = set(args.regimes.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[noise-dense] device={device} sigmas={sigmas} regimes={regimes_filter}")

    ckpts = []
    for root in args.roots:
        rp = Path(root)
        ckpts.extend(sorted(rp.glob("raw/**/best_model.pt")))
    print(f"[noise-dense] found {len(ckpts)} candidate checkpoints across {len(args.roots)} roots")

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
        out_path = c.parent / "noise_dense.json"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue
        try:
            t1 = time.time()
            r = evaluate_checkpoint(c, args.data_dir, sigmas, device, args.n_batches)
            json.dump(r, open(out_path, "w"), indent=2)
            n_done += 1
            dt = time.time() - t1
            mn = {f"{s:g}": f"{r['noise'][f'{s:g}']['mean']:.3e}" for s in sigmas}
            print(f"  [{n_done:3d}/{n_total}] {family}/{regime}/{model_name}/{seed} "
                  f"({dt:.1f}s) sigma_means={mn}")
        except Exception as ex:
            n_failed += 1
            print(f"  FAILED {family}/{regime}/{model_name}/{seed}: {ex}")
    elapsed = time.time() - t0
    print(f"[noise-dense] done={n_done} skipped={n_skipped} failed={n_failed} "
          f"total_seen={n_total} elapsed={elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
