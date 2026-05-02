"""Dense-k equivariance eval across ALL local checkpoints.

Differs from `eval_equivariance.py`:
  - Denser k grid: k in {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64}
  - Crawls multiple sweep roots in one invocation
  - Writes per-cell `equivariance_dense.json` next to each `best_model.pt`
  - Skips cells whose dense file already exists (resumable)
  - Optional `--families` / `--models` filters
  - Larger n_batches (16 by default) for tighter std bars
  - Logs FP32 forward-pass floor by computing F(x) twice and reporting
    ||F(x) - F(x)||/||F(x)||  (numerical reproducibility on the same input)

Use on Caltech via slurm/eval_equivariance_dense.sbatch.
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


DENSE_SHIFTS_DEFAULT = "1,2,4,6,8,12,16,24,32,48,64"


def cyclic_shift_state_only(x: torch.Tensor, k: int, n_state_channels: int) -> torch.Tensor:
    if k % x.shape[1] == 0:
        return x.clone()
    state = x[..., :n_state_channels]
    rest = x[..., n_state_channels:]
    state_shifted = torch.roll(state, shifts=k, dims=1)
    return torch.cat([state_shifted, rest], dim=-1)


def compute_e_orbit(model, test_loader, device, shifts, n_state_channels=1,
                    n_batches=16):
    out = {int(k): [] for k in shifts}
    fp_floor = []
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= n_batches:
                break
            x = batch["input"].to(device).float()
            Fx = model(x)
            Fx_repeat = model(x)  # numerical floor: same input twice
            Fx_flat = Fx.reshape(Fx.shape[0], -1)
            Fx_norm = torch.norm(Fx_flat, dim=1)
            denom_floor = max(Fx_norm.median().item() * 0.1, 1e-12)
            denom = Fx_norm.clamp_min(denom_floor)
            fp_diff = (Fx - Fx_repeat).reshape(Fx.shape[0], -1)
            fp_floor.extend((torch.norm(fp_diff, dim=1) / denom).cpu().tolist())
            for k in shifts:
                sk_x = cyclic_shift_state_only(x, int(k), n_state_channels)
                F_sk_x = model(sk_x)
                sk_Fx = torch.roll(Fx, shifts=int(k), dims=1)
                diff = (F_sk_x - sk_Fx).reshape(Fx.shape[0], -1)
                err = torch.norm(diff, dim=1) / denom
                out[int(k)].extend(err.cpu().tolist())
    res = {k: {"mean": float(np.mean(v)) if v else float("nan"),
                "std":  float(np.std(v))  if v else float("nan"),
                "min":  float(np.min(v))  if v else float("nan"),
                "max":  float(np.max(v))  if v else float("nan"),
                "n":    len(v)}
            for k, v in out.items()}
    res["__fp32_floor__"] = {
        "mean": float(np.mean(fp_floor)) if fp_floor else float("nan"),
        "std":  float(np.std(fp_floor))  if fp_floor else float("nan"),
        "n":    len(fp_floor),
    }
    return res


def evaluate_checkpoint(ckpt_path: Path, data_dir: str, shifts, device,
                        n_batches: int):
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
    e = compute_e_orbit(model, test_loader, device, shifts,
                        n_state_channels=out_ch, n_batches=n_batches)
    return {
        "family": family, "regime": regime, "model": model_name, "seed": seed,
        "shifts": [int(k) for k in shifts],
        "e_orbit": e,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="One or more sweep roots, each containing raw/<fam>/<reg>/<model>/<seed>/best_model.pt")
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--shifts", default=DENSE_SHIFTS_DEFAULT)
    ap.add_argument("--families", default=None,
                    help="Comma-separated families; default = all found")
    ap.add_argument("--models", default=None,
                    help="Comma-separated models; default = all found")
    ap.add_argument("--regimes", default="clean",
                    help="Comma-separated regimes; default = clean only")
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing equivariance_dense.json files")
    args = ap.parse_args()

    shifts = [int(s) for s in args.shifts.split(",")]
    families_filter = set(args.families.split(",")) if args.families else None
    models_filter = set(args.models.split(",")) if args.models else None
    regimes_filter = set(args.regimes.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dense-k-equiv] device={device} shifts={shifts} regimes={regimes_filter}")

    ckpts = []
    for root in args.roots:
        rp = Path(root)
        ckpts.extend(sorted(rp.glob("raw/**/best_model.pt")))
    print(f"[dense-k-equiv] found {len(ckpts)} candidate checkpoints across {len(args.roots)} roots")

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
        out_path = c.parent / "equivariance_dense.json"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue
        try:
            t1 = time.time()
            r = evaluate_checkpoint(c, args.data_dir, shifts, device, args.n_batches)
            json.dump(r, open(out_path, "w"), indent=2)
            n_done += 1
            dt = time.time() - t1
            mn = {k: f"{r['e_orbit'][k]['mean']:.2e}" for k in shifts}
            fp = r['e_orbit'].get('__fp32_floor__', {}).get('mean', float('nan'))
            print(f"  [{n_done:3d}/{n_total}] {family}/{regime}/{model_name}/{seed} "
                  f"({dt:.1f}s) FP32floor={fp:.2e} k_means={mn}")
        except Exception as ex:
            n_failed += 1
            print(f"  FAILED {family}/{regime}/{model_name}/{seed}: {ex}")
    elapsed = time.time() - t0
    print(f"[dense-k-equiv] done={n_done} skipped={n_skipped} failed={n_failed} "
          f"total_seen={n_total} elapsed={elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
