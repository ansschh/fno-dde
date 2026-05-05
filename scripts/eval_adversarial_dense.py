"""Dense-epsilon FGSM eval across ALL local checkpoints.

Mirrors the structure of `eval_equivariance_dense.py`:
  - Denser ε grid: ε in {0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1}
  - Crawls multiple sweep roots in one invocation
  - Writes per-cell `adversarial_dense.json` next to each `best_model.pt`
  - Skips cells whose dense file already exists (resumable)
  - Optional `--families` / `--models` filters

FGSM: x_perturbed = x + eps * sign(grad of loss wrt x_state). Perturbs only
state channels (not time/mask/params), matching post_hoc_analyses.py's
`adversarial_fgsm`. Records per-eps mean and std rel-L2.
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

DENSE_EPS_DEFAULT = "0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1"


def fgsm_eval(model, test_loader, device, epsilons, n_batches=16):
    """Returns {eps -> {mean, std, n}}. Perturbs state channels only."""
    out = {float(e): [] for e in epsilons}
    seen_batches = 0
    for batch in test_loader:
        if seen_batches >= n_batches:
            break
        x = batch["input"].to(device).float()
        y = batch["target"].to(device).float()
        mask = batch["loss_mask"].to(device).float()
        n_spatial = y.dim() - 3
        mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
        n_state_ch = y.shape[-1]
        x_state = x[..., :n_state_ch].detach().clone().requires_grad_(True)
        x_aux = x[..., n_state_ch:].detach()
        x_full = torch.cat([x_state, x_aux], dim=-1)
        yhat = model(x_full)
        diff_sq = ((yhat - y) ** 2 * mask_bc).sum()
        grad = None
        for eps in epsilons:
            eps = float(eps)
            if eps > 0:
                if grad is None:
                    grad = torch.autograd.grad(diff_sq, x_state, retain_graph=True)[0]
                x_pert = (x_state + eps * grad.sign()).detach()
            else:
                x_pert = x_state.detach()
            x_full_p = torch.cat([x_pert, x_aux], dim=-1)
            with torch.no_grad():
                yhat_p = model(x_full_p)
            num = ((yhat_p - y) ** 2 * mask_bc).sum(dim=tuple(range(1, yhat_p.dim()))).sqrt()
            den = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))).sqrt().clamp_min(1e-12)
            rel = (num / den).cpu().tolist()
            out[eps].extend(rel)
        seen_batches += 1
    return {f"{e:g}": {"mean": float(np.mean(v)) if v else float("nan"),
                        "std":  float(np.std(v))  if v else float("nan"),
                        "n":    len(v)}
            for e, v in out.items()}


def evaluate_checkpoint(ckpt_path: Path, data_dir: str, epsilons, device, n_batches: int):
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
    res = fgsm_eval(model, test_loader, device, epsilons, n_batches=n_batches)
    return {"family": family, "regime": regime, "model": model_name, "seed": seed,
            "epsilons": [float(e) for e in epsilons], "fgsm": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None,
                    help="Sweep roots to crawl. Required unless --shard given.")
    ap.add_argument("--shard", default=None,
                    help="Optional newline-separated list of best_model.pt paths.")
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--epsilons", default=DENSE_EPS_DEFAULT)
    ap.add_argument("--families", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--regimes", default="clean,lowres,noisy")
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    epsilons = [float(s) for s in args.epsilons.split(",")]
    families_filter = set(args.families.split(",")) if args.families else None
    models_filter = set(args.models.split(",")) if args.models else None
    regimes_filter = set(args.regimes.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[adv-dense] device={device} epsilons={epsilons} regimes={regimes_filter}")

    ckpts = []
    if args.shard:
        ckpts = [Path(p) for p in Path(args.shard).read_text().splitlines() if p.strip()]
        print(f"[adv-dense] using shard with {len(ckpts)} checkpoints from {args.shard}")
    else:
        if not args.roots:
            print("[adv-dense] ERROR: must provide --roots or --shard")
            return
        for root in args.roots:
            rp = Path(root)
            ckpts.extend(sorted(rp.rglob("best_model.pt")))
        print(f"[adv-dense] found {len(ckpts)} candidate checkpoints across {len(args.roots)} roots")

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
        out_path = c.parent / "adversarial_dense.json"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue
        try:
            t1 = time.time()
            r = evaluate_checkpoint(c, args.data_dir, epsilons, device, args.n_batches)
            json.dump(r, open(out_path, "w"), indent=2)
            n_done += 1
            dt = time.time() - t1
            mn = {f"{e:g}": f"{r['fgsm'][f'{e:g}']['mean']:.3e}" for e in epsilons}
            print(f"  [{n_done:3d}/{n_total}] {family}/{regime}/{model_name}/{seed} "
                  f"({dt:.1f}s) eps_means={mn}")
        except Exception as ex:
            n_failed += 1
            print(f"  FAILED {family}/{regime}/{model_name}/{seed}: {ex}")
    elapsed = time.time() - t0
    print(f"[adv-dense] done={n_done} skipped={n_skipped} failed={n_failed} "
          f"total_seen={n_total} elapsed={elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
