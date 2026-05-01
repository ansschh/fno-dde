"""Lag-distribution OOD evaluation.

For each (dataset, model, seed), evaluate the trained checkpoint on a
test set generated with tau values OUTSIDE the training range.

Training tau range was {1.0, 1.5, 2.0} for most benchmarks; OOD tau
will be {0.5, 2.5, 3.0} (smaller and larger than training).

This is a fast post-hoc eval — no retraining, just inference on a
freshly-generated OOD test shard.

Usage:
    python3 scripts/eval_lag_shift_ood.py \\
        --layer5_root outputs/layer5_final_sweep_p1 \\
        --ood_data_dir data_dde_pde_ood
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


def evaluate_one_cell(ckpt_path: Path, ood_data_dir: str, family: str,
                       device: str = "cuda") -> dict:
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    _, _, test_loader = create_apebench_dataloaders(
        ood_data_dir, family, batch_size=8, residual_anchor=False, seed=42)
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    rel_l2_list = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device).float()
            y_target = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            y_pred = model(x)
            n_spatial = y_target.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            diff = (y_pred - y_target) * mask_bc
            tgt = y_target * mask_bc
            num = torch.sqrt((diff ** 2).sum(dim=tuple(range(1, diff.dim()))) + 1e-10)
            den = torch.sqrt((tgt ** 2).sum(dim=tuple(range(1, tgt.dim()))) + 1e-10)
            rel = num / den
            rel_l2_list.append(rel.cpu().numpy())
    return {"test_rel_l2_ood": float(np.mean(np.concatenate(rel_l2_list))),
            "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer5_root", required=True,
                    help="e.g. outputs/layer5_final_sweep_p1")
    ap.add_argument("--ood_data_dir", default="data_dde_pde_ood")
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    layer5 = Path(args.layer5_root)
    ckpts = sorted(layer5.glob("raw/**/best_model.pt"))
    print(f"found {len(ckpts)} checkpoints under {layer5}")
    results = {}
    for c in ckpts:
        parts = c.parts
        # raw/{family}/{regime}/{model}/s{seed}/best_model.pt
        family = parts[-5]; regime = parts[-4]; model_name = parts[-3]; seed = parts[-2]
        if regime != "clean":   # only OOD-eval on clean training cells
            continue
        try:
            r = evaluate_one_cell(c, args.ood_data_dir, family)
            r.update(family=family, model=model_name, seed=seed)
            results[(family, model_name, seed)] = r
            print(f"  {family}/{model_name}/{seed}: ood_relL2 = {r['test_rel_l2_ood']:.4f}")
        except Exception as e:
            print(f"  {family}/{model_name}/{seed}: FAILED ({e})")

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({f"{k[0]}__{k[1]}__{k[2]}": v for k, v in results.items()},
              open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
