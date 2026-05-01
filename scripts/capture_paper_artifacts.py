"""Post-hoc data capture for paper figures.

Runs on existing checkpoints — no retraining.  For every cell in a sweep
output dir, this produces:

  - `per_frame.json`        — per-rollout-step relL2 + naive-copy baseline
  - `viz_samples.npz`       — 4 sample (input, target, prediction) tuples
  - `kernel_snapshot.npz`   — learned weights + FiLM γ/β extracted from ckpt
  - `residuals.npz`         — per-sample residuals for hardest-decile + correlation analysis

Usage:
    python3 scripts/capture_paper_artifacts.py \\
        --layer_root outputs/dist_kernel_v2_p1 \\
        --data_dir data_dde_pde \\
        [--n_viz_samples 4] [--device cuda]

Skips cells whose `per_frame.json` already exists (idempotent).
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


def load_cell(ckpt_path: Path, data_dir: str, family: str, device: str):
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    _, _, test_loader = create_apebench_dataloaders(
        data_dir, family, batch_size=8, residual_anchor=False, seed=42)
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, test_loader, cfg


def compute_per_frame(model, test_loader, device):
    """Per-rollout-step error metrics averaged over test set + naive-copy baseline.

    Computes THREE complementary metrics per output frame:
      - rel_l2_per_step:   sqrt(sum_test diff_sq / sum_test tgt_sq)  [aggregate-then-sqrt; matches training metric, robust to small targets]
      - mse_per_step:      mean over test set of (diff^2)            [absolute, scale-aware]
      - rel_l2_mean_per_step (legacy): mean(sqrt(diff_sq/tgt_sq))     [Jensen-inflated; kept for backwards compat, but use rel_l2_per_step]
    Plus naive-copy versions for each.
    """
    diff_sq_acc = None  # sum over batch + spatial, shape (T,)
    tgt_sq_acc = None
    naive_diff_sq_acc = None
    rel_legacy_acc = None
    naive_rel_legacy_acc = None
    mask_count_per_step = None  # how many (sample, spatial) cells contribute per t (for averaging mse)
    n_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            yhat = model(x)
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            spatial_dims = tuple(range(2, y.dim()))  # spatial + channel
            B = y.shape[0]
            T = y.shape[1]
            # Per-(B, T) sums over spatial+channel:
            diff_sq_bt = ((yhat - y) ** 2 * mask_bc).sum(dim=spatial_dims)  # (B, T)
            tgt_sq_bt = (y ** 2 * mask_bc).sum(dim=spatial_dims)
            # Naive: predict last-input-frame state channels.
            n_out_ch = y.shape[-1]
            last_in = x[:, -1:, ..., :n_out_ch].expand_as(y)
            naive_diff_sq_bt = ((last_in - y) ** 2 * mask_bc).sum(dim=spatial_dims)
            # Per-(B, T) cell count from mask:
            cell_count_bt = mask_bc.sum(dim=spatial_dims).clamp_min(1.0)  # (B, T)

            # Aggregate sums per-step (over batch dim for now):
            diff_sq_t = diff_sq_bt.sum(dim=0).cpu().numpy()
            tgt_sq_t = tgt_sq_bt.sum(dim=0).cpu().numpy()
            naive_diff_sq_t = naive_diff_sq_bt.sum(dim=0).cpu().numpy()
            cell_count_t = cell_count_bt.sum(dim=0).cpu().numpy()
            # Legacy per-sample relL2 (Jensen-inflated; kept for compat):
            rel_legacy = torch.sqrt(diff_sq_bt / tgt_sq_bt.clamp_min(1e-12))
            naive_rel_legacy = torch.sqrt(naive_diff_sq_bt / tgt_sq_bt.clamp_min(1e-12))
            rel_legacy_t = rel_legacy.sum(dim=0).cpu().numpy()
            naive_rel_legacy_t = naive_rel_legacy.sum(dim=0).cpu().numpy()

            if diff_sq_acc is None:
                diff_sq_acc = diff_sq_t
                tgt_sq_acc = tgt_sq_t
                naive_diff_sq_acc = naive_diff_sq_t
                mask_count_per_step = cell_count_t
                rel_legacy_acc = rel_legacy_t
                naive_rel_legacy_acc = naive_rel_legacy_t
            else:
                diff_sq_acc = diff_sq_acc + diff_sq_t
                tgt_sq_acc = tgt_sq_acc + tgt_sq_t
                naive_diff_sq_acc = naive_diff_sq_acc + naive_diff_sq_t
                mask_count_per_step = mask_count_per_step + cell_count_t
                rel_legacy_acc = rel_legacy_acc + rel_legacy_t
                naive_rel_legacy_acc = naive_rel_legacy_acc + naive_rel_legacy_t
            n_samples += B

    # Aggregate-then-sqrt (correct relL2 per step):
    rel_l2_per_step = np.sqrt(diff_sq_acc / np.clip(tgt_sq_acc, 1e-12, None))
    naive_rel_l2_per_step = np.sqrt(naive_diff_sq_acc / np.clip(tgt_sq_acc, 1e-12, None))
    # Absolute MSE per step (averaged over masked cells):
    mse_per_step = diff_sq_acc / np.clip(mask_count_per_step, 1.0, None)
    naive_mse_per_step = naive_diff_sq_acc / np.clip(mask_count_per_step, 1.0, None)
    # Legacy Jensen-inflated metric for backwards compat:
    rel_l2_mean_per_step = rel_legacy_acc / max(n_samples, 1)
    naive_rel_l2_mean_per_step = naive_rel_legacy_acc / max(n_samples, 1)

    return {
        "rel_l2_per_step": rel_l2_per_step.tolist(),
        "mse_per_step": mse_per_step.tolist(),
        "rel_l2_mean_per_step_legacy": rel_l2_mean_per_step.tolist(),
        "naive_rel_l2_per_step": naive_rel_l2_per_step.tolist(),
        "naive_mse_per_step": naive_mse_per_step.tolist(),
        "naive_rel_l2_mean_per_step_legacy": naive_rel_l2_mean_per_step.tolist(),
    }


def capture_viz_samples(model, test_loader, device, n=4):
    """Save n (input, target, prediction) tuples for visualization."""
    with torch.no_grad():
        batch = next(iter(test_loader))
        x = batch["input"][:n].to(device).float()
        y = batch["target"][:n].to(device).float()
        yhat = model(x)
    return {
        "input":  x.cpu().numpy().astype(np.float32),
        "target": y.cpu().numpy().astype(np.float32),
        "pred":   yhat.cpu().numpy().astype(np.float32),
    }


def capture_kernel_snapshot(ckpt_path: Path):
    """Extract learned LEMO weights + FiLM params from ckpt.  Returns {} for non-LEMO."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    out = {}
    for k, v in sd.items():
        if any(s in k for s in ("weights_time", "weights",  # spectral lag kernel
                                 "film_net", "A_lag",         # FiLM
                                 "A_spat.weights")):           # spatial spectral
            arr = v.detach().cpu().numpy()
            # Complex tensors → store as 2-tuple
            if np.iscomplexobj(arr):
                out[k + "__re"] = arr.real.astype(np.float32)
                out[k + "__im"] = arr.imag.astype(np.float32)
            else:
                out[k] = arr.astype(np.float32)
    return out


def capture_residuals(model, test_loader, device, max_samples=64):
    """Per-sample relL2 + per-sample residual norm (for Jaccard hardest-decile)."""
    per_sample_rel = []
    per_sample_resid = []
    seen = 0
    with torch.no_grad():
        for batch in test_loader:
            if seen >= max_samples:
                break
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            yhat = model(x)
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            diff_sq = ((yhat - y) ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim())))
            tgt_sq = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))) + 1e-12
            rel = torch.sqrt(diff_sq / tgt_sq).cpu().numpy()
            resid = (yhat - y).cpu().numpy().reshape(yhat.shape[0], -1)
            per_sample_rel.append(rel)
            per_sample_resid.append(np.linalg.norm(resid, axis=1))
            seen += yhat.shape[0]
    return {
        "rel_l2_per_sample": np.concatenate(per_sample_rel)[:max_samples],
        "resid_norm_per_sample": np.concatenate(per_sample_resid)[:max_samples],
    }


def process_cell(ckpt_path: Path, data_dir: str, family: str,
                 device: str, n_viz: int):
    cell_dir = ckpt_path.parent
    if (cell_dir / "per_frame.json").exists():
        return "skip (already captured)"
    try:
        model, test_loader, cfg = load_cell(ckpt_path, data_dir, family, device)
    except Exception as e:
        return f"FAIL load ({e})"
    # 1) per-frame (multi-metric: aggregate-then-sqrt + abs MSE + legacy)
    metrics = compute_per_frame(model, test_loader, device)
    (cell_dir / "per_frame.json").write_text(json.dumps(metrics, indent=2))
    # 2) viz samples
    viz = capture_viz_samples(model, test_loader, device, n=n_viz)
    np.savez_compressed(cell_dir / "viz_samples.npz", **viz)
    # 3) kernel snapshot
    snap = capture_kernel_snapshot(ckpt_path)
    if snap:
        np.savez_compressed(cell_dir / "kernel_snapshot.npz", **snap)
    # 4) residuals
    res = capture_residuals(model, test_loader, device)
    np.savez_compressed(cell_dir / "residuals.npz", **res)
    rel = np.array(metrics["rel_l2_per_step"])
    naive = np.array(metrics["naive_rel_l2_per_step"])
    return f"ok (rel_l2_overall={rel.mean():.4f} naive={naive.mean():.4f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_root", required=True, help="e.g. outputs/dist_kernel_v2_p1")
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--n_viz_samples", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    layer = Path(args.layer_root)
    ckpts = sorted(layer.glob("raw/**/best_model.pt"))
    print(f"found {len(ckpts)} checkpoints under {layer}")

    summary = []
    for c in ckpts:
        parts = c.parts
        family = parts[-5]
        cell = "/".join(parts[-5:-1])
        msg = process_cell(c, args.data_dir, family,
                            args.device, args.n_viz_samples)
        print(f"  {cell}: {msg}")
        summary.append({"cell": cell, "result": msg})

    out_path = layer / "capture_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
