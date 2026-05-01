"""Generate GIFs for DDE-PDE benchmarks.

Two flavors per benchmark:
  (A) trajectory-only GIF — animate u(t) for a few test samples
  (B) predicted-vs-ground-truth GIF — load LEMO_PC_ND checkpoint from
      Layer 4 audit, predict on a test sample, animate
      [ground truth | prediction | abs error] side-by-side

Outputs to {out_dir}/{family}/.
Requires matplotlib + Pillow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from datasets.apebench_dataset import create_apebench_dataloaders
from train.build_model import build_model


def make_trajectory_gif(traj: np.ndarray, out_path: Path, title: str,
                         fps: int = 10):
    """traj: (T, H, W) — animate as imshow.  Saves a GIF."""
    T = traj.shape[0]
    vmin, vmax = float(traj.min()), float(traj.max())
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(traj[0], vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    title_text = ax.set_title(f"{title}  t=0/{T-1}")
    fig.colorbar(im, ax=ax, fraction=0.046)

    def update(t):
        im.set_data(traj[t])
        title_text.set_text(f"{title}  t={t}/{T-1}")
        return im, title_text

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_pred_gt_gif(gt: np.ndarray, pred: np.ndarray, out_path: Path,
                      title: str, n_hist: int, fps: int = 10):
    """gt, pred: each (T, H, W).  Side-by-side animation."""
    T = gt.shape[0]
    err = np.abs(gt - pred)
    vmin = min(gt.min(), pred.min()); vmax = max(gt.max(), pred.max())
    e_max = float(err.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    im_gt = axes[0].imshow(gt[0], vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
    im_pr = axes[1].imshow(pred[0], vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
    im_er = axes[2].imshow(err[0], vmin=0, vmax=e_max, cmap="magma", origin="lower")
    for ax, lbl in zip(axes, ["ground truth", "prediction", "|error|"]):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lbl)
    fig.colorbar(im_gt, ax=axes[0], fraction=0.04)
    fig.colorbar(im_pr, ax=axes[1], fraction=0.04)
    fig.colorbar(im_er, ax=axes[2], fraction=0.04)
    sup = fig.suptitle(f"{title}  t=0/{T-1}  (history t<{n_hist}, future t>={n_hist})")

    def update(t):
        im_gt.set_data(gt[t])
        im_pr.set_data(pred[t])
        im_er.set_data(err[t])
        flag = " (HISTORY)" if t < n_hist else " (FUTURE)"
        sup.set_text(f"{title}  t={t}/{T-1}{flag}")
        return im_gt, im_pr, im_er, sup

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def load_lemo_checkpoint(layer4_root: Path, family: str, device: str = "cuda"):
    """Find and load LEMO_PC_ND checkpoint from Layer 4 audit results.

    Returns (model, config) or (None, None) if not found.
    """
    ckpt_paths = sorted(layer4_root.glob(
        f"raw/{family}/clean/lemo_pc_nd/s*/best_model.pt"))
    if not ckpt_paths:
        ckpt_paths = sorted(layer4_root.glob(
            f"{family}/clean/lemo_pc_nd/s*/best_model.pt"))
    if not ckpt_paths:
        return None, None
    ckpt = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
    cfg = ckpt["config"]
    # Build matching architecture
    # Need in_channels and out_channels — read from data
    from datasets.apebench_dataset import APEBenchDataset
    ds = APEBenchDataset("data_dde_pde", family, "test")
    sample = ds[0]
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[0]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_dde_pde")
    ap.add_argument("--layer4_root", default="outputs/layer4_audit")
    ap.add_argument("--out_dir", default="data_dde_pde/_gifs")
    ap.add_argument("--n_traj_gifs", type=int, default=3,
                    help="Trajectory-only GIFs per benchmark.")
    ap.add_argument("--n_pred_gifs", type=int, default=2,
                    help="Pred-vs-GT GIFs per benchmark.")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root); out_dir = Path(args.out_dir)
    layer4_root = Path(args.layer4_root)

    fams = sorted([
        d.name for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "test" / "shard_000.npz").exists()
    ])
    print(f"families found: {fams}")

    for fam in fams:
        print(f"\n=== {fam} ===")
        # Trajectory GIFs from test split.
        d = np.load(data_root / fam / "test" / "shard_000.npz")
        phi = d["phi"]; y = d["y"]
        full = np.concatenate([phi, y], axis=1)
        if full.shape[-1] == 1:
            full = full[..., 0]                    # (N, T, H, W)
        N, T, H, W = full.shape
        n_hist = phi.shape[1]
        idxs = rng.choice(N, size=min(args.n_traj_gifs, N), replace=False)
        for i, idx in enumerate(idxs):
            out_path = out_dir / fam / f"traj_{i}_idx{idx}.gif"
            print(f"  -> {out_path}")
            make_trajectory_gif(full[idx], out_path,
                                 title=f"{fam} test #{idx}",
                                 fps=args.fps)

        # Predicted-vs-truth GIFs.
        model, cfg = load_lemo_checkpoint(layer4_root, fam, device=device)
        if model is None:
            print(f"  no LEMO_PC_ND checkpoint found for {fam}, skipping pred GIFs")
            continue
        # Use the standard test loader so input has proper aux channels.
        ra = bool(cfg.get("residual_anchor", False)) if isinstance(cfg, dict) else True
        _, _, test_loader = create_apebench_dataloaders(
            args.data_root, fam, batch_size=4, residual_anchor=ra, seed=args.seed)
        # Fetch one batch.
        with torch.no_grad():
            for batch in test_loader:
                x = batch["input"].to(device).float()
                y_target = batch["target"].to(device).float()
                y_pred = model(x).cpu().numpy()
                y_target = y_target.cpu().numpy()
                break
        # y_pred / y_target shape: (B, T, H, W, C); pick C=0
        if y_pred.shape[-1] == 1:
            y_pred = y_pred[..., 0]
            y_target = y_target[..., 0]
        for i in range(min(args.n_pred_gifs, y_pred.shape[0])):
            out_path = out_dir / fam / f"pred_vs_gt_{i}.gif"
            print(f"  -> {out_path}")
            make_pred_gt_gif(y_target[i], y_pred[i], out_path,
                              title=f"{fam} test #{i}: LEMO_PC_ND",
                              n_hist=n_hist, fps=args.fps)


if __name__ == "__main__":
    main()
