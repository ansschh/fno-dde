"""Quick render of random train trajectories from each DDE-PDE benchmark.

For each benchmark with a generated train shard, picks 3 random
trajectories, renders 8 evenly-spaced time snapshots into a PNG strip.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_traj(traj: np.ndarray, title: str, out_path: Path, n_frames: int = 8):
    """traj: (T, H, W, 1) or (T, H, W).  Renders strip of n_frames snapshots."""
    if traj.ndim == 4:
        traj = traj[..., 0]
    T = traj.shape[0]
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2.0, 2.4),
                              constrained_layout=True)
    vmin, vmax = traj.min(), traj.max()
    for ax, k in zip(axes, idxs):
        im = ax.imshow(traj[k], vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
        ax.set_title(f"t={k}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{title}  range=[{vmin:.3f}, {vmax:.3f}]")
    fig.colorbar(im, ax=axes, fraction=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_dde_pde")
    ap.add_argument("--out_dir", default="data_dde_pde/_viz")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)

    families = sorted([d.name for d in data_root.iterdir()
                       if d.is_dir() and not d.name.startswith("_")
                       and (d / "train" / "shard_000.npz").exists()])

    for fam in families:
        d = np.load(data_root / fam / "train" / "shard_000.npz")
        phi = d["phi"]; y = d["y"]
        # Concatenate history + future.
        full = np.concatenate([phi, y], axis=1)  # (N, T, H, W, C)
        N = full.shape[0]
        idxs = rng.choice(N, size=min(args.n_samples, N), replace=False)
        for i, idx in enumerate(idxs):
            traj = full[idx]
            render_traj(traj, title=f"{fam} train sample {idx}",
                         out_path=out_root / fam / f"train_sample_{i}_idx{idx}.png")
            print(f"  wrote {out_root / fam / f'train_sample_{i}_idx{idx}.png'}")


if __name__ == "__main__":
    main()
