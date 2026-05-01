"""3D bulk audit visualization.

For 3D PDE benchmarks (shape N x T x Z x Y x X x C), render trajectories
as central-slice strips: 8 time frames, each showing the central XY slice
of channel 0 (and channel 1 if present, channel 2 if present).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_3d_traj_slices(traj: np.ndarray, out_path: Path, title: str,
                           n_frames: int = 8):
    """traj: (T, Z, Y, X, C).  Renders 3-row strip:
       row 0 = central XY slice (mid-z) per time
       row 1 = central XZ slice (mid-y)
       row 2 = central YZ slice (mid-x)
    Channels are averaged for visualization (or first channel for clarity).
    """
    T, Z, Y, X, C = traj.shape
    # Use channel-0 (or channel-norm if multi-channel)
    if C > 1:
        # Show RMS over channels
        u = np.sqrt((traj.astype(np.float32) ** 2).sum(axis=-1))   # (T, Z, Y, X)
    else:
        u = traj[..., 0].astype(np.float32)
    z_mid, y_mid, x_mid = Z // 2, Y // 2, X // 2
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(3, n_frames, figsize=(n_frames * 1.8, 5.4),
                              constrained_layout=True)
    vmin, vmax = float(u.min()), float(u.max())
    for ax_row, name, slc_fn in [
            (axes[0], f"XY @ z={z_mid}", lambda t: u[t, z_mid, :, :]),
            (axes[1], f"XZ @ y={y_mid}", lambda t: u[t, :, y_mid, :]),
            (axes[2], f"YZ @ x={x_mid}", lambda t: u[t, :, :, x_mid]),
        ]:
        for ax, k in zip(ax_row, idxs):
            im = ax.imshow(slc_fn(k), vmin=vmin, vmax=vmax,
                            cmap="viridis", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
        ax_row[0].set_ylabel(name, fontsize=10)
    for ax, k in zip(axes[0], idxs):
        ax.set_title(f"t={k}", fontsize=9)
    fig.suptitle(f"{title}  range=[{vmin:.3f}, {vmax:.3f}]")
    fig.colorbar(im, ax=axes, fraction=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_apebench")
    ap.add_argument("--out_dir", default="data_apebench/_viz3d")
    ap.add_argument("--families", default="burgers_3d,gray_scott_3d")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    data_root = Path(args.data_root); out_root = Path(args.out_dir)

    for fam in args.families.split(","):
        shard = np.load(data_root / fam / "train" / "shard_000.npz")
        phi = shard["phi"]; y = shard["y"]
        full = np.concatenate([phi, y], axis=1)            # (N, T, Z, Y, X, C)
        N = full.shape[0]
        idxs = rng.choice(N, size=min(args.n_samples, N), replace=False)
        for i, idx in enumerate(idxs):
            out_path = out_root / fam / f"slices_sample_{i}_idx{idx}.png"
            print(f"  {out_path}")
            render_3d_traj_slices(full[idx], out_path,
                                    title=f"{fam} train #{idx}")


if __name__ == "__main__":
    main()
