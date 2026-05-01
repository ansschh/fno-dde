"""Bulk data-quality audit visualizer.

For each DDE-PDE benchmark in data_dde_pde/, render a grid of N random
train trajectories.  Layout per benchmark:

  Rows     = N samples (default 16, picked uniformly at random)
  Cols     = 6 evenly-spaced time frames per trajectory
            (so we see initial -> final dynamics for each sample)
  Per cell = colormap of u(x, y, t)

Plus a summary panel above:
  - sample-mean trajectory u_bar(t) for each of the N samples (overlay)
  - histogram of per-trajectory final-time spatial std (diversity check)
  - histogram of per-trajectory initial->final relative L2 (dynamics check)

Saves one PNG per benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_benchmark(fam: str, data_root: Path, out_dir: Path,
                     n_samples: int, n_frames: int, seed: int):
    shard = np.load(data_root / fam / "train" / "shard_000.npz")
    phi = shard["phi"]  # (N, n_hist, H, W, C)
    y   = shard["y"]    # (N, n_out, H, W, C)
    full = np.concatenate([phi, y], axis=1)            # (N, T, H, W, C)
    if full.ndim == 5 and full.shape[-1] == 1:
        full = full[..., 0]                            # (N, T, H, W)
    N, T, H, W = full.shape

    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, N)
    sample_idx = rng.choice(N, size=n_samples, replace=False)
    frame_idx = np.linspace(0, T - 1, n_frames, dtype=int)

    samples = full[sample_idx]                         # (n_samples, T, H, W)
    snapshots = samples[:, frame_idx]                  # (n_samples, n_frames, H, W)

    # Diagnostics.
    u_bar = samples.mean(axis=(2, 3))                  # (n_samples, T)
    final_std = samples[:, -1].std(axis=(1, 2))        # (n_samples,)
    init_to_final = np.linalg.norm(
        (samples[:, -1] - samples[:, 0]).reshape(n_samples, -1), axis=1) / (
        np.linalg.norm(samples[:, 0].reshape(n_samples, -1), axis=1) + 1e-12)

    # Layout: top row = diagnostics (3 panels), then n_samples rows of n_frames imshow.
    h_panel = 1.6
    fig = plt.figure(figsize=(2.0 * n_frames, 2.0 + h_panel * n_samples))
    gs = fig.add_gridspec(
        n_samples + 1, n_frames,
        height_ratios=[2.5] + [1.0] * n_samples,
        wspace=0.05, hspace=0.05,
    )

    # Top: split into 3 panels via merge.
    n_diag = 3
    diag_axes = []
    for k in range(n_diag):
        ax = fig.add_subplot(gs[0, n_frames * k // n_diag : n_frames * (k + 1) // n_diag])
        diag_axes.append(ax)
    # u_bar overlay.
    diag_axes[0].plot(u_bar.T, alpha=0.5, lw=0.5)
    diag_axes[0].set_title(f"u_bar(t) for {n_samples} samples")
    diag_axes[0].set_xlabel("t-step"); diag_axes[0].grid(True, alpha=0.3)
    diag_axes[1].hist(final_std, bins=20)
    diag_axes[1].set_title(f"final-time spatial std  (median {np.median(final_std):.3f})")
    diag_axes[2].hist(init_to_final, bins=20)
    diag_axes[2].set_title(
        f"||u(T)-u(0)|| / ||u(0)||  (median {np.median(init_to_final):.3f})")

    # Per-sample image strips.
    vmin, vmax = float(samples.min()), float(samples.max())
    for r in range(n_samples):
        for c in range(n_frames):
            ax = fig.add_subplot(gs[r + 1, c])
            im = ax.imshow(snapshots[r, c], vmin=vmin, vmax=vmax,
                            cmap="viridis", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"#{sample_idx[r]}", fontsize=8)
            if r == 0:
                ax.set_title(f"t={frame_idx[c]}", fontsize=9)

    fig.suptitle(
        f"{fam}  bulk audit: N={n_samples} train samples (of {N})  "
        f"shape ({T},{H},{W})  range [{vmin:.3f}, {vmax:.3f}]")
    out_path = out_dir / f"{fam}_bulk_audit.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return {
        "fam": fam, "N": int(N), "T": int(T), "H": int(H), "W": int(W),
        "range": (float(vmin), float(vmax)),
        "median_final_std": float(np.median(final_std)),
        "median_init_to_final_relL2": float(np.median(init_to_final)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_dde_pde")
    ap.add_argument("--out_dir", default="data_dde_pde/_bulk_audit")
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    fams = sorted([
        d.name for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "train" / "shard_000.npz").exists()
    ])
    print(f"families found: {fams}")
    summary = []
    for fam in fams:
        s = render_benchmark(fam, data_root, out_dir,
                              args.n_samples, args.n_frames, args.seed)
        summary.append(s)
        print(f"   {fam}: range={s['range']}  median_final_std={s['median_final_std']:.4f}"
              f"  median_init_to_final_relL2={s['median_init_to_final_relL2']:.4f}")
    print("\n=== Summary ===")
    print(f"{'family':>22s}  {'samples':>7s}  {'range':>22s}  "
          f"{'final_std':>9s}  {'evol_relL2':>10s}")
    for s in summary:
        print(f"{s['fam']:>22s}  {s['N']:>7d}  "
              f"[{s['range'][0]:>7.3f}, {s['range'][1]:>7.3f}]  "
              f"{s['median_final_std']:>9.4f}  {s['median_init_to_final_relL2']:>10.4f}")


if __name__ == "__main__":
    main()
