"""Interactive 3D volume + isosurface visualization with plotly.

For each 3D sample (T, Z, Y, X, C) saved as .npy, render:
  - One time-slider HTML with Volume rendering
  - One isosurface HTML at multiple time frames

Saves to interactive HTML files you can open in a browser and rotate/zoom.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def make_volume_html(traj: np.ndarray, out_path: Path, title: str,
                      stride: int = 4):
    """traj: (T, Z, Y, X) — make a Volume frame per time, slider to scrub time.

    stride: time-step subsampling for animation (every N-th frame).
    """
    T, Z, Y, X = traj.shape
    # Build coordinate grids.
    z, y, x = np.mgrid[0:Z, 0:Y, 0:X]
    z = z.flatten(); y = y.flatten(); x = x.flatten()
    # Subsample time.
    frames_idx = list(range(0, T, stride))
    if frames_idx[-1] != T - 1:
        frames_idx.append(T - 1)
    vmin, vmax = float(traj.min()), float(traj.max())

    # Build initial volume frame.
    def vol_frame(k):
        return go.Volume(
            x=x, y=y, z=z,
            value=traj[k].flatten(),
            isomin=vmin + (vmax - vmin) * 0.10,
            isomax=vmax,
            opacity=0.18,
            surface_count=15,
            colorscale="Viridis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )

    fig = go.Figure(data=[vol_frame(frames_idx[0])])
    fig.update(frames=[
        go.Frame(data=[vol_frame(k)], name=str(k)) for k in frames_idx
    ])

    sliders = [dict(
        active=0,
        steps=[dict(method="animate",
                    label=f"t={k}",
                    args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
               for k in frames_idx],
    )]
    fig.update_layout(
        title=title,
        sliders=sliders,
        scene=dict(
            xaxis=dict(range=[0, X]), yaxis=dict(range=[0, Y]), zaxis=dict(range=[0, Z]),
            aspectmode="cube",
        ),
        width=900, height=700,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def make_isosurface_html(traj: np.ndarray, out_path: Path, title: str,
                          n_frames: int = 6):
    """traj: (T, Z, Y, X) — single HTML with N isosurfaces side-by-side.

    Different time frames as subplots in a 2-row grid.
    """
    from plotly.subplots import make_subplots
    T, Z, Y, X = traj.shape
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    # Layout: 2 rows × 3 cols (or n_frames-aware).
    cols = min(n_frames, 3)
    rows = (n_frames + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                          specs=[[{"type": "scene"}] * cols] * rows,
                          subplot_titles=[f"t={k}" for k in idxs])
    z, y, x = np.mgrid[0:Z, 0:Y, 0:X]
    z = z.flatten(); y = y.flatten(); x = x.flatten()
    vmin, vmax = float(traj.min()), float(traj.max())
    # Two iso levels: a low one and a high one.
    iso_lo = vmin + (vmax - vmin) * 0.30
    iso_hi = vmin + (vmax - vmin) * 0.70

    for i, k in enumerate(idxs):
        r = i // cols + 1; c = i % cols + 1
        fig.add_trace(go.Isosurface(
            x=x, y=y, z=z, value=traj[k].flatten(),
            isomin=iso_lo, isomax=iso_hi,
            opacity=0.4, surface_count=3,
            colorscale="Viridis",
            showscale=(i == 0),
            caps=dict(x_show=False, y_show=False, z_show=False),
        ), row=r, col=c)

    fig.update_layout(title=title, width=1400, height=900)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", default="data_apebench/_3d_samples")
    ap.add_argument("--out_dir", default="data_apebench/_3d_html")
    args = ap.parse_args()
    samples = sorted(Path(args.samples_dir).glob("*.npy"))
    print(f"found {len(samples)} samples")
    for p in samples:
        traj = np.load(p)
        # Reduce to single channel via RMS over channel axis if multi-channel.
        if traj.ndim == 5:
            traj_disp = np.sqrt((traj.astype(np.float32) ** 2).sum(axis=-1))
        else:
            traj_disp = traj
        name = p.stem
        out_iso = Path(args.out_dir) / f"{name}_isosurface.html"
        make_isosurface_html(traj_disp, out_iso, title=f"{name} isosurface", n_frames=6)
        print(f"  iso  {out_iso}")
        out_vol = Path(args.out_dir) / f"{name}_volume.html"
        make_volume_html(traj_disp, out_vol, title=f"{name} volume + time slider", stride=4)
        print(f"  vol  {out_vol}")


if __name__ == "__main__":
    main()
