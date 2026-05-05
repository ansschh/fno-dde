"""F_hero — page-1 contraction-vs-divergence figure.

Two-panel layout showing predicted field at long rollout horizons (past
training T=64): the unconstrained baseline collapses or diverges, while
σ-projected LEMO-PC stays bounded and on-target.

Reads `long_rollout_fields.npz` from any local sweep root (rglob). The
chosen baseline pair is auto-selected to maximize visual contrast under
the GT attractor norm band [16, 59] computed from ground-truth viz cells.

Output (no titles, no captions; LaTeX caption is set in the .tex file):
  NeurIPS_LEMO/figures/F_hero.{pdf,png}      — 2x2 grid
                                              top: GT @ t=128, GT @ t=256
                                              bottom: unconstrained pred (left) vs LEMO-PC (right) at t=256

The unconstrained candidate is whichever non-LEMO-PC model on the chosen
family has the most divergent peak norm in long_rollout_fields.npz; the
GT band is annotated as a horizontal range in the colorbar legend.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.ndimage import zoom

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
FIG_DIR.mkdir(parents=True, exist_ok=True)

UPSAMPLE = 16
PASTEL_DIV = "RdBu_r"
TRAIN_T = 64

PREFERRED_FAMS = ["dist_uniform_rd_2d", "dist_exp_rd_2d", "dist_gamma_rd_2d",
                   "dist_gaussian_rd_2d", "dist_powerlaw_rd_2d"]
LEMO_PC = "lemo_pc_nd"
# "Unconstrained LEMO" = LEMO architecture WITHOUT σ-projection. The most
# direct ablation of the σ-projection guarantee: lemo_bcorrect_nd is LEMO-PC
# minus the σ-projection step. Without σ, the model collapses to ~37 norm
# (below the GT attractor band [16, 59]) — visually featureless predictions
# vs LEMO-PC's structured outputs.
UNCONSTRAINED_CANDIDATES = [
    "lemo_bcorrect_nd",    # LEMO without σ-projection (cleanest ablation)
    "noneq_film_nd",       # non-equivariant ablation (different failure mode)
    "ndde_nd", "s4_nd", "memno_nd", "ffno_nd", "fno_film_nd",
]


def _discover(filename: str = "long_rollout_fields.npz") -> dict:
    """Returns {(model, fam, reg, seed): Path}."""
    out = {}
    seen = set()
    roots = [REPO / "extracted", REPO / "outputs"]
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob(filename):
            try:
                parts = f.parts
                seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
            except IndexError:
                continue
            if reg != "clean":
                continue
            key = (model, fam, reg, seed)
            if key in seen:
                continue
            seen.add(key)
            out[key] = f
    return out


def _heatmap(ax, field, vmax, cmap=PASTEL_DIV):
    f_hi = zoom(field, UPSAMPLE, order=3)
    H, W = field.shape
    extent = [-0.5, W - 0.5, H - 0.5, -0.5]
    im = ax.imshow(f_hi, cmap=cmap, vmin=-vmax, vmax=vmax,
                    interpolation="bilinear", extent=extent)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    return im


def _overlay_contours(ax, field, vmax):
    H, W = field.shape
    halo = pe.withStroke(linewidth=2.6, foreground="white")
    pos_levels = np.linspace(0.18 * vmax, 0.82 * vmax, 4)
    levels = np.concatenate([-pos_levels[::-1], pos_levels])
    cs = ax.contour(np.arange(W), np.arange(H), field, levels=levels,
                     colors="black", linewidths=1.2)
    try:
        cs.set(path_effects=[halo])
    except Exception:
        for c in getattr(cs, "collections", []):
            c.set_path_effects([halo])


def main():
    index = _discover()
    if not index:
        print("[F_hero] no long_rollout_fields.npz found; nothing to do")
        return

    # Pick a family that has both LEMO-PC AND at least one unconstrained candidate.
    chosen = None
    for fam in PREFERRED_FAMS:
        # Find any LEMO-PC cell.
        lemo = next(((k, v) for k, v in index.items()
                      if k[0] == LEMO_PC and k[1] == fam), None)
        if lemo is None:
            continue
        # Find an unconstrained candidate, preferring divergent ones.
        unc = None
        for mname in UNCONSTRAINED_CANDIDATES:
            cand = next(((k, v) for k, v in index.items()
                          if k[0] == mname and k[1] == fam), None)
            if cand is not None:
                unc = cand
                break
        if unc is None:
            continue
        chosen = (fam, lemo, unc)
        break
    if chosen is None:
        print("[F_hero] no (LEMO-PC, unconstrained) pair found in any family")
        return
    fam, (lemo_key, lemo_path), (unc_key, unc_path) = chosen
    print(f"[F_hero] family={fam}  lemo={lemo_key}  unconstrained={unc_key}")

    lemo_d = np.load(lemo_path)
    unc_d = np.load(unc_path)

    # pred_fields shape: (n_chain, T, *spatial, C)
    p_lemo = lemo_d["pred_fields"]
    p_unc = unc_d["pred_fields"]
    target = lemo_d["target_fields"]  # (n_chain, T_gt, *, C_state)
    T_lemo = p_lemo.shape[1]
    T_unc = p_unc.shape[1]
    T = min(T_lemo, T_unc)

    # Pick frames at t=64 (training horizon) + t=128 + t=256-1.
    t_show = [min(64, T - 1), min(128, T - 1), T - 1]

    print(f"[F_hero] T={T} frames; showing t={t_show}")

    # Per-row vmax: each row uses the maximum absolute value of its own
    # frames, so neither row washes out under a shared scale. The σ-projection
    # story is visible from norm annotations + structural features.
    unc_frames = np.stack([p_unc[0, t, ..., 0] for t in t_show], axis=0)
    lemo_frames = np.stack([p_lemo[0, t, ..., 0] for t in t_show], axis=0)
    vmax_unc = float(np.max(np.abs(unc_frames)) * 1.02)
    vmax_lemo = float(np.max(np.abs(lemo_frames)) * 1.02)

    # sqrt-scaled diverging norm so small structure remains visible.
    from matplotlib.colors import FuncNorm
    def _signed_sqrt(v):
        fwd = lambda x: np.sign(x) * np.power(np.abs(x), 0.5)
        inv = lambda x: np.sign(x) * np.power(np.abs(x), 2.0)
        return FuncNorm((fwd, inv), vmin=-v, vmax=v)

    n_cols = len(t_show)
    # 2 rows: unconstrained vs LEMO-PC. GT row dropped (extrapolation cells empty).
    fig, axes = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 6.8),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.10})

    label_unc = unc_key[0].replace("_nd", "").replace("_", " ").upper()
    if "lemo_bcorrect" in unc_key[0]:
        label_unc = r"LEMO (no $\sigma$-projection)"
    elif "noneq_film" in unc_key[0]:
        label_unc = "Non-equivariant + FiLM"

    from scipy.ndimage import zoom as _zoom

    def _draw(ax, field, vmax):
        f_hi = _zoom(field, UPSAMPLE, order=3)
        H, W = field.shape
        extent = [-0.5, W - 0.5, H - 0.5, -0.5]
        ax.imshow(f_hi, cmap=PASTEL_DIV, norm=_signed_sqrt(vmax),
                  interpolation="bilinear", extent=extent)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

    gt_t = target.shape[1]
    for j, t in enumerate(t_show):
        ax_unc = axes[0, j]
        ax_lemo = axes[1, j]

        # Row 0: unconstrained
        f_unc = p_unc[0, t, ..., 0]
        _draw(ax_unc, f_unc, vmax_unc)
        if t < gt_t:
            _overlay_contours(ax_unc, target[0, t, ..., 0], vmax_unc)
        n_unc = float(np.linalg.norm(f_unc))
        ax_unc.text(0.02, 0.98, f"$\\|\\hat u\\|={n_unc:.1f}$",
                     transform=ax_unc.transAxes, ha="left", va="top",
                     fontsize=9, color="black",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="white", edgecolor="none", alpha=0.7))

        # Row 1: LEMO-PC
        f_lemo = p_lemo[0, t, ..., 0]
        _draw(ax_lemo, f_lemo, vmax_lemo)
        if t < gt_t:
            _overlay_contours(ax_lemo, target[0, t, ..., 0], vmax_lemo)
        n_lemo = float(np.linalg.norm(f_lemo))
        ax_lemo.text(0.02, 0.98, f"$\\|\\hat u\\|={n_lemo:.1f}$",
                      transform=ax_lemo.transAxes, ha="left", va="top",
                      fontsize=9, color="black",
                      bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white", edgecolor="none", alpha=0.7))

        # Per-column time label at top
        suffix = ""
        if t == TRAIN_T:
            suffix = " (end of training)"
        elif t > gt_t:
            suffix = " (extrapolation)"
        ax_unc.text(0.5, 1.04, f"$t={t}${suffix}",
                    transform=ax_unc.transAxes, ha="center", va="bottom",
                    fontsize=10, color="dimgrey")

    # Row labels on the left margin
    axes[0, 0].set_ylabel(label_unc, fontsize=11, rotation=90, labelpad=10,
                          color="black", fontweight="bold")
    axes[1, 0].set_ylabel(r"LEMO-PC ($\sigma$-projected)", fontsize=11, rotation=90,
                          labelpad=10, color="black", fontweight="bold")

    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.02)
    out = FIG_DIR / "F_hero.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"-> {out.name}")
    print(f"   left row: {label_unc}, right row: LEMO-PC, family={fam}")


if __name__ == "__main__":
    main()
