"""Prototype 3 layouts for the scalar-DDE baseline figure (hutchinson/MG/wright).

Generates:
  F06b_scalar_optionA.png  — 3-row x 3-col grid (regime x family)
  F06b_scalar_optionB.png  — 1 row x 3 cols (one per family); linestyle = regime
  F06b_scalar_optionC.png  — 1 row x 3 cols (one per regime); families averaged

All variants share the same 6 models with consistent color coding.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Global Times New Roman style for all paper figures.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _figstyle  # noqa: F401  (sets Times New Roman globally)

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE = REPO / "extracted" / "pod1" / "outputs" / "layer5_final_sweep_p1" / "raw"

FAMS = ["hutchinson_2d", "mackey_glass_2d", "wright_2d"]
FAM_LABELS = {"hutchinson_2d": "Hutchinson", "mackey_glass_2d": "Mackey-Glass",
              "wright_2d": "Wright"}
REGIMES = ["clean", "lowres", "noisy"]
REGIME_LABELS = {"clean": "Clean", "lowres": "Low-res", "noisy": "Noisy"}
REGIME_LS = {"clean": "-", "lowres": "--", "noisy": ":"}

MODEL_ORDER = ["lemo_pc_nd", "lemo_nd", "fno_nd", "markov_fno_nd",
               "windowed_fno_nd", "unet_nd"]
MODEL_LABELS = {"lemo_pc_nd": "LEMO-PC", "lemo_nd": "LEMO",
                "fno_nd": "FNO", "markov_fno_nd": "Markov-FNO",
                "windowed_fno_nd": "Window-FNO", "unet_nd": "UNet"}
MODEL_COLOR = {"lemo_pc_nd": "#d62728", "lemo_nd": "#ff7f0e",
               "fno_nd": "#1f77b4", "markov_fno_nd": "#2ca02c",
               "windowed_fno_nd": "#9467bd", "unet_nd": "#8c564b"}


def _trim_history_zeros(arr: np.ndarray) -> np.ndarray:
    """Drop the leading history block where rel_l2 is exactly 0 (no forecast yet)."""
    if arr.ndim == 1:
        nz = np.flatnonzero(arr > 0)
        return arr[nz[0]:] if nz.size else arr
    # 2D: [n_seeds, T]; trim where ALL seeds are zero
    col_any = (arr > 0).any(axis=0)
    nz = np.flatnonzero(col_any)
    return arr[:, nz[0]:] if nz.size else arr


def load_curves(fam: str, reg: str, model: str) -> np.ndarray | None:
    """Return stacked per-step relL2 curves [n_seeds, T_forecast] or None."""
    d = BASE / fam / reg / model
    if not d.exists():
        return None
    curves = []
    for sd in sorted(d.glob("s*")):
        f = sd / "per_frame.json"
        if not f.exists():
            continue
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        r = j.get("rel_l2_per_step", [])
        if r:
            curves.append(np.array(r, dtype=float))
    if not curves:
        return None
    L = min(len(c) for c in curves)
    arr = np.stack([c[:L] for c in curves], axis=0)
    return _trim_history_zeros(arr)


def naive_curve(fam: str, reg: str) -> np.ndarray | None:
    """Use naive copy per-step from the first model with data.
    Layer5 saves it as `naive_copy_per_step`; older layers used `naive_rel_l2_per_step`.
    """
    for model in MODEL_ORDER:
        d = BASE / fam / reg / model
        if not d.exists():
            continue
        for sd in sorted(d.glob("s*")):
            f = sd / "per_frame.json"
            if not f.exists():
                continue
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            n = j.get("naive_copy_per_step") or j.get("naive_rel_l2_per_step")
            if n:
                return _trim_history_zeros(np.array(n, dtype=float))
    return None


def plot_panel(ax, fam: str, reg: str, *, show_naive=True, show_band=True,
               linestyle="-", legend_set=None):
    """Plot all models for (fam, reg) on ax. Returns dict of label->handle."""
    handles = {}
    for model in MODEL_ORDER:
        c = load_curves(fam, reg, model)
        if c is None:
            continue
        steps = np.arange(c.shape[1])
        m = c.mean(axis=0)
        color = MODEL_COLOR[model]
        line, = ax.plot(steps, m, color=color, lw=1.2,
                        linestyle=linestyle, label=MODEL_LABELS[model])
        if show_band and c.shape[0] > 1:
            ax.fill_between(steps, m - c.std(axis=0), m + c.std(axis=0),
                            color=color, alpha=0.15, linewidth=0)
        if legend_set is None or MODEL_LABELS[model] not in legend_set:
            handles[MODEL_LABELS[model]] = line
            if legend_set is not None:
                legend_set.add(MODEL_LABELS[model])
    if show_naive:
        n = naive_curve(fam, reg)
        if n is not None:
            line, = ax.plot(np.arange(len(n)), n, color="grey", lw=1.0,
                            linestyle="--", label="naive copy")
            if legend_set is None or "naive copy" not in legend_set:
                handles["naive copy"] = line
                if legend_set is not None:
                    legend_set.add("naive copy")
    return handles


def option_A():
    """3 rows (regime) x 3 cols (family). Each panel: 6 models + naive."""
    fig, axes = plt.subplots(len(REGIMES), len(FAMS),
                              figsize=(4.0 * len(FAMS), 2.8 * len(REGIMES)),
                              sharex=True, sharey=True)
    legend_set = set()
    all_handles = {}
    for i, reg in enumerate(REGIMES):
        for j, fam in enumerate(FAMS):
            ax = axes[i, j]
            h = plot_panel(ax, fam, reg, legend_set=legend_set)
            all_handles.update(h)
            ax.set_yscale("log")
            ax.grid(linestyle="--", alpha=0.3)
            if i == 0:
                ax.set_title(FAM_LABELS[fam])
            if j == 0:
                ax.set_ylabel(f"{REGIME_LABELS[reg]}\nrel$L_2$")
            if i == len(REGIMES) - 1:
                ax.set_xlabel("rollout step t")
    fig.suptitle("Per-step rel$L_2$ — scalar-DDE baselines (Option A: regime × family grid)",
                  y=0.995)
    fig.legend(list(all_handles.values()), list(all_handles.keys()),
                loc="lower center", bbox_to_anchor=(0.5, -0.02),
                ncol=min(len(all_handles), 7), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = FIG_DIR / "F06b_scalar_optionA.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def option_B():
    """1 row x 3 cols (family). Linestyle encodes regime, color encodes model."""
    fig, axes = plt.subplots(1, len(FAMS), figsize=(4.6 * len(FAMS), 4.0),
                              sharey=True)
    if len(FAMS) == 1:
        axes = [axes]
    model_handles, regime_handles = {}, {}
    for j, fam in enumerate(FAMS):
        ax = axes[j]
        for reg in REGIMES:
            ls = REGIME_LS[reg]
            for model in MODEL_ORDER:
                c = load_curves(fam, reg, model)
                if c is None:
                    continue
                steps = np.arange(c.shape[1])
                line, = ax.plot(steps, c.mean(axis=0),
                                color=MODEL_COLOR[model],
                                linestyle=ls, lw=1.1, alpha=0.85)
                if MODEL_LABELS[model] not in model_handles:
                    model_handles[MODEL_LABELS[model]] = plt.Line2D(
                        [0], [0], color=MODEL_COLOR[model], lw=1.5,
                        label=MODEL_LABELS[model])
            if REGIME_LABELS[reg] not in regime_handles:
                regime_handles[REGIME_LABELS[reg]] = plt.Line2D(
                    [0], [0], color="black", lw=1.5,
                    linestyle=ls, label=REGIME_LABELS[reg])
        n = naive_curve(fam, "clean")
        if n is not None:
            ax.plot(np.arange(len(n)), n, color="grey", lw=0.9,
                    linestyle=(0, (1, 1)), alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("rollout step t")
        ax.set_title(FAM_LABELS[fam])
        ax.grid(linestyle="--", alpha=0.3)
    axes[0].set_ylabel(r"per-step rel$L_2$")
    leg1 = fig.legend(list(model_handles.values()),
                       list(model_handles.keys()),
                       loc="lower center",
                       bbox_to_anchor=(0.30, -0.06),
                       ncol=3, frameon=False, fontsize=9, title="Model")
    leg2 = fig.legend(list(regime_handles.values()),
                       list(regime_handles.keys()),
                       loc="lower center",
                       bbox_to_anchor=(0.75, -0.06),
                       ncol=3, frameon=False, fontsize=9, title="Regime")
    fig.add_artist(leg1)
    fig.suptitle("Per-step rel$L_2$ — scalar-DDE baselines (Option B: regime as linestyle)",
                  y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = FIG_DIR / "F06b_scalar_optionB.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def option_C():
    """1 row x 3 cols (regime). Families averaged inside each panel.
    History block trimmed; x-axis is forecast step from 1.
    """
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(4.2 * len(REGIMES), 3.4),
                              sharey=True)
    legend_set = set()
    all_handles = {}
    for j, reg in enumerate(REGIMES):
        ax = axes[j]
        for model in MODEL_ORDER:
            stacks = []
            for fam in FAMS:
                c = load_curves(fam, reg, model)
                if c is not None:
                    stacks.append(c.mean(axis=0))
            if not stacks:
                continue
            L = min(len(s) for s in stacks)
            arr = np.stack([s[:L] for s in stacks], axis=0)
            steps = np.arange(1, L + 1)
            color = MODEL_COLOR[model]
            line, = ax.plot(steps, arr.mean(axis=0), color=color, lw=1.6,
                            label=MODEL_LABELS[model])
            ax.fill_between(steps, arr.min(axis=0), arr.max(axis=0),
                            color=color, alpha=0.12, linewidth=0)
            if MODEL_LABELS[model] not in legend_set:
                all_handles[MODEL_LABELS[model]] = line
                legend_set.add(MODEL_LABELS[model])
        naives = [naive_curve(fam, reg) for fam in FAMS]
        naives = [n for n in naives if n is not None]
        if naives:
            L = min(len(n) for n in naives)
            n_arr = np.stack([n[:L] for n in naives], axis=0)
            line, = ax.plot(np.arange(1, L + 1), n_arr.mean(axis=0),
                            color="dimgrey", lw=1.1,
                            linestyle="--", label="naive copy")
            if "naive copy" not in legend_set:
                all_handles["naive copy"] = line
                legend_set.add("naive copy")
        ax.set_yscale("log")
        ax.set_xlabel("forecast step")
        ax.set_title(REGIME_LABELS[reg], fontsize=11)
        ax.grid(linestyle=":", alpha=0.4, linewidth=0.6)
        ax.set_xlim(left=1)
        ax.margins(x=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel(r"rel$L_2$")
    fig.legend(list(all_handles.values()), list(all_handles.keys()),
                loc="lower center", bbox_to_anchor=(0.5, -0.04),
                ncol=min(len(all_handles), 7), frameon=False, fontsize=10,
                handlelength=2.2, columnspacing=1.5)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = FIG_DIR / "F06b_scalar_baselines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    # Option C is the chosen layout (regime panels, families averaged with min/max band).
    # option_A (regime x family grid) and option_B (regime as linestyle) are kept
    # in this file for reference / future use (e.g. when porting this layout to
    # the distributed-kernel families) — disabled by default to avoid stale output.
    out = option_C()
    print(f"  option_C -> {out.name}")
    print(f"[layouts] generated in {FIG_DIR}")
