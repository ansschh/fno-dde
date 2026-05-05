"""F08 redesign: three layouts to choose between.

  F08a_equivariance_filled_band.{pdf,png}  — two architectural-group bands
  F08b_equivariance_lines_clean.{pdf,png}  — clean per-model lines + std bands
  F08c_equivariance_strip.{pdf,png}        — horizontal strip plot, sorted

Reads the same JSONs as fig08 in make_paper_figures.py:
  - equivariance_dense.json (k grid {1,2,4,6,8,12,16,24,32,48,64}, e_orbit.<k>.mean)
  - equivariance.json (legacy sparse k={1,4,16}, equiv_shift_<k>_mean)
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")

# Global Times New Roman style for all paper figures.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _figstyle  # noqa: F401
import matplotlib.pyplot as plt

REPO = Path(r"A:\dde research\dde-fno")
FIG_DIR = (REPO.parent / "NeurIPS_LEMO" / "figures").resolve()

# Architectural grouping for F08a (filled-band).
EQUIV_GROUP = {
    "lemo_pc_nd",
    "lemo_bcorrect_nd",
    "fno_film_nd",
    "causal_smooth_lemo_pc_nd",
}
# Everything else lands in "delay-blind / non-equivariant".

MODEL_LABEL = {
    "lemo_pc_nd": "LEMO-PC",
    "lemo_bcorrect_nd": "LEMO (b-correct)",
    "fno_film_nd": "FNO+FiLM",
    "causal_smooth_lemo_pc_nd": "LEMO-PC (causal)",
    "noneq_film_nd": "Non-equiv +FiLM",
    "memno_nd": "MemNO",
    "ffno_nd": "F-FNO",
    "s4_nd": "S4",
    "nide_nd": "NIDE",
    "ndde_nd": "NDDE",
    "fno_nd": "FNO",
    "markov_fno_nd": "Markov-FNO",
    "windowed_fno_nd": "Window-FNO",
    "unet_nd": "UNet",
}
MODEL_COLOR = {
    "lemo_pc_nd": "#d62728",
    "lemo_bcorrect_nd": "#bcbd22",
    "fno_film_nd": "#17becf",
    "causal_smooth_lemo_pc_nd": "#c49c94",
    "noneq_film_nd": "#c5b0d5",
    "memno_nd": "#e377c2",
    "ffno_nd": "#8c564b",
    "s4_nd": "#9bba2c",
    "nide_nd": "#aec7e8",
    "ndde_nd": "#98df8a",
    "fno_nd": "#1f77b4",
    "markov_fno_nd": "#2ca02c",
    "windowed_fno_nd": "#9467bd",
    "unet_nd": "#7f7f7f",
}
MODEL_ORDER = list(MODEL_LABEL.keys())


def discover(name):
    """Discover equivariance JSON files, preferring fresh sources for lemo_pc_nd.

    The post-A-fix re-eval (h100_pull, May 2026) uses cyclic_shift_full per the
    T1 theorem. Older pulls used cyclic_shift_state_only and are stale. Sort
    candidate paths so h100_pull / a_fix_runpod are seen first."""
    out = defaultdict(dict)
    seen = set()
    candidates = []
    for r in (REPO / "extracted", REPO / "outputs"):
        if not r.exists():
            continue
        for p in r.rglob(name):
            candidates.append(p)
    # Prefer h100_pull / a_fix_runpod first (fresh full-roll equivariance).
    def priority(p):
        s = str(p).replace("\\", "/")
        if "h100_pull" in s or "a_fix_runpod" in s:
            return 0
        return 1
    candidates.sort(key=priority)
    for p in candidates:
        parts = p.parts
        if len(parts) < 5:
            continue
        seed = parts[-2]; m = parts[-3]; reg = parts[-4]; fam = parts[-5]
        key = (m, fam, reg, seed)
        if key in seen:
            continue
        seen.add(key)
        try:
            out[m][(fam, reg, seed)] = json.loads(p.read_text())
        except Exception:
            pass
    return out


def rows_for(model, sparse, dense):
    """Returns list of (k, err) tuples pooled across cells."""
    rows = []
    for d in dense.get(model, {}).values():
        e = d.get("e_orbit", {})
        for k_str, stats in e.items():
            if not isinstance(stats, dict):
                continue
            v = stats.get("mean")
            if v is None:
                continue
            try:
                rows.append((int(k_str), float(v)))
            except (TypeError, ValueError):
                pass
    for d in sparse.get(model, {}).values():
        for key, v in d.items():
            m = re.match(r"equiv_shift_(\d+)_mean", key)
            if not m or v is None:
                continue
            try:
                rows.append((int(m.group(1)), float(v)))
            except (TypeError, ValueError):
                pass
    return rows


def aggregate_per_model(rows):
    """Group by k. Returns shifts (sorted), means, stds."""
    by_k = defaultdict(list)
    for k, v in rows:
        by_k[k].append(v)
    shifts = sorted(by_k.keys())
    means = np.array([np.mean(by_k[k]) for k in shifts])
    stds = np.array([np.std(by_k[k]) for k in shifts])
    return np.asarray(shifts), means, stds


def f08a_filled_band(per_model):
    """Two architectural-group bands. Min-max envelope per group, mean line on top."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    GROUPS = {
        "Cyclic-FFT (T1-equivariant)": (
            [m for m in MODEL_ORDER if m in EQUIV_GROUP and m in per_model],
            "#2ca02c",
        ),
        "Delay-blind / non-equivariant": (
            [m for m in MODEL_ORDER if m not in EQUIV_GROUP and m in per_model],
            "#d62728",
        ),
    }
    handles = []
    for label, (members, color) in GROUPS.items():
        if not members:
            continue
        # Collect (k, err) across all members.
        all_shifts = sorted({k for m in members for k in per_model[m]["shifts"]})
        if not all_shifts:
            continue
        # Compute envelope.
        envelope_lo = np.full(len(all_shifts), np.nan)
        envelope_hi = np.full(len(all_shifts), np.nan)
        means_per_k = np.full(len(all_shifts), np.nan)
        for i, k in enumerate(all_shifts):
            vals = []
            for m in members:
                shifts = per_model[m]["shifts"]
                means = per_model[m]["means"]
                if k in shifts:
                    j = list(shifts).index(k)
                    vals.append(means[j])
            if not vals:
                continue
            envelope_lo[i] = np.min(vals)
            envelope_hi[i] = np.max(vals)
            means_per_k[i] = np.mean(vals)
        valid = ~np.isnan(means_per_k)
        if not valid.any():
            continue
        x = np.array(all_shifts)[valid]
        ax.fill_between(x, envelope_lo[valid], envelope_hi[valid],
                         color=color, alpha=0.18, linewidth=0)
        line, = ax.plot(x, means_per_k[valid], color=color, lw=2.0, label=label)
        handles.append(line)

    # FP32 floor band.
    ax.axhspan(1e-7, 1e-6, color="grey", alpha=0.12, linewidth=0)
    ax.text(0.985, 0.03, "FP32 FFT floor",
             color="black", fontsize=14, ha="right", va="bottom",
             style="italic", transform=ax.transAxes)

    ax.set_yscale("log")
    ax.set_xlabel("cyclic shift size $k$")
    ax.set_ylabel(r"$\| f(\rho_k x) - \rho_k f(x) \|_2 / \|f(x)\|_2$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=len(handles), frameon=False)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.28)
    out = FIG_DIR / "F08a_equivariance_filled_band.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return out


def f08b_lines_clean(per_model):
    """Clean line plot: thin mean line + soft std fill, no markers."""
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    handles = []
    for m in MODEL_ORDER:
        if m not in per_model:
            continue
        shifts = per_model[m]["shifts"]
        means = per_model[m]["means"]
        stds = per_model[m]["stds"]
        if len(shifts) == 0:
            continue
        color = MODEL_COLOR.get(m, "#888")
        line, = ax.plot(shifts, means, color=color, lw=1.8,
                          label=MODEL_LABEL.get(m, m))
        ax.fill_between(shifts,
                          np.maximum(means - stds, 1e-12),
                          means + stds,
                          color=color, alpha=0.10, linewidth=0)
        handles.append(line)
    ax.axhspan(1e-7, 1e-6, color="grey", alpha=0.12, linewidth=0)
    ax.text(0.985, 0.03, "FP32 FFT floor",
             color="black", fontsize=14, ha="right", va="bottom",
             style="italic", transform=ax.transAxes)
    ax.set_yscale("log")
    ax.set_xlabel("cyclic shift size $k$")
    ax.set_ylabel(r"$\| f(\rho_k x) - \rho_k f(x) \|_2 / \|f(x)\|_2$")
    n = len(handles)
    ncol = 4 if n > 6 else (3 if n > 3 else max(1, n))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=ncol, frameon=False,
               columnspacing=1.4, handlelength=1.8, handletextpad=0.5)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.34)
    out = FIG_DIR / "F08b_equivariance_lines_clean.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return out


def f08c_strip(per_model):
    """Horizontal strip plot: y-axis = model (sorted by mean error), x-axis = log error.

    For each model, draw the distribution of equivariance errors across all
    (k, family, seed, regime) cells as a soft horizontal strip plus median + IQR.
    """
    # Collect all raw values per model.
    raw = {}
    for m in MODEL_ORDER:
        if m not in per_model:
            continue
        rows = per_model[m]["rows"]
        if not rows:
            continue
        vals = np.array([r[1] for r in rows], dtype=float)
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        raw[m] = vals
    # Sort models by median error ascending.
    order = sorted(raw.keys(), key=lambda m: np.median(raw[m]))
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(order) + 1.2))
    yticks, ylabels = [], []
    for i, m in enumerate(order):
        vals = raw[m]
        log_vals = np.log10(vals)
        color = MODEL_COLOR.get(m, "#888")
        # Faint scatter strip.
        ax.scatter(log_vals, np.full_like(log_vals, i),
                    color=color, alpha=0.30, s=10, linewidth=0)
        # Median + IQR overlay.
        med = np.median(log_vals)
        q25, q75 = np.percentile(log_vals, [25, 75])
        ax.plot([q25, q75], [i, i], color=color, lw=2.2)
        ax.scatter([med], [i], color=color, s=46, zorder=5, edgecolor="white", linewidth=0.6)
        yticks.append(i)
        ylabels.append(MODEL_LABEL.get(m, m))
    # FP32 floor reference.
    ax.axvspan(-7, -6, color="grey", alpha=0.12, linewidth=0)
    ax.text(-6.5, len(order) - 0.4, "FP32 FFT floor",
             color="black", fontsize=9, ha="center", va="center",
             style="italic")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\log_{10}\;\| f(\rho_k x) - \rho_k f(x) \|_2 / \|f(x)\|_2$")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "F08c_equivariance_strip.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return out


def main():
    sparse = discover("equivariance.json")
    dense = discover("equivariance_dense.json")
    per_model = {}
    for m in MODEL_ORDER:
        rows = rows_for(m, sparse, dense)
        if not rows:
            continue
        shifts, means, stds = aggregate_per_model(rows)
        per_model[m] = {"shifts": list(shifts), "means": means,
                         "stds": stds, "rows": rows}
    print(f"[F08-redesigns] models with data: {len(per_model)}")
    for m in per_model:
        print(f"  {m}: {len(per_model[m]['shifts'])} shifts, {len(per_model[m]['rows'])} rows")
    if not per_model:
        return
    out_a = f08a_filled_band(per_model)
    print(f"  -> {out_a.name}")
    out_b = f08b_lines_clean(per_model)
    print(f"  -> {out_b.name}")
    out_c = f08c_strip(per_model)
    print(f"  -> {out_c.name}")


if __name__ == "__main__":
    main()
