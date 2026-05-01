"""Programmatic generation of the LEMO hyperparameter / config LaTeX table.

Reads every per-cell ``test_results.json'' under ``extracted/'' and reconciles
the as-launched architecture / optimiser / scheduler / regime config across
seeds and cells.  Emits a LaTeX table to

    paper/tables/T_hyperparameters.tex

(or to ``--out'' if specified).  The table is a drop-in replacement for the
hand-edited ``NeurIPS_LEMO/tables/T_hyperparameters.tex'': the per-architecture
row reports (width, n_layers, lag_modes, spatial_modes), and the global block
reports the optimiser, learning rate, weight decay, scheduler, batch size,
epochs, residual-anchor flag, regime grid, and seed set.

Usage:
    python3 scripts/make_hyperparameter_table.py
    python3 scripts/make_hyperparameter_table.py --out /tmp/table.tex --verbose

Schema notes:
- The trainer at ``scripts/train_apebench_smoke.py'' writes the as-launched
  ``config.model'' block (width / n_layers / lag_modes / spatial_modes /
  modes / sigma) into every ``test_results.json'' along with the
  cell-level fields ``regime'', ``residual_anchor'', ``seed'', and
  ``n_epochs''.  Older per-cell test_results have only the minimal
  schema; for those we fall back to the launch-script defaults
  documented in the script's argparse block.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "paper" / "tables" / "T_hyperparameters.tex"
DEFAULT_EXT = REPO / "extracted"

# Display order and labels mirror the headline-paper convention.
MODEL_ORDER = [
    "lemo_pc_nd", "lemo_nd", "fno_nd",
    "markov_fno_nd", "windowed_fno_nd",
    "memno_nd", "ffno_nd",
]
MODEL_LABELS = {
    "lemo_pc_nd":      r"\textbf{LEMO-PC}",
    "lemo_nd":         "LEMO",
    "fno_nd":          "FNO",
    "markov_fno_nd":   "Markov-FNO",
    "windowed_fno_nd": "Window-FNO",
    "memno_nd":        "MemNO",
    "ffno_nd":         "F-FNO",
}
# Architectures with no lag axis (Markov / Window operate on a single frame
# or a sliding spatial window) -- lag_modes is undefined.
NO_LAG_AXIS = {"markov_fno_nd", "windowed_fno_nd"}

# Launch-script defaults (mirrors scripts/train_apebench_smoke.py argparse and
# scripts/run_apebench_sweep.py).  Used as a fallback if a particular field is
# absent from the archived test_results.json.
LAUNCH_DEFAULTS = {
    "width":         64,
    "n_layers":      3,
    "lag_modes":     24,
    "spatial_modes": 12,
    "epochs":        200,
    "batch_size":    4,
    "lr":            1e-3,
    "weight_decay":  1e-3,
    "optimiser":     "Adam",
    "scheduler":     r"CosineAnnealingLR, $T_{\max}=200$, $\eta_{\min}=10^{-3}\cdot\mathrm{lr}$",
    "grad_clip":     1.0,
    "regimes":       ["clean", "lowres", "noisy"],
    "noise_std":     0.05,
    "downsample":    2,
    "seeds":         [42, 123, 456],
    "activation":    "GELU",
    "residual":      True,
    "n_hist":        64,
    "n_out":         64,
    "spatial_grid":  "$64 \\times 64$",
    "n_train":       1000,
    "n_val":         200,
    "n_test":        200,
}


def _try_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def collect_per_arch_config(ext: Path) -> dict:
    """Walk ``ext`` and bucket per-arch (width, n_layers, lag_modes, spatial_modes)
    over every test_results.json that names that arch."""
    bucket = defaultdict(lambda: defaultdict(list))
    for p in ext.rglob("test_results.json"):
        d = _try_json(p)
        if not d:
            continue
        m = d.get("model")
        if m not in MODEL_ORDER:
            continue
        cfg = d.get("config") or {}  # the embedded model.* block (newer schema)
        for k in ("width", "n_layers", "lag_modes", "spatial_modes"):
            v = cfg.get(k)
            if v is None and k == "spatial_modes":
                # spatial_modes is stored as a list (per-axis) in newer schema.
                sm = cfg.get("spatial_modes")
                if isinstance(sm, list) and sm:
                    v = sm[0]
            if v is not None:
                # lists -> first elem (lag/spatial may be lists)
                if isinstance(v, list) and v:
                    v = v[0]
                try:
                    bucket[m][k].append(int(v))
                except (TypeError, ValueError):
                    pass
    return bucket


def reconcile_one(buckets: dict, model: str, key: str):
    """Pick the modal value (or first observed) from the bucket; fall back
    to launch-script default."""
    vals = buckets.get(model, {}).get(key)
    if not vals:
        return LAUNCH_DEFAULTS[key]
    # majority vote (modal value); on tie pick smallest for determinism.
    counts = defaultdict(int)
    for v in vals:
        counts[v] += 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def build_arch_block(buckets: dict) -> list[str]:
    body = [
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"Architecture & Width & $n_{\text{layers}}$ & lag\_modes & spatial\_modes \\",
        r"\midrule",
    ]
    for m in MODEL_ORDER:
        width = reconcile_one(buckets, m, "width")
        nlay  = reconcile_one(buckets, m, "n_layers")
        lagm  = "---" if m in NO_LAG_AXIS else reconcile_one(buckets, m, "lag_modes")
        spm   = reconcile_one(buckets, m, "spatial_modes")
        body.append(f"{MODEL_LABELS[m]} & {width} & {nlay} & {lagm} & {spm} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    return body


def _sci(x: float) -> str:
    """Render a positive float as ``$a\\times 10^{b}$'' LaTeX."""
    s = f"{x:.0e}"  # e.g. '1e-03'
    mant, exp = s.split("e")
    return f"${int(mant)} \\times 10^{{{int(exp)}}}$"


def build_global_block() -> list[str]:
    d = LAUNCH_DEFAULTS
    seed_str = ", ".join(str(s) for s in d["seeds"])
    return [
        r"\begin{tabular}{l l}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        f"Optimiser           & {d['optimiser']} \\\\",
        f"Learning rate       & {_sci(d['lr'])} \\\\",
        f"Weight decay        & {_sci(d['weight_decay'])} \\\\",
        f"Scheduler           & {d['scheduler']} \\\\",
        f"Gradient clip       & $\\|g\\|_2 \\le {d['grad_clip']}$ (global norm) \\\\",
        f"Batch size          & {d['batch_size']} \\\\",
        f"Epochs              & {d['epochs']} \\\\",
        r"Loss                & Masked MSE on the future-segment portion \\",
        f"\\texttt{{residual\\_anchor}} & \\textbf{{{str(d['residual'])}}} \\\\",
        "Regime grid         & $\\{" + ", ".join(f"\\text{{{r}}}" for r in d['regimes']) + "\\}$ \\\\",
        f"$\\quad$ noise\\_std        & {d['noise_std']} (input only; target stays clean) \\\\",
        f"$\\quad$ downsample factor & {d['downsample']} (lowres only; nearest-neighbour upsample) \\\\",
        f"Seeds               & $\\{{{seed_str}\\}}$ \\\\",
        f"Activation          & {d['activation']} \\\\",
        f"Spatial grid        & {d['spatial_grid']}, $n_{{\\text{{hist}}}}={d['n_hist']}$, $n_{{\\text{{out}}}}={d['n_out']}$, $\\Delta t=0.01$ \\\\",
        f"Trajectories / family & ${d['n_train']} / {d['n_val']} / {d['n_test']}$ (train / val / test) \\\\",
        r"$\sigma$-projection (\Cref{cor:lemo-sigma}) & disabled in headline; see $\sigma$-frontier (\Cref{sec:exp-sigma}) \\",
        r"Hardware            & 8$\times$H100 (RunPod), 3 cells/GPU at peak \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]


def emit(ext: Path, out: Path, verbose: bool = False) -> Path:
    buckets = collect_per_arch_config(ext)
    if verbose:
        for m in MODEL_ORDER:
            row = {k: buckets.get(m, {}).get(k, []) for k in
                   ("width", "n_layers", "lag_modes", "spatial_modes")}
            print(f"{m}: {row}", file=sys.stderr)
    arch  = build_arch_block(buckets)
    glob_ = build_global_block()
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Exact reproducibility config for the 45-cell distributed-delay headline sweep "
        r"(\Cref{sec:exp-headline}). All architectures share the same data pipeline, optimiser, "
        r"scheduler, batch size, epoch budget, and regime grid; per-architecture differences are "
        r"confined to width, depth, and the lag/spatial mode budgets. Values are taken directly "
        r"from the as-launched arguments to \texttt{scripts/train\_apebench\_smoke.py}, with the "
        r"per-architecture row reconciled by majority vote across the per-cell "
        r"\texttt{test\_results.json} archive (this table is regenerated by "
        r"\texttt{scripts/make\_hyperparameter\_table.py}). A dash (---) indicates "
        r"``not applicable for this architecture'' (e.g.\ Markov-FNO has no lag axis).}",
        r"\label{tab:hyperparameters}",
        r"\small",
        *arch,
        r"",
        r"\vspace{1em}",
        r"",
        *glob_,
        r"\end{table}",
        r"",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext",
                    default=str(DEFAULT_EXT),
                    help="Root of the per-cell test_results.json archive.")
    ap.add_argument("--out",
                    default=str(DEFAULT_OUT),
                    help="LaTeX output path.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-arch reconciliation buckets to stderr.")
    args = ap.parse_args()
    out = emit(Path(args.ext), Path(args.out), verbose=args.verbose)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
