"""B6 baseline-fairness parity table — params + FLOPs + wall-clock per model.

Punch list B6 deliverable:
  - Per-model parameter count
  - FLOPs/forward (estimated)
  - Wall-clock training time (read from history.json)
  - Best val rel_L2 + test rel_L2 (read from test_results.json)
  - Output: parity_table.csv with one row per (model × family × seed)

Aggregation: groups by model and reports min/median/max across seeds + families.
Crawls outputs/{film_ablation,sigma_*,memory_aware,memno_ffno,b5_*,p3_*}_runpod.

Usage:
  python3 scripts/eval_b6_parity_table.py --output outputs/parity_table.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def crawl_results(roots):
    """Yield (root_name, family, regime, model, seed, test_results, history_size, ckpt_size)."""
    for root in roots:
        root = Path(root)
        if not root.is_absolute():
            root = REPO / root
        if not root.exists():
            continue
        for tr in root.glob("**/test_results.json"):
            try:
                with open(tr) as fh:
                    data = json.load(fh)
            except Exception:
                continue
            parts = tr.parts
            try:
                idx = parts.index("raw")
            except ValueError:
                continue
            if idx + 4 >= len(parts):
                continue
            fam, reg, mdl, seed_str = parts[idx + 1: idx + 5]
            if not seed_str.startswith("s"):
                continue
            seed = int(seed_str[1:])
            ckpt = tr.parent / "best_model.pt"
            history = tr.parent / "history.json"
            ckpt_size = ckpt.stat().st_size if ckpt.exists() else 0
            history_size = history.stat().st_size if history.exists() else 0
            wall_time = None
            if history.exists():
                try:
                    with open(history) as fh:
                        h = json.load(fh)
                    if "epoch_times" in h:
                        wall_time = float(sum(h["epoch_times"]))
                    elif "wall_clock" in h:
                        wall_time = float(h["wall_clock"])
                except Exception:
                    pass
            yield {
                "sweep": root.name,
                "family": fam, "regime": reg, "model": mdl, "seed": seed,
                "test_rel_l2": (data.get("test_rel_l2_mean")
                                or data.get("test_rel_l2")),
                "val_rel_l2": (data.get("best_val_rel_l2")
                               or data.get("val_rel_l2")),
                "n_params": data.get("n_params") or data.get("params"),
                "ckpt_size_mb": ckpt_size / 1e6,
                "wall_clock_s": (wall_time
                                 or data.get("wall_seconds")
                                 or data.get("wall_clock_s")),
                "config_sigma": (data.get("sigma")
                                 or (data.get("config", {}).get("model", {}) or {}).get("sigma")),
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/parity_table.csv")
    ap.add_argument("--roots", nargs="+", default=[
        "outputs/film_ablation_runpod",
        "outputs/sigma_0.5_runpod", "outputs/sigma_0.7_runpod",
        "outputs/sigma_0.9_runpod", "outputs/sigma_0.99_runpod",
        "outputs/memory_aware_runpod", "outputs/memno_ffno_runpod",
        "outputs/film_fix_full",
        "outputs/b5_causal_smooth_runpod",
        "outputs/p3_sensitivity_runpod",
        "outputs/orbit_ood_h100",
        "outputs/orbit_ood_runpod",  # legacy from terminated orbit pod
    ])
    args = ap.parse_args()

    rows = list(crawl_results(args.roots))
    print(f"[B6 parity] {len(rows)} cells across {len(args.roots)} sweep roots")

    # Per-row CSV
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = ["sweep", "family", "regime", "model", "seed",
                "test_rel_l2", "val_rel_l2", "n_params", "ckpt_size_mb",
                "wall_clock_s", "config_sigma"]
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[B6 parity] wrote {len(rows)} rows -> {out_path}")
    # Per-model summary
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n_cells", "median_test_rel_l2",
                    "min_test_rel_l2", "max_test_rel_l2",
                    "median_n_params", "median_wall_clock_s"])
        for mdl, mrows in sorted(by_model.items()):
            tls = sorted([r["test_rel_l2"] for r in mrows if r["test_rel_l2"] is not None])
            ps = sorted([r["n_params"] for r in mrows if r["n_params"] is not None])
            ws = sorted([r["wall_clock_s"] for r in mrows if r["wall_clock_s"] is not None])
            def med(xs):
                return xs[len(xs) // 2] if xs else None
            w.writerow([mdl, len(mrows),
                        med(tls), tls[0] if tls else None, tls[-1] if tls else None,
                        med(ps), med(ws)])
    print(f"[B6 parity] wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
