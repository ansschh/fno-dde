#!/usr/bin/env python3
"""Aggregate Phase A test_results.json files into a summary table."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/phase_a_theorem_suite_v1")
    p.add_argument("--metric", default="rel_l2_normalized_median")
    args = p.parse_args()

    root = Path(args.root)
    rows = []
    for ds_dir in sorted(d for d in root.iterdir()
                          if d.is_dir() and d.name in ("t1", "t2", "t3")):
        for run_dir in sorted(ds_dir.iterdir()):
            for ts_dir in run_dir.iterdir():
                tr = ts_dir / "test_results.json"
                if tr.exists():
                    m = json.loads(tr.read_text())
                    parts = run_dir.name.rsplit("_s", 1)
                    rows.append((ds_dir.name, parts[0], int(parts[1]),
                                  m[args.metric]))

    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for ds, model, seed, v in rows:
        agg[(ds, model)].append(v)

    print(f"{len(rows)} runs aggregated  (metric: {args.metric})")
    print()
    print(f"{'dataset':<4}  {'model':<20}  {'median':<12}  {'min':<12}  {'max':<12}")
    print("-" * 75)
    for (ds, model), vals in sorted(agg.items()):
        vs = sorted(vals)
        print(f"{ds:<4}  {model:<20}  {median(vs):<12.4e}  {vs[0]:<12.4e}  {vs[-1]:<12.4e}")


if __name__ == "__main__":
    main()
